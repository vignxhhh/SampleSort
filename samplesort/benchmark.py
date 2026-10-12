"""Run N seeded sim trials and report the metrics as JSON and Markdown.

Each trial is a fresh world built from its own seed, so the whole benchmark is
reproducible: the same ``--trials`` and starting seed always produce the same
layouts and therefore the same numbers.

Reported metrics, per spec section 5.10:

* **Sorting accuracy** — of the tubes that were placed, how many landed in the
  rack their *true* class belongs in. This catches misclassification, which a
  grasp-success number cannot.
* **Grasp success rate** — attempts whose grasp stage worked, counting retries.
* **Mean time per sample** — wall-clock seconds per successfully sorted tube.
* **Failures by type** — the :class:`~samplesort.control.scripted.FailureReason`
  taxonomy, plus the planner's skip reasons.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from samplesort.config import SampleSortConfig
from samplesort.hal.factory import build_backend
from samplesort.logging_db.sort_log import SortLog
from samplesort.pipeline import JobOutcome, PipelineReport, SortPipeline

logger = logging.getLogger(__name__)

#: Default filenames written into the output directory.
METRICS_FILENAME = "benchmark.json"
TABLE_FILENAME = "benchmark.md"


@dataclass
class TrialResult:
    """Metrics from a single benchmark trial.

    Attributes:
        seed: The world seed this trial used.
        mode: Which controller ran it.
        tubes: How many tubes were spawned.
        attempted: How many jobs were attempted.
        succeeded: How many jobs the controller reported as successful.
        correctly_sorted: How many tubes reached the rack their true class maps to.
        grasped: How many attempts got as far as a successful grasp.
        total_attempts: Attempts including retries.
        duration_s: Wall-clock seconds for the trial.
        mean_time_per_sample_s: Seconds per successfully sorted tube.
        failures: Failure reason counts.
        skipped: Planner skip reason counts.
    """

    seed: int
    mode: str
    tubes: int
    attempted: int
    succeeded: int
    correctly_sorted: int
    grasped: int
    total_attempts: int
    duration_s: float
    mean_time_per_sample_s: float
    failures: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def sorting_accuracy(self) -> float:
        """Fraction of *placed* tubes that landed in the correct rack."""
        return self.correctly_sorted / self.succeeded if self.succeeded else 0.0

    @property
    def grasp_success_rate(self) -> float:
        """Fraction of grasp attempts, retries included, that picked a tube up."""
        return self.grasped / self.total_attempts if self.total_attempts else 0.0

    @property
    def completion_rate(self) -> float:
        """Fraction of spawned tubes that were successfully sorted."""
        return self.succeeded / self.tubes if self.tubes else 0.0


@dataclass
class BenchmarkReport:
    """Aggregated results across every trial.

    Attributes:
        mode: Which controller was benchmarked.
        trials: Per-trial results.
        tubes_per_trial: How many tubes each trial spawned.
        perception_noise_m: Localisation noise injected into each detection.
        started_at: ISO-8601 UTC timestamp of the run.
        total_duration_s: Wall-clock seconds for the whole benchmark.
        environment: Python and platform versions, for reproducibility.
    """

    mode: str
    trials: list[TrialResult]
    tubes_per_trial: int
    perception_noise_m: float
    started_at: str
    total_duration_s: float
    environment: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- aggregates

    @property
    def num_trials(self) -> int:
        """How many trials ran."""
        return len(self.trials)

    @property
    def total_tubes(self) -> int:
        """Total tubes spawned across every trial."""
        return sum(t.tubes for t in self.trials)

    @property
    def total_succeeded(self) -> int:
        """Total tubes successfully sorted."""
        return sum(t.succeeded for t in self.trials)

    @property
    def total_correct(self) -> int:
        """Total tubes that reached the rack their true class maps to."""
        return sum(t.correctly_sorted for t in self.trials)

    @property
    def sorting_accuracy(self) -> float:
        """Correct-rack rate across every placed tube."""
        return self.total_correct / self.total_succeeded if self.total_succeeded else 0.0

    @property
    def grasp_success_rate(self) -> float:
        """Grasp success rate across every attempt, retries included."""
        attempts = sum(t.total_attempts for t in self.trials)
        return sum(t.grasped for t in self.trials) / attempts if attempts else 0.0

    @property
    def completion_rate(self) -> float:
        """Fraction of all spawned tubes that were sorted."""
        return self.total_succeeded / self.total_tubes if self.total_tubes else 0.0

    @property
    def mean_time_per_sample_s(self) -> float:
        """Mean seconds per successfully sorted tube, across every trial."""
        times = [t.mean_time_per_sample_s for t in self.trials if t.succeeded]
        return statistics.fmean(times) if times else 0.0

    @property
    def stdev_time_per_sample_s(self) -> float:
        """Standard deviation of per-trial mean times, or 0.0 with fewer than two."""
        times = [t.mean_time_per_sample_s for t in self.trials if t.succeeded]
        return statistics.stdev(times) if len(times) > 1 else 0.0

    def failures_by_type(self) -> dict[str, int]:
        """Failure counts summed across every trial, highest first."""
        totals: dict[str, int] = {}
        for trial in self.trials:
            for reason, count in trial.failures.items():
                totals[reason] = totals.get(reason, 0) + count
        return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    def skips_by_type(self) -> dict[str, int]:
        """Planner skip counts summed across every trial, highest first."""
        totals: dict[str, int] = {}
        for trial in self.trials:
            for reason, count in trial.skipped.items():
                totals[reason] = totals.get(reason, 0) + count
        return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))

    # -------------------------------------------------------------- rendering

    def to_dict(self) -> dict[str, object]:
        """Serialise the report, aggregates included, for the JSON artefact."""
        return {
            "mode": self.mode,
            "started_at": self.started_at,
            "num_trials": self.num_trials,
            "tubes_per_trial": self.tubes_per_trial,
            "perception_noise_m": self.perception_noise_m,
            "total_tubes": self.total_tubes,
            "total_duration_s": round(self.total_duration_s, 3),
            "environment": self.environment,
            "summary": {
                "sorting_accuracy": round(self.sorting_accuracy, 4),
                "grasp_success_rate": round(self.grasp_success_rate, 4),
                "completion_rate": round(self.completion_rate, 4),
                "mean_time_per_sample_s": round(self.mean_time_per_sample_s, 4),
                "stdev_time_per_sample_s": round(self.stdev_time_per_sample_s, 4),
                "total_succeeded": self.total_succeeded,
                "total_correct": self.total_correct,
            },
            "failures_by_type": self.failures_by_type(),
            "skips_by_type": self.skips_by_type(),
            "trials": [asdict(trial) for trial in self.trials],
        }

    def to_markdown(self, *, comparison: BenchmarkReport | None = None) -> str:
        """Render the report as a Markdown section.

        Args:
            comparison: An optional second report — typically the learned-mode
                run — rendered as an extra column beside this one.

        Returns:
            Markdown text, ready to paste into ``docs/results.md``.
        """
        reports = [self] + ([comparison] if comparison is not None else [])
        headers = " | ".join(f"{r.mode} mode" for r in reports)
        separator = " | ".join(["---:"] * len(reports))

        rows: list[tuple[str, list[str]]] = [
            ("Trials", [str(r.num_trials) for r in reports]),
            ("Tubes per trial", [str(r.tubes_per_trial) for r in reports]),
            (
                "Perception noise (σ)",
                [
                    "none" if r.perception_noise_m <= 0 else f"{r.perception_noise_m * 1000:.0f} mm"
                    for r in reports
                ],
            ),
            ("Tubes attempted", [str(r.total_tubes) for r in reports]),
            ("Sorting accuracy (correct rack)", [f"{r.sorting_accuracy:.1%}" for r in reports]),
            ("Grasp success rate", [f"{r.grasp_success_rate:.1%}" for r in reports]),
            ("Completion rate", [f"{r.completion_rate:.1%}" for r in reports]),
            (
                "Mean time per sample",
                [
                    f"{r.mean_time_per_sample_s:.2f} s ± {r.stdev_time_per_sample_s:.2f}"
                    for r in reports
                ],
            ),
            ("Total wall-clock", [f"{r.total_duration_s:.1f} s" for r in reports]),
        ]

        lines = [
            f"| Metric | {headers} |",
            f"| --- | {separator} |",
        ]
        lines += [f"| {name} | {' | '.join(values)} |" for name, values in rows]

        lines.append("")
        lines.append("**Failures by type**")
        lines.append("")
        failures = self.failures_by_type()
        if not failures:
            lines.append("No failures were recorded.")
        else:
            lines.append("| Failure | Count |")
            lines.append("| --- | ---: |")
            lines += [f"| `{reason}` | {count} |" for reason, count in failures.items()]

        skips = self.skips_by_type()
        if skips:
            lines += ["", "**Detections skipped by the planner**", "", "| Reason | Count |"]
            lines.append("| --- | ---: |")
            lines += [f"| `{reason}` | {count} |" for reason, count in skips.items()]

        return "\n".join(lines) + "\n"


def _package_version(name: str) -> str:
    """Return an installed package's version, or ``"unknown"`` if it has none."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - only in odd installs
        return "unknown"


def _environment() -> dict[str, str]:
    """Capture the versions that would change these numbers if they changed."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "opencv": _package_version("opencv-python-headless"),
        "pybullet": _package_version("pybullet"),
        "samplesort": _package_version("samplesort"),
    }


def _trial_from_report(
    report: PipelineReport, *, seed: int, tubes: int, correctly_sorted: int
) -> TrialResult:
    """Fold a pipeline report into a trial result."""
    skipped: dict[str, int] = {}
    for _, reason in report.skipped:
        skipped[reason] = skipped.get(reason, 0) + 1

    return TrialResult(
        seed=seed,
        mode=report.mode,
        tubes=tubes,
        attempted=report.attempted,
        succeeded=report.succeeded,
        correctly_sorted=correctly_sorted,
        grasped=sum(1 for o in report.outcomes if o.result.grasped),
        total_attempts=sum(o.attempts for o in report.outcomes),
        duration_s=report.duration_s,
        mean_time_per_sample_s=report.mean_duration_s,
        failures=report.failures_by_reason(),
        skipped=skipped,
    )


def run_benchmark(
    config: SampleSortConfig,
    *,
    trials: int = 20,
    num_tubes: int = 6,
    start_seed: int | None = None,
    sort_log: SortLog | None = None,
    controller_factory: object | None = None,
) -> BenchmarkReport:
    """Run ``trials`` seeded sim trials and aggregate the metrics.

    Args:
        config: The validated configuration bundle.
        trials: How many trials to run.
        num_tubes: Tubes spawned per trial.
        start_seed: First seed; trial *i* uses ``start_seed + i``. Defaults to
            ``config.seed``.
        sort_log: Where to record attempts. One is opened from the config when
            omitted, so benchmark rows land in the same log the dashboard reads.
        controller_factory: Optional callable ``(config, backend) -> controller``
            used instead of the scripted controller. This is how learned mode is
            benchmarked.

    Returns:
        The aggregated report.

    Raises:
        ValueError: If ``trials`` or ``num_tubes`` is not positive.
    """
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if num_tubes < 1:
        raise ValueError(f"num_tubes must be at least 1, got {num_tubes}")

    from samplesort.logging_db.sort_log import utc_now

    base_seed = config.seed if start_seed is None else start_seed
    owns_log = sort_log is None
    log = sort_log or SortLog(config.database_path)
    started = time.perf_counter()
    results: list[TrialResult] = []

    try:
        for index in range(trials):
            seed = base_seed + index
            backend = build_backend(config, gui=False, seed=seed)
            with backend:
                assert backend.world is not None
                backend.world.spawn_tubes(num_tubes)

                controller = None
                if controller_factory is not None:
                    controller = controller_factory(config, backend)  # type: ignore[operator]

                pipeline = SortPipeline(config, backend, sort_log=log, controller=controller)
                # Capture true labels before the run retires or moves anything.
                report = pipeline.run(run_id=f"bench-{config.pipeline.control_mode}-{seed}")
                correct = sum(
                    1
                    for outcome in report.outcomes
                    if outcome.success and _is_correct_rack(config, pipeline, outcome)
                )

            results.append(
                _trial_from_report(report, seed=seed, tubes=num_tubes, correctly_sorted=correct)
            )
            logger.info(
                "trial %d/%d (seed %d): %d/%d sorted",
                index + 1,
                trials,
                seed,
                results[-1].succeeded,
                num_tubes,
            )
    finally:
        if owns_log:
            log.close()

    return BenchmarkReport(
        mode=config.pipeline.control_mode,
        trials=results,
        tubes_per_trial=num_tubes,
        perception_noise_m=config.pipeline.perception_noise_m,
        started_at=utc_now(),
        total_duration_s=time.perf_counter() - started,
        environment=_environment(),
    )


def _is_correct_rack(config: SampleSortConfig, pipeline: SortPipeline, outcome: JobOutcome) -> bool:
    """Whether a placed tube's *true* class maps to the rack it was put in.

    This is what turns "the controller said it succeeded" into "the sample really
    is in the right rack": a misclassified tube is placed perfectly but in the
    wrong place, and only ground truth catches that.
    """
    true_label = pipeline.true_label_for(outcome.job)
    if true_label is None:
        # No ground truth available (real hardware): trust the controller.
        return True
    return config.rack_for_class(true_label) == outcome.job.rack_id


def write_report(
    report: BenchmarkReport,
    output_dir: Path | str,
    *,
    comparison: BenchmarkReport | None = None,
) -> tuple[Path, Path]:
    """Write the metrics JSON and the Markdown table.

    Args:
        report: The report to write.
        output_dir: Directory for the artefacts; created if missing.
        comparison: Optional second report to render as an extra column.

    Returns:
        The ``(json_path, markdown_path)`` that were written.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    payload = report.to_dict()
    if comparison is not None:
        payload["comparison"] = comparison.to_dict()

    json_path = directory / METRICS_FILENAME
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    markdown_path = directory / TABLE_FILENAME
    markdown_path.write_text(report.to_markdown(comparison=comparison), encoding="utf-8")

    logger.info("wrote %s and %s", json_path, markdown_path)
    return json_path, markdown_path


def _main() -> int:  # pragma: no cover - exercised through the CLI
    """Allow ``python -m samplesort.benchmark`` for quick local runs."""
    from samplesort.config import load_config

    config = load_config()
    report = run_benchmark(config, trials=5, num_tubes=6)
    sys.stdout.write(report.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
