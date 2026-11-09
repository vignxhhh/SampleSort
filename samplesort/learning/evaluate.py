"""Roll out a trained policy and report its success rate."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from samplesort.config import SampleSortConfig
from samplesort.hal.factory import build_backend
from samplesort.logging_db.sort_log import SortLog
from samplesort.pipeline import SortPipeline

logger = logging.getLogger(__name__)

#: Written into the checkpoint directory alongside the weights.
EVALUATION_FILENAME = "evaluation.json"


@dataclass
class RolloutResult:
    """The outcome of one evaluation rollout.

    Attributes:
        seed: The world seed the rollout used.
        success: Whether the tube reached its slot.
        reason: Failure reason, or ``"none"`` on success.
        duration_s: Wall-clock seconds the rollout took.
        place_error_m: Distance from the target slot, when measurable.
        class_label: The class the perception layer assigned.
    """

    seed: int
    success: bool
    reason: str
    duration_s: float
    place_error_m: float | None = None
    class_label: str = ""


@dataclass
class EvaluationReport:
    """Aggregated results across every rollout.

    Attributes:
        policy_path: The checkpoint that was evaluated.
        rollouts: Per-rollout results.
        duration_s: Wall-clock seconds for the whole evaluation.
    """

    policy_path: Path
    rollouts: list[RolloutResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def episodes(self) -> int:
        """How many rollouts ran."""
        return len(self.rollouts)

    @property
    def successes(self) -> int:
        """How many rollouts succeeded."""
        return sum(1 for r in self.rollouts if r.success)

    @property
    def success_rate(self) -> float:
        """Fraction of rollouts that succeeded, or 0.0 when none ran."""
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def mean_duration_s(self) -> float:
        """Mean seconds per rollout, or 0.0 when none ran."""
        return sum(r.duration_s for r in self.rollouts) / self.episodes if self.episodes else 0.0

    def failures_by_reason(self) -> dict[str, int]:
        """Failure counts by reason, highest first."""
        counts: dict[str, int] = {}
        for rollout in self.rollouts:
            if not rollout.success:
                counts[rollout.reason] = counts.get(rollout.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report for ``evaluation.json``."""
        return {
            "policy_path": str(self.policy_path),
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "mean_duration_s": round(self.mean_duration_s, 4),
            "duration_s": round(self.duration_s, 3),
            "failures_by_reason": self.failures_by_reason(),
            "rollouts": [asdict(r) for r in self.rollouts],
        }

    def summary_lines(self) -> list[str]:
        """Human-readable summary lines for CLI output."""
        lines = [
            f"policy       : {self.policy_path}",
            f"episodes     : {self.episodes}",
            f"successes    : {self.successes} ({self.success_rate:.0%})",
            f"mean duration: {self.mean_duration_s:.2f} s",
        ]
        failures = self.failures_by_reason()
        if failures:
            breakdown = ", ".join(f"{name}={count}" for name, count in failures.items())
            lines.append(f"failures     : {breakdown}")
        return lines


def evaluate(
    config: SampleSortConfig,
    policy_path: Path | str,
    *,
    episodes: int = 10,
    num_tubes: int = 1,
    seed: int | None = None,
    output_path: Path | str | None = None,
) -> EvaluationReport:
    """Run ``episodes`` policy rollouts in simulation and report the success rate.

    Each rollout is one tube in its own seeded world, which matches how the
    demonstrations were recorded.

    Args:
        config: The validated configuration bundle.
        policy_path: The ACT checkpoint directory to evaluate.
        episodes: How many rollouts to run.
        num_tubes: Tubes spawned per rollout.
        seed: First world seed; rollout *i* uses ``seed + i``.
        output_path: Where to write the JSON report. Defaults to
            ``<policy_path>/evaluation.json``.

    Returns:
        The aggregated report.

    Raises:
        ValueError: If ``episodes`` or ``num_tubes`` is not positive.
        TrainingError: If ``policy_path`` holds no checkpoint.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be at least 1, got {episodes}")
    if num_tubes < 1:
        raise ValueError(f"num_tubes must be at least 1, got {num_tubes}")

    from samplesort.control.learned import LearnedController
    from samplesort.learning.train import load_policy

    checkpoint = Path(policy_path)
    policy = load_policy(config, checkpoint)

    base_seed = config.seed if seed is None else seed
    started = time.perf_counter()
    report = EvaluationReport(policy_path=checkpoint)

    for index in range(episodes):
        rollout_seed = base_seed + index
        backend = build_backend(config, gui=False, seed=rollout_seed)
        with backend, SortLog(":memory:") as scratch_log:
            assert backend.world is not None
            backend.world.spawn_tubes(num_tubes)

            controller = LearnedController(
                config, backend.arm, backend.camera, world=backend.world, policy=policy
            )
            pipeline = SortPipeline(config, backend, sort_log=scratch_log, controller=controller)
            pipeline_report = pipeline.run(run_id=f"eval-{rollout_seed}")

        if pipeline_report.outcomes:
            outcome = pipeline_report.outcomes[0]
            report.rollouts.append(
                RolloutResult(
                    seed=rollout_seed,
                    success=outcome.success,
                    reason=outcome.result.reason.value,
                    duration_s=outcome.result.duration_s,
                    place_error_m=outcome.result.place_error_m,
                    class_label=outcome.job.class_label,
                )
            )
        else:
            report.rollouts.append(
                RolloutResult(
                    seed=rollout_seed,
                    success=False,
                    reason="no_detections",
                    duration_s=0.0,
                )
            )
        logger.info(
            "rollout %d/%d (seed %d): %s",
            index + 1,
            episodes,
            rollout_seed,
            "success" if report.rollouts[-1].success else report.rollouts[-1].reason,
        )

    report.duration_s = time.perf_counter() - started

    destination = Path(output_path) if output_path else checkpoint / EVALUATION_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", destination)
    return report
