"""Tests for the benchmark runner, its aggregates and its artefacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from samplesort.benchmark import (
    METRICS_FILENAME,
    TABLE_FILENAME,
    BenchmarkReport,
    TrialResult,
    run_benchmark,
    write_report,
)
from samplesort.config import SampleSortConfig
from samplesort.logging_db.sort_log import SortLog


@pytest.fixture
def sort_log() -> Iterator[SortLog]:
    """An in-memory sort log."""
    with SortLog(":memory:") as opened:
        yield opened


def trial(
    *,
    seed: int = 1,
    tubes: int = 6,
    succeeded: int = 6,
    correct: int = 6,
    grasped: int = 6,
    total_attempts: int = 6,
    mean_time: float = 0.4,
    failures: dict[str, int] | None = None,
    skipped: dict[str, int] | None = None,
) -> TrialResult:
    """Build a trial result with sensible defaults."""
    return TrialResult(
        seed=seed,
        mode="scripted",
        tubes=tubes,
        attempted=tubes,
        succeeded=succeeded,
        correctly_sorted=correct,
        grasped=grasped,
        total_attempts=total_attempts,
        duration_s=mean_time * tubes,
        mean_time_per_sample_s=mean_time,
        failures=failures or {},
        skipped=skipped or {},
    )


def report(trials: list[TrialResult], *, mode: str = "scripted") -> BenchmarkReport:
    """Build a benchmark report around some trials."""
    return BenchmarkReport(
        mode=mode,
        trials=trials,
        tubes_per_trial=6,
        perception_noise_m=0.0,
        started_at="2026-01-01T00:00:00+00:00",
        total_duration_s=12.5,
        environment={"python": "3.11.0"},
    )


# ------------------------------------------------------------------ aggregates


def test_trial_rates() -> None:
    item = trial(tubes=6, succeeded=3, correct=2, grasped=4, total_attempts=8)
    assert item.completion_rate == pytest.approx(0.5)
    assert item.sorting_accuracy == pytest.approx(2 / 3)
    assert item.grasp_success_rate == pytest.approx(0.5)


def test_trial_rates_are_zero_when_nothing_happened() -> None:
    item = trial(tubes=0, succeeded=0, correct=0, grasped=0, total_attempts=0)
    assert item.completion_rate == 0.0
    assert item.sorting_accuracy == 0.0
    assert item.grasp_success_rate == 0.0


def test_report_aggregates_across_trials() -> None:
    aggregate = report([trial(succeeded=6, correct=6), trial(seed=2, succeeded=4, correct=3)])
    assert aggregate.num_trials == 2
    assert aggregate.total_tubes == 12
    assert aggregate.total_succeeded == 10
    assert aggregate.total_correct == 9
    assert aggregate.sorting_accuracy == pytest.approx(0.9)
    assert aggregate.completion_rate == pytest.approx(10 / 12)


def test_report_timing_statistics() -> None:
    aggregate = report([trial(mean_time=0.30), trial(seed=2, mean_time=0.50)])
    assert aggregate.mean_time_per_sample_s == pytest.approx(0.40)
    assert aggregate.stdev_time_per_sample_s > 0.0


def test_stdev_is_zero_for_a_single_trial() -> None:
    assert report([trial()]).stdev_time_per_sample_s == 0.0


def test_empty_report_does_not_divide_by_zero() -> None:
    empty = report([])
    assert empty.sorting_accuracy == 0.0
    assert empty.grasp_success_rate == 0.0
    assert empty.completion_rate == 0.0
    assert empty.mean_time_per_sample_s == 0.0


def test_failures_are_summed_and_sorted() -> None:
    aggregate = report(
        [
            trial(failures={"grasp_missed": 2, "misplaced": 1}),
            trial(seed=2, failures={"grasp_missed": 3}),
        ]
    )
    assert list(aggregate.failures_by_type().items()) == [("grasp_missed", 5), ("misplaced", 1)]


def test_skips_are_summed() -> None:
    aggregate = report([trial(skipped={"rack_full": 2}), trial(seed=2, skipped={"rack_full": 1})])
    assert aggregate.skips_by_type() == {"rack_full": 3}


# ------------------------------------------------------------------- rendering


def test_markdown_contains_every_required_metric() -> None:
    text = report([trial()]).to_markdown()
    for heading in (
        "Sorting accuracy (correct rack)",
        "Grasp success rate",
        "Mean time per sample",
        "Failures by type",
    ):
        assert heading in text
    assert text.startswith("| Metric |")


def test_markdown_reports_no_failures_when_there_were_none() -> None:
    assert "No failures were recorded." in report([trial()]).to_markdown()


def test_markdown_lists_failures_when_there_were_some() -> None:
    text = report([trial(succeeded=4, failures={"grasp_missed": 2})]).to_markdown()
    assert "`grasp_missed`" in text
    assert "No failures were recorded." not in text


def test_markdown_adds_a_comparison_column() -> None:
    scripted = report([trial()])
    learned = report([trial(succeeded=4, correct=4)], mode="learned")
    text = scripted.to_markdown(comparison=learned)
    assert "scripted mode" in text and "learned mode" in text
    assert text.splitlines()[0].count("|") == 4


def test_markdown_reports_the_noise_setting() -> None:
    quiet = report([trial()])
    assert "none" in quiet.to_markdown()
    noisy = report([trial()])
    noisy.perception_noise_m = 0.012
    assert "12 mm" in noisy.to_markdown()


def test_to_dict_is_json_serialisable() -> None:
    payload = report([trial()]).to_dict()
    restored = json.loads(json.dumps(payload))
    assert restored["num_trials"] == 1
    assert restored["summary"]["sorting_accuracy"] == pytest.approx(1.0)
    assert restored["trials"][0]["seed"] == 1


def test_write_report_creates_both_artefacts(tmp_path: Path) -> None:
    json_path, markdown_path = write_report(report([trial()]), tmp_path / "out")
    assert json_path.name == METRICS_FILENAME
    assert markdown_path.name == TABLE_FILENAME
    assert json.loads(json_path.read_text())["num_trials"] == 1
    assert markdown_path.read_text().startswith("| Metric |")


def test_write_report_embeds_the_comparison(tmp_path: Path) -> None:
    json_path, _ = write_report(
        report([trial()]), tmp_path, comparison=report([trial()], mode="learned")
    )
    payload = json.loads(json_path.read_text())
    assert payload["comparison"]["mode"] == "learned"


# ---------------------------------------------------------------------- runner


def test_rejects_bad_arguments(config: SampleSortConfig) -> None:
    with pytest.raises(ValueError, match="trials must be at least 1"):
        run_benchmark(config, trials=0)
    with pytest.raises(ValueError, match="num_tubes must be at least 1"):
        run_benchmark(config, trials=1, num_tubes=0)


@pytest.mark.slow
def test_runs_real_trials_and_sorts_everything(config: SampleSortConfig, sort_log: SortLog) -> None:
    result = run_benchmark(config, trials=2, num_tubes=4, start_seed=42, sort_log=sort_log)

    assert result.num_trials == 2
    assert result.total_tubes == 8
    assert result.mode == "scripted"
    assert result.completion_rate == pytest.approx(1.0), result.failures_by_type()
    assert result.sorting_accuracy == pytest.approx(1.0)
    assert result.mean_time_per_sample_s > 0.0
    assert result.environment["python"]


@pytest.mark.slow
def test_each_trial_uses_its_own_seed(config: SampleSortConfig, sort_log: SortLog) -> None:
    result = run_benchmark(config, trials=3, num_tubes=3, start_seed=100, sort_log=sort_log)
    assert [t.seed for t in result.trials] == [100, 101, 102]


@pytest.mark.slow
def test_benchmark_is_reproducible(config: SampleSortConfig) -> None:
    def summarise() -> tuple[int, int]:
        with SortLog(":memory:") as log:
            result = run_benchmark(config, trials=2, num_tubes=4, start_seed=7, sort_log=log)
        return result.total_succeeded, result.total_correct

    assert summarise() == summarise()


@pytest.mark.slow
def test_benchmark_writes_rows_to_the_log(config: SampleSortConfig, sort_log: SortLog) -> None:
    result = run_benchmark(config, trials=2, num_tubes=3, start_seed=5, sort_log=sort_log)
    records = sort_log.all_records()
    assert len(records) == sum(t.attempted for t in result.trials)
    assert {r.run_id for r in records} == {f"bench-scripted-{t.seed}" for t in result.trials}


@pytest.mark.slow
def test_localisation_noise_degrades_the_grasp_rate(
    config: SampleSortConfig, sort_log: SortLog
) -> None:
    """Large localisation error must push tubes outside the grasp tolerance."""
    clean = run_benchmark(config, trials=2, num_tubes=4, start_seed=42, sort_log=sort_log)

    noisy_config = config.model_copy(deep=True)
    # Comfortably beyond the gripper's horizontal grasp tolerance.
    noisy_config.pipeline.perception_noise_m = config.arm.gripper.grasp_tolerance_xy * 1.5
    noisy = run_benchmark(noisy_config, trials=2, num_tubes=4, start_seed=42, sort_log=sort_log)

    assert clean.grasp_success_rate == pytest.approx(1.0)
    assert noisy.grasp_success_rate < clean.grasp_success_rate
    assert "grasp_missed" in noisy.failures_by_type()
    assert noisy.perception_noise_m > 0.0
