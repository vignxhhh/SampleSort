"""End-to-end sort tests in simulation.

Spec section 8, phase 5: an end-to-end sim test sorting at least four tubes of
mixed colours into the correct racks, with every attempt written to the log.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from samplesort.config import SampleSortConfig
from samplesort.control.scripted import ExecutionResult, FailureReason, ScriptedController
from samplesort.hal.factory import Backend, build_backend
from samplesort.logging_db.sort_log import SortLog
from samplesort.pipeline import SortPipeline
from samplesort.planning.rack_state import RackState
from samplesort.planning.task_planner import PickPlaceJob


class MissingGraspController(ScriptedController):
    """A test double that reports a missed grasp for every job it is handed."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Start with an empty call counter."""
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.calls = 0

    def execute(self, job: PickPlaceJob) -> ExecutionResult:  # noqa: ARG002 - stub ignores the job
        """Count the call and always report a missed grasp."""
        self.calls += 1
        return ExecutionResult(success=False, reason=FailureReason.GRASP_MISSED, duration_s=0.01)


@pytest.fixture
def sort_log() -> Iterator[SortLog]:
    """An in-memory sort log."""
    with SortLog(":memory:") as opened:
        yield opened


@pytest.fixture
def backend(config: SampleSortConfig) -> Iterator[Backend]:
    """A connected headless sim backend."""
    built = build_backend(config, gui=False, seed=config.seed)
    with built:
        yield built


def run_sort(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog, num_tubes: int, **kwargs: object
) -> tuple[SortPipeline, object]:
    """Spawn tubes and run the pipeline to completion."""
    assert backend.world is not None
    backend.world.spawn_tubes(num_tubes)
    pipeline = SortPipeline(config, backend, sort_log=sort_log)
    report = pipeline.run(**kwargs)  # type: ignore[arg-type]
    return pipeline, report


# ------------------------------------------------------------------- happy path


def test_sorts_six_mixed_tubes_into_the_right_racks(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    pipeline, report = run_sort(config, backend, sort_log, 6)

    assert report.attempted == 6, f"expected 6 jobs, got {report.attempted}"
    assert report.succeeded == 6, f"failures: {report.failures_by_reason()}"
    assert report.success_rate == pytest.approx(1.0)

    # Every job went to the rack its class maps to.
    for outcome in report.outcomes:
        assert outcome.job.rack_id == config.rack_for_class(outcome.job.class_label)


def test_sorted_tubes_physically_reach_their_slots(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    assert backend.world is not None
    _, report = run_sort(config, backend, sort_log, 4)

    for outcome in report.outcomes:
        assert outcome.result.place_error_m is not None
        assert outcome.result.place_error_m <= 0.03


def test_mixed_colours_are_covered(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    _, report = run_sort(config, backend, sort_log, 8)
    labels = {outcome.job.class_label for outcome in report.outcomes}
    assert labels == set(config.classes.labels), f"only saw {labels}"


def test_pickup_zone_ends_up_clear(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    assert backend.world is not None
    pipeline, _ = run_sort(config, backend, sort_log, 6)
    assert pipeline.detector.detect_in_pickup_zone(backend.camera.read()) == []


def test_arm_returns_home_after_a_run(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    import numpy as np

    run_sort(config, backend, sort_log, 4)
    assert np.allclose(backend.arm.get_joint_positions(), config.arm.home_position, atol=0.05)


# --------------------------------------------------------------------- logging


def test_every_attempt_is_logged(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    _, report = run_sort(config, backend, sort_log, 6)
    records = sort_log.all_records()
    assert len(records) == report.attempted

    for stored in records:
        assert stored.mode == "scripted"
        assert stored.run_id == report.run_id
        assert stored.rack_id == config.rack_for_class(stored.class_label)
        assert stored.duration_s > 0.0
        assert stored.timestamp is not None


def test_logged_successes_match_the_report(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    _, report = run_sort(config, backend, sort_log, 6)
    assert sort_log.count(success=True) == report.succeeded
    assert sort_log.success_rate() == pytest.approx(report.success_rate)


def test_log_records_the_pickup_position(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    run_sort(config, backend, sort_log, 4)
    zone = config.workspace.pickup_zone
    for stored in sort_log.all_records():
        assert zone.contains(*stored.pickup_xy)


def test_failures_carry_a_reason(config: SampleSortConfig, sort_log: SortLog) -> None:
    backend = build_backend(config, gui=False, seed=config.seed)
    with backend:
        assert backend.world is not None
        backend.world.spawn_tubes(4)
        controller = MissingGraspController(config, backend.arm, backend.world)
        pipeline = SortPipeline(config, backend, sort_log=sort_log, controller=controller)
        report = pipeline.run()

    assert report.succeeded == 0
    assert report.failures_by_reason() == {"grasp_missed": 4}
    for stored in sort_log.all_records():
        assert stored.success is False
        assert stored.failure_reason == "grasp_missed"


def test_failed_jobs_release_their_reserved_slot(
    config: SampleSortConfig, sort_log: SortLog
) -> None:
    backend = build_backend(config, gui=False, seed=config.seed)
    with backend:
        assert backend.world is not None
        backend.world.spawn_tubes(4)
        pipeline = SortPipeline(
            config,
            backend,
            sort_log=sort_log,
            controller=MissingGraspController(config, backend.arm, backend.world),
        )
        pipeline.run()
        assert pipeline.rack_state.total_occupied == 0


def test_retries_are_bounded_by_the_config(config: SampleSortConfig, sort_log: SortLog) -> None:
    backend = build_backend(config, gui=False, seed=config.seed)
    with backend:
        assert backend.world is not None
        backend.world.spawn_tubes(2)
        controller = MissingGraspController(config, backend.arm, backend.world)
        pipeline = SortPipeline(config, backend, sort_log=sort_log, controller=controller)
        report = pipeline.run()

    expected_per_job = 1 + config.pipeline.grasp_retries
    assert controller.calls == report.attempted * expected_per_job
    assert all(outcome.attempts == expected_per_job for outcome in report.outcomes)


# --------------------------------------------------------------------- dry run


def test_dry_run_plans_without_moving_or_logging(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    import numpy as np

    assert backend.world is not None
    backend.world.spawn_tubes(5)
    before = [backend.world.tube_xy(t.body_id) for t in backend.world.tubes]

    pipeline = SortPipeline(config, backend, sort_log=sort_log)
    report = pipeline.run(dry_run=True)

    assert report.dry_run
    assert report.attempted == 0
    assert report.detections_seen == 5
    assert sort_log.all_records() == []

    after = [backend.world.tube_xy(t.body_id) for t in backend.world.tubes]
    for (x0, y0), (x1, y1) in zip(before, after, strict=True):
        assert np.hypot(x1 - x0, y1 - y0) < 0.002


def test_dry_run_on_an_empty_table_is_a_no_op(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    pipeline = SortPipeline(config, backend, sort_log=sort_log)
    report = pipeline.run(dry_run=True)
    assert report.attempted == 0
    assert report.detections_seen == 0


# ------------------------------------------------------------------- behaviour


def test_full_racks_stop_the_loop_cleanly(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    assert backend.world is not None
    backend.world.spawn_tubes(4)
    pipeline = SortPipeline(config, backend, sort_log=sort_log)
    # Pre-fill every rack so nothing can be placed.
    for rack in config.workspace.racks:
        for _ in range(rack.num_slots):
            label = next(
                label for label in config.classes.labels if config.rack_for_class(label) == rack.id
            )
            try:
                pipeline.rack_state.occupy(label)
            except Exception:  # noqa: BLE001 - rack already full is the point
                break

    report = pipeline.run()
    assert report.attempted == 0
    assert any(reason == "rack_full" for _, reason in report.skipped)


def test_run_id_is_unique_per_run(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    assert backend.world is not None
    pipeline = SortPipeline(config, backend, sort_log=sort_log)
    first = pipeline.run(dry_run=True)
    second = pipeline.run(dry_run=True)
    assert first.run_id != second.run_id


def test_explicit_run_id_is_honoured(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    assert backend.world is not None
    backend.world.spawn_tubes(3)
    pipeline = SortPipeline(config, backend, sort_log=sort_log)
    report = pipeline.run(run_id="my-run")
    assert report.run_id == "my-run"
    assert all(r.run_id == "my-run" for r in sort_log.all_records())


def test_report_summary_lines_are_populated(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    _, report = run_sort(config, backend, sort_log, 4)
    text = "\n".join(report.summary_lines())
    assert report.run_id in text
    assert "jobs succeeded" in text


def test_rack_state_matches_the_successes(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    pipeline, report = run_sort(config, backend, sort_log, 6)
    assert pipeline.rack_state.total_occupied == report.succeeded


def test_a_second_run_reuses_the_remaining_slots(
    config: SampleSortConfig, backend: Backend, sort_log: SortLog
) -> None:
    assert backend.world is not None
    pipeline = SortPipeline(config, backend, sort_log=sort_log)

    backend.world.spawn_tubes(4)
    first = pipeline.run()
    occupied_after_first = pipeline.rack_state.total_occupied

    backend.world.spawn_tubes(4)
    second = pipeline.run()

    assert first.succeeded > 0 and second.succeeded > 0
    assert pipeline.rack_state.total_occupied > occupied_after_first


def test_rack_state_is_independent_of_the_pipeline(config: SampleSortConfig) -> None:
    state = RackState(config.workspace, config.classes)
    assert state.total_occupied == 0
