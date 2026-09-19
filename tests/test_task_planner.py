"""Tests for turning detections into ordered pick-and-place jobs."""

from __future__ import annotations

import pytest

from samplesort.config import SampleSortConfig
from samplesort.perception.types import Detection
from samplesort.planning.rack_state import RackState
from samplesort.planning.task_planner import PickPlaceJob, TaskPlanner


@pytest.fixture
def planner(config: SampleSortConfig) -> TaskPlanner:
    """A planner over a fresh, empty rack state."""
    return TaskPlanner(config, RackState(config.workspace, config.classes))


def make_detection(
    label: str, x: float, y: float, confidence: float = 0.9, sample_id: str | None = None
) -> Detection:
    """Build a detection at a table position, bypassing the camera."""
    return Detection(
        pixel_xy=(0.0, 0.0),
        table_xy=(x, y),
        class_label=label,
        confidence=confidence,
        area_px=200.0,
        sample_id=sample_id,
    )


def test_each_detection_becomes_a_job(planner: TaskPlanner) -> None:
    detections = [
        make_detection("red", 0.18, 0.05),
        make_detection("blue", 0.20, -0.05),
        make_detection("green", 0.22, 0.00),
    ]
    plan = planner.plan(detections)
    assert len(plan.jobs) == 3
    assert not plan.skipped
    assert not plan.is_empty


def test_jobs_target_the_rack_their_class_maps_to(
    planner: TaskPlanner, config: SampleSortConfig
) -> None:
    for job in planner.plan(
        [make_detection(label, 0.19, 0.0) for label in config.classes.labels]
    ).jobs:
        assert job.rack_id == config.rack_for_class(job.class_label)


def test_jobs_are_ordered_nearest_first(planner: TaskPlanner) -> None:
    detections = [
        make_detection("red", 0.24, 0.10),
        make_detection("blue", 0.15, 0.00),
        make_detection("green", 0.20, 0.05),
    ]
    jobs = planner.plan(detections).jobs
    distances = [(j.pick_xy[0] ** 2 + j.pick_xy[1] ** 2) ** 0.5 for j in jobs]
    assert distances == sorted(distances)


def test_slots_are_allocated_without_collisions(planner: TaskPlanner) -> None:
    detections = [make_detection("red", 0.16 + 0.02 * i, 0.0) for i in range(4)]
    jobs = planner.plan(detections).jobs
    slots = [(j.rack_id, j.slot_index) for j in jobs]
    assert len(set(slots)) == len(slots)


def test_planning_reserves_slots_in_the_rack_state(
    planner: TaskPlanner, config: SampleSortConfig
) -> None:
    planner.plan([make_detection("red", 0.18, 0.0), make_detection("red", 0.20, 0.0)])
    assert planner.rack_state.count_occupied(config.rack_for_class("red")) == 2


def test_low_confidence_detections_are_skipped(
    planner: TaskPlanner, config: SampleSortConfig
) -> None:
    threshold = config.classes.detector.min_confidence
    plan = planner.plan(
        [
            make_detection("red", 0.18, 0.0, confidence=threshold - 0.01),
            make_detection("blue", 0.20, 0.0, confidence=threshold + 0.10),
        ]
    )
    assert len(plan.jobs) == 1
    assert plan.jobs[0].class_label == "blue"
    assert [s.reason for s in plan.skipped] == ["low_confidence"]


def test_detections_outside_the_pickup_zone_are_skipped(
    planner: TaskPlanner, config: SampleSortConfig
) -> None:
    zone = config.workspace.pickup_zone
    plan = planner.plan(
        [
            make_detection("red", zone.x_max + 0.08, zone.y_max + 0.08),
            make_detection("blue", (zone.x_min + zone.x_max) / 2, 0.0),
        ]
    )
    assert len(plan.jobs) == 1
    assert [s.reason for s in plan.skipped] == ["outside_pickup_zone"]


def test_detections_for_a_full_rack_are_skipped(
    planner: TaskPlanner, config: SampleSortConfig
) -> None:
    capacity = config.workspace.rack(config.rack_for_class("red")).num_slots
    detections = [make_detection("red", 0.15 + 0.008 * i, 0.0) for i in range(capacity + 2)]

    plan = planner.plan(detections)
    assert len(plan.jobs) == capacity
    assert [s.reason for s in plan.skipped] == ["rack_full"] * 2


def test_skipping_one_class_does_not_block_another(
    planner: TaskPlanner, config: SampleSortConfig
) -> None:
    capacity = config.workspace.rack(config.rack_for_class("red")).num_slots
    detections = [make_detection("red", 0.15 + 0.008 * i, 0.0) for i in range(capacity + 1)]
    detections.append(make_detection("blue", 0.22, 0.05))

    plan = planner.plan(detections)
    assert any(job.class_label == "blue" for job in plan.jobs)
    assert len(plan.skipped) == 1


def test_sample_id_and_confidence_are_carried_onto_the_job(planner: TaskPlanner) -> None:
    plan = planner.plan([make_detection("red", 0.18, 0.0, confidence=0.83, sample_id="S-007")])
    job = plan.jobs[0]
    assert job.sample_id == "S-007"
    assert job.confidence == pytest.approx(0.83)


def test_planning_nothing_yields_an_empty_plan(planner: TaskPlanner) -> None:
    plan = planner.plan([])
    assert plan.is_empty
    assert not plan.skipped


def test_successive_plans_keep_allocating_new_slots(planner: TaskPlanner) -> None:
    first = planner.plan([make_detection("red", 0.18, 0.0)]).jobs[0]
    second = planner.plan([make_detection("red", 0.18, 0.0)]).jobs[0]
    assert first.slot_index != second.slot_index


def test_job_helpers(config: SampleSortConfig) -> None:
    job = PickPlaceJob(
        class_label="red", pick_xy=(0.2, 0.05), rack_id="rack_a", slot_index=1, place_xy=(0.1, 0.2)
    )
    assert job.pick_pose(0.12).position.tolist() == [0.2, 0.05, 0.12]
    assert job.place_pose(0.09).position.tolist() == [0.1, 0.2, 0.09]
    assert "rack_a slot 1" in job.describe()
    assert config.rack_for_class("red") == "rack_a"
