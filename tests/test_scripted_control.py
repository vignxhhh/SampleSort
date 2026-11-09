"""Tests for the scripted pick-and-place controller in simulation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from samplesort.config import SampleSortConfig
from samplesort.control.scripted import ExecutionResult, FailureReason, ScriptedController
from samplesort.hal.factory import Backend, build_backend
from samplesort.planning.kinematics import forward_kinematics
from samplesort.planning.task_planner import PickPlaceJob


@pytest.fixture
def backend(config: SampleSortConfig) -> Iterator[Backend]:
    """A connected headless sim backend."""
    built = build_backend(config, gui=False, seed=config.seed)
    with built:
        yield built


def _job_for(
    config: SampleSortConfig, backend: Backend, tube_index: int, slot: int
) -> PickPlaceJob:
    assert backend.world is not None
    tube = backend.world.tubes[tube_index]
    x, y = backend.world.tube_xy(tube.body_id)
    rack_id = config.rack_for_class(tube.label)
    return PickPlaceJob(
        class_label=tube.label,
        pick_xy=(x, y),
        rack_id=rack_id,
        slot_index=slot,
        place_xy=config.workspace.rack(rack_id).slots[slot],
        body_id=tube.body_id,
    )


def test_waypoint_stages_are_in_the_expected_order(
    config: SampleSortConfig, backend: Backend
) -> None:
    controller = ScriptedController(config, backend.arm, backend.world)
    job = PickPlaceJob("red", (0.20, 0.0), "rack_a", 0, config.workspace.racks[0].slots[0])
    stages = [name for name, _ in controller.waypoint_poses(job)]
    assert stages == ["pre_grasp", "grasp", "lift", "pre_place", "place", "retreat"]


def test_approach_waypoints_sit_above_their_contact_waypoints(
    config: SampleSortConfig, backend: Backend
) -> None:
    controller = ScriptedController(config, backend.arm, backend.world)
    job = PickPlaceJob("red", (0.20, 0.0), "rack_a", 0, config.workspace.racks[0].slots[0])
    poses = dict(controller.waypoint_poses(job))
    clearance = config.workspace.approach_clearance
    assert poses["pre_grasp"].z == pytest.approx(poses["grasp"].z + clearance)
    assert poses["lift"].z == pytest.approx(poses["grasp"].z + clearance)
    assert poses["pre_place"].z == pytest.approx(poses["place"].z + clearance)
    assert poses["retreat"].z == pytest.approx(poses["place"].z + clearance)


def test_solved_waypoints_match_their_cartesian_targets(
    config: SampleSortConfig, backend: Backend
) -> None:
    controller = ScriptedController(config, backend.arm, backend.world)
    job = PickPlaceJob("blue", (0.18, 0.05), "rack_b", 1, config.workspace.racks[1].slots[1])
    for (stage, pose), (same_stage, q) in zip(
        controller.waypoint_poses(job), controller.solve_waypoints(job), strict=True
    ):
        assert stage == same_stage
        achieved = forward_kinematics(config.arm, q)
        np.testing.assert_allclose(achieved.position, pose.position, atol=1e-9)


def test_scripted_pick_and_place_succeeds(config: SampleSortConfig, backend: Backend) -> None:
    assert backend.world is not None
    backend.world.spawn_tubes(3)
    controller = ScriptedController(config, backend.arm, backend.world)
    job = _job_for(config, backend, 0, 0)

    result = controller.execute(job)

    assert result.success, f"expected success, got {result.reason.value}"
    assert result.grasped
    assert result.reason is FailureReason.NONE
    assert result.place_error_m is not None
    assert result.place_error_m <= ScriptedController.PLACE_TOLERANCE_M
    assert len(result.waypoints) == 6


def test_placed_tube_ends_up_at_its_slot(config: SampleSortConfig, backend: Backend) -> None:
    assert backend.world is not None
    backend.world.spawn_tubes(2)
    controller = ScriptedController(config, backend.arm, backend.world)
    job = _job_for(config, backend, 0, 2)

    assert controller.execute(job).success
    x, y = backend.world.tube_xy(job.body_id)  # type: ignore[arg-type]
    assert np.hypot(x - job.place_xy[0], y - job.place_xy[1]) <= 0.03


def test_gripper_is_empty_after_a_successful_place(
    config: SampleSortConfig, backend: Backend
) -> None:
    assert backend.world is not None
    backend.world.spawn_tubes(2)
    controller = ScriptedController(config, backend.arm, backend.world)
    assert controller.execute(_job_for(config, backend, 0, 0)).success
    assert not backend.arm.has_object()  # type: ignore[attr-defined]


def test_three_tubes_in_a_row_all_succeed(config: SampleSortConfig, backend: Backend) -> None:
    assert backend.world is not None
    backend.world.spawn_tubes(3)
    controller = ScriptedController(config, backend.arm, backend.world)
    results: list[ExecutionResult] = []
    for index in range(3):
        results.append(controller.execute(_job_for(config, backend, index, index)))
    assert all(r.success for r in results), [r.reason.value for r in results]


def test_grasp_at_an_empty_spot_reports_a_missed_grasp(
    config: SampleSortConfig, backend: Backend
) -> None:
    assert backend.world is not None
    backend.world.spawn_tubes(1)
    controller = ScriptedController(config, backend.arm, backend.world)
    empty = PickPlaceJob(
        class_label="red",
        pick_xy=(0.24, 0.13),  # inside the zone, but nothing is there
        rack_id="rack_a",
        slot_index=0,
        place_xy=config.workspace.racks[0].slots[0],
        body_id=None,
    )
    # Make sure the spot really is empty before asserting on the failure mode.
    occupied = [backend.world.tube_xy(t.body_id) for t in backend.world.tubes]
    assert all(np.hypot(x - 0.24, y - 0.13) > 0.03 for x, y in occupied)

    result = controller.execute(empty)
    assert not result.success
    assert result.reason is FailureReason.GRASP_MISSED
    assert not result.grasped


def test_out_of_reach_pick_is_reported(config: SampleSortConfig, backend: Backend) -> None:
    controller = ScriptedController(config, backend.arm, backend.world)
    job = PickPlaceJob("red", (1.5, 0.0), "rack_a", 0, config.workspace.racks[0].slots[0])
    assert not controller.is_feasible(job)
    result = controller.execute(job)
    assert result.reason is FailureReason.UNREACHABLE_PICK
    assert not result.grasped


def test_out_of_reach_place_is_reported(config: SampleSortConfig, backend: Backend) -> None:
    controller = ScriptedController(config, backend.arm, backend.world)
    job = PickPlaceJob("red", (0.20, 0.0), "rack_a", 0, (1.5, 0.0))
    result = controller.execute(job)
    assert result.reason is FailureReason.UNREACHABLE_PLACE


def test_feasible_jobs_are_reported_feasible(config: SampleSortConfig, backend: Backend) -> None:
    controller = ScriptedController(config, backend.arm, backend.world)
    for rack in config.workspace.racks:
        for index, slot in enumerate(rack.slots):
            job = PickPlaceJob("red", (0.19, 0.0), rack.id, index, slot)
            assert controller.is_feasible(job), f"{rack.id} slot {index}"
