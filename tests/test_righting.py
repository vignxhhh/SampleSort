"""Standing knocked-over tubes back up: geometry, perception, the manoeuvre, the loop."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from samplesort.config import RackConfig, SampleSortConfig, load_config
from samplesort.control.scripted import FailureReason, ScriptedController
from samplesort.hal.factory import build_backend
from samplesort.logging_db.sort_log import SortLog
from samplesort.perception.fallen import body_side, body_side_scores
from samplesort.pipeline import SortPipeline
from samplesort.planning.kinematics import (
    TOP_DOWN_PITCH,
    Pose,
    UnreachableError,
    inverse_kinematics,
    tool_rotation,
    top_down_roll_for_axis,
)
from samplesort.planning.task_planner import PickPlaceJob
from samplesort.sim._bullet import pb

# ----------------------------------------------------------------- kinematics


def test_tool_rotation_is_a_proper_rotation() -> None:
    rotation = tool_rotation(Pose(0.15, -0.1, 0.1, -1.05, -1.2))
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_top_down_points_the_tool_straight_down() -> None:
    approach = tool_rotation(Pose(0.2, 0.05, 0.1, TOP_DOWN_PITCH, 0.4))[:, 0]
    assert np.allclose(approach, [0.0, 0.0, -1.0], atol=1e-12)


def test_level_roll_decides_which_way_tool_z_points() -> None:
    up = tool_rotation(Pose(0.0, 0.29, 0.13, 0.0, 0.0))[:, 2]
    down = tool_rotation(Pose(0.0, 0.29, 0.13, 0.0, math.pi))[:, 2]
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(down, [0.0, 0.0, -1.0], atol=1e-9)


def test_tool_rotation_matches_the_simulated_arm(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=1)
    with backend:
        assert backend.world is not None
        for pose in (
            Pose(0.2, 0.05, 0.12, TOP_DOWN_PITCH, 0.3),
            Pose(0.15, -0.1, 0.1, -math.pi / 3, -1.2),
        ):
            backend.world.reset_arm(inverse_kinematics(config.arm, pose))
            _, quat = backend.world.tool_pose()
            simulated = np.asarray(pb.getMatrixFromQuaternion(quat.tolist())).reshape(3, 3)
            assert np.abs(simulated - tool_rotation(pose)).max() < 1e-4


@pytest.mark.parametrize("axis_deg", [0.0, 37.0, 90.0, 151.0, -120.0])
def test_roll_lays_tool_z_along_an_axis(axis_deg: float) -> None:
    x, y, axis = 0.17, 0.06, math.radians(axis_deg)
    roll = top_down_roll_for_axis(x, y, axis)
    tool_z = tool_rotation(Pose(x, y, 0.1, TOP_DOWN_PITCH, roll))[:, 2]
    # A line, so either direction along it will do.
    assert abs(float(tool_z @ [math.cos(axis), math.sin(axis), 0.0])) == pytest.approx(1.0)
    assert -math.pi <= roll <= math.pi
    assert abs(roll) <= math.pi / 2.0 + 1e-9


# --------------------------------------------------------------------- config


def test_zone_margin_grows_the_zone(config: SampleSortConfig) -> None:
    zone = config.workspace.pickup_zone
    outside = (zone.x_max + 0.02, zone.y_max)
    assert not zone.contains(*outside)
    assert zone.contains(*outside, margin=0.03)


def test_station_inside_the_pickup_zone_is_rejected(config_dir: Path) -> None:
    config = load_config(config_dir, mode="sim")
    data = config.workspace.model_dump()
    data["righting_station"] = config.workspace.pickup_zone.center
    with pytest.raises(ValidationError, match="outside the pickup zone"):
        type(config.workspace).model_validate(data)


def test_station_on_a_rack_slot_is_rejected(config_dir: Path) -> None:
    config = load_config(config_dir, mode="sim")
    data = config.workspace.model_dump()
    data["righting_station"] = tuple(config.workspace.racks[0].slots[0])
    with pytest.raises(ValidationError, match="overlaps a slot"):
        type(config.workspace).model_validate(data)


def test_station_whose_sweep_crosses_a_rack_is_rejected(config_dir: Path) -> None:
    config = load_config(config_dir, mode="sim")
    data = config.model_dump()
    # Directly beyond Rack C: standing a tube up would drag the wrist link across it.
    data["workspace"]["righting_station"] = (0.36, 0.0)
    with pytest.raises(ValidationError, match="sweep over rack_c"):
        SampleSortConfig.model_validate(data)


def test_shipped_station_is_reachable_for_every_righting_pose(config: SampleSortConfig) -> None:
    ws = config.workspace
    assert ws.righting_station is not None
    sx, sy = ws.righting_station
    for z in (ws.table_grasp_height, ws.table_grasp_height + ws.approach_clearance):
        inverse_kinematics(config.arm, Pose(sx, sy, z, TOP_DOWN_PITCH, 0.0))
    safe = ws.tube.total_height + ws.approach_clearance
    for roll in (0.0, math.pi - 1e-4):
        inverse_kinematics(config.arm, Pose(sx, sy, safe, 0.0, roll))


# ----------------------------------------------------------------- perception


def test_calibration_projects_a_raised_point_back_to_itself(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=1)
    with backend:
        calibration = SortPipeline(config, backend).calibration
    for height in (0.0, 0.01, 0.1):
        u, v = calibration.point_to_pixel(0.2, 0.05, height)
        x, y = calibration.pixel_to_table(u, v, object_height=height)
        assert (x, y) == pytest.approx((0.2, 0.05), abs=1e-6)


def test_knock_over_lays_the_tube_along_the_requested_direction(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=3)
    with backend:
        world = backend.world
        assert world is not None
        tube = world.spawn_tubes(1)[0]
        world.knock_over(tube.body_id, math.radians(60))
        assert not world.tube_is_upright(tube.body_id)
        direction = world.tube_cap_position(tube.body_id) - world.tube_position(tube.body_id)
        assert math.degrees(math.atan2(direction[1], direction[0])) == pytest.approx(60, abs=5)


def test_detector_separates_fallen_tubes_from_standing_ones(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=42)
    with backend:
        world = backend.world
        assert world is not None
        tubes = world.spawn_tubes(4)
        fallen_tube = tubes[1]
        world.knock_over(fallen_tube.body_id, math.radians(40))
        pipeline = SortPipeline(config, backend)
        scene = pipeline.perceive_scene()
        cap = world.tube_cap_position(fallen_tube.body_id)
        truth = world.tube_cap_position(fallen_tube.body_id) - world.tube_position(
            fallen_tube.body_id
        )

    assert len(scene.upright) == 3
    assert len(scene.fallen) == 1
    fallen = scene.fallen[0]
    assert fallen.class_label == fallen_tube.label
    assert math.hypot(fallen.table_xy[0] - cap[0], fallen.table_xy[1] - cap[1]) < 0.006
    # The axis is a line, so compare modulo pi.
    error = abs(
        (fallen.axis_yaw - math.atan2(truth[1], truth[0]) + math.pi / 2) % math.pi - math.pi / 2
    )
    assert math.degrees(error) < 6.0


def _body_frame(
    config: SampleSortConfig, side: float | None
) -> tuple[np.ndarray, object, np.ndarray, np.ndarray]:
    """A synthetic overhead frame: flat table, a held cap, and optionally a body."""
    backend = build_backend(config, seed=1)
    with backend:
        calibration = SortPipeline(config, backend).calibration
    frame = np.full((config.camera.height, config.camera.width, 3), 212, dtype=np.uint8)
    cap = np.array([0.19, 0.0, 0.11])
    axis = np.array([0.0, 1.0, 0.0])
    tube = config.workspace.tube
    if side is not None:
        start = cap[:2] + axis[:2] * side * tube.cap_height / 2
        end = cap[:2] + axis[:2] * side * (tube.cap_height / 2 + tube.body_height)
        a = tuple(int(round(c)) for c in calibration.point_to_pixel(*start, cap[2]))
        b = tuple(int(round(c)) for c in calibration.point_to_pixel(*end, cap[2]))
        cv2.line(frame, a, b, (165, 168, 175), thickness=16)
    return frame, calibration, cap, axis


def test_body_side_finds_the_body(config: SampleSortConfig) -> None:
    for side in (1.0, -1.0):
        frame, calibration, cap, axis = _body_frame(config, side)
        assert body_side(frame, calibration, cap, axis, config.workspace.tube) == side  # type: ignore[arg-type]


def test_body_side_admits_when_it_cannot_tell(config: SampleSortConfig) -> None:
    frame, calibration, cap, axis = _body_frame(config, None)
    scores = body_side_scores(frame, calibration, cap, axis, config.workspace.tube)  # type: ignore[arg-type]
    assert scores == (0.0, 0.0)
    assert body_side(frame, calibration, cap, axis, config.workspace.tube) is None  # type: ignore[arg-type]


def test_a_coloured_cap_beside_the_tube_is_not_mistaken_for_its_body(
    config: SampleSortConfig,
) -> None:
    frame, calibration, cap, axis = _body_frame(config, 1.0)
    tube = config.workspace.tube
    beside = cap[:2] - axis[:2] * (tube.cap_height / 2 + 0.03)
    centre = tuple(int(round(c)) for c in calibration.point_to_pixel(*beside, cap[2]))
    cv2.circle(frame, centre, 14, (40, 60, 200), thickness=-1)  # a saturated red cap
    assert body_side(frame, calibration, cap, axis, tube) == 1.0  # type: ignore[arg-type]


def test_a_rack_beside_the_tube_is_not_mistaken_for_its_body(config: SampleSortConfig) -> None:
    # Measured in sim: a tube inspected beside Rack A was stood on its cap,
    # because the grey plate and its dark holes scored as body.
    frame, calibration, cap, axis = _body_frame(config, -1.0)
    rack = RackConfig(id="r", label="R", slots=[(0.17, 0.045), (0.21, 0.045)])
    x0, x1, y0, y1 = rack.footprint
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    polygon = np.array([calibration.point_to_pixel(x, y, 0.0) for x, y in corners])  # type: ignore[attr-defined]
    cv2.fillConvexPoly(frame, np.round(polygon).astype(np.int32), (158, 158, 160))
    for x, y in rack.slots:
        centre = tuple(int(round(c)) for c in calibration.point_to_pixel(x, y, 0.0))  # type: ignore[attr-defined]
        cv2.circle(frame, centre, 13, (60, 60, 62), thickness=-1)
    tube = config.workspace.tube
    assert body_side(frame, calibration, cap, axis, tube) != -1.0  # type: ignore[arg-type]
    assert body_side(frame, calibration, cap, axis, tube, [rack]) == -1.0  # type: ignore[arg-type]


# ----------------------------------------------------------------- manoeuvre


def _one_fallen(config: SampleSortConfig, seed: int = 7) -> tuple[object, object, object]:
    backend = build_backend(config, seed=seed)
    backend.connect()
    world = backend.world
    assert world is not None
    tube = world.spawn_tubes(1)[0]
    world.knock_over(tube.body_id, math.radians(37 + 13 * seed))
    return backend, tube, SortPipeline(config, backend)


def test_right_tube_stands_a_tube_up_cap_first_at_the_station(config: SampleSortConfig) -> None:
    backend, tube, pipeline = _one_fallen(config)
    try:
        fallen = pipeline.perceive_scene().fallen[0]  # type: ignore[attr-defined]
        result = pipeline.controller.right_tube(fallen, pipeline._body_side_probe)  # type: ignore[attr-defined]
        world = backend.world  # type: ignore[attr-defined]
        world.step(240)
        station = np.array(config.workspace.righting_station)
        landed = np.array(world.tube_xy(tube.body_id))  # type: ignore[attr-defined]
        assert result.success, result.reason
        assert world.tube_is_upright(tube.body_id)  # type: ignore[attr-defined]
        # Cap up: the cap sits near the top of the standing tube.
        assert world.tube_cap_position(tube.body_id)[2] > config.workspace.tube.body_height  # type: ignore[attr-defined]
        assert np.linalg.norm(landed - station) < config.workspace.righting_station_radius
    finally:
        backend.disconnect()  # type: ignore[attr-defined]


def test_an_undecided_body_side_puts_the_tube_back_down(config: SampleSortConfig) -> None:
    backend, tube, pipeline = _one_fallen(config)
    try:
        fallen = pipeline.perceive_scene().fallen[0]  # type: ignore[attr-defined]
        result = pipeline.controller.right_tube(fallen, lambda cap, axis: None)  # type: ignore[attr-defined]
        assert not result.success
        assert result.reason is FailureReason.BODY_SIDE_UNKNOWN
        assert backend.arm.has_object() is False  # type: ignore[attr-defined]
        # Set down where it was found, still lying, ready to be tried again.
        assert not backend.world.tube_is_upright(tube.body_id)  # type: ignore[attr-defined]
        assert pipeline.perceive_scene().fallen  # type: ignore[attr-defined]
    finally:
        backend.disconnect()  # type: ignore[attr-defined]


def test_right_tube_needs_a_station(config: SampleSortConfig) -> None:
    config.workspace.righting_station = None
    backend, _, pipeline = _one_fallen(config)
    try:
        fallen = pipeline.perceive_scene().fallen[0]  # type: ignore[attr-defined]
        result = ScriptedController(config, backend.arm).right_tube(fallen, lambda c, a: 1.0)  # type: ignore[attr-defined]
        assert result.reason is FailureReason.NO_RIGHTING_STATION
    finally:
        backend.disconnect()  # type: ignore[attr-defined]


def test_picks_from_the_station_route_via_the_pickup_zone(config: SampleSortConfig) -> None:
    controller = ScriptedController(config, arm=None)  # type: ignore[arg-type]
    station = config.workspace.righting_station
    assert station is not None
    job = PickPlaceJob(
        class_label="green",
        pick_xy=station,
        rack_id="rack_c",
        slot_index=3,
        place_xy=(0.295, -0.075),
    )
    names = [name for name, _ in controller.waypoint_poses(job)]
    assert names == ["pre_grasp", "grasp", "lift", "via", "pre_place", "place", "retreat"]
    ordinary = PickPlaceJob(
        class_label="green",
        pick_xy=(0.19, 0.0),
        rack_id="rack_c",
        slot_index=3,
        place_xy=(0.295, -0.075),
    )
    assert "via" not in [name for name, _ in controller.waypoint_poses(ordinary)]


def _job_at(xy: tuple[float, float]) -> PickPlaceJob:
    return PickPlaceJob(
        class_label="red", pick_xy=xy, rack_id="rack_a", slot_index=0, place_xy=(0.1, 0.2)
    )


def test_ordinary_picks_keep_the_full_approach_clearance(config: SampleSortConfig) -> None:
    controller = ScriptedController(config, arm=None)  # type: ignore[arg-type]
    assert controller.pick_clearance(_job_at((0.19, 0.0))) == config.workspace.approach_clearance


def test_a_pick_at_the_reach_edge_approaches_lower(config: SampleSortConfig) -> None:
    # Measured in a noisy run: a righted tube detected here was given up on,
    # because the point 70 mm above it is out of reach though the cap is not.
    controller = ScriptedController(config, arm=None)  # type: ignore[arg-type]
    job = _job_at((0.2853, 0.1645))
    ws = config.workspace
    with pytest.raises(UnreachableError):
        inverse_kinematics(config.arm, job.pick_pose(ws.table_grasp_height + ws.approach_clearance))
    clearance = controller.pick_clearance(job)
    assert ws.tube.cap_height + 0.02 <= clearance < ws.approach_clearance
    assert controller.is_feasible(job)
    poses = dict(controller.waypoint_poses(job))
    assert poses["pre_grasp"].z == pytest.approx(ws.table_grasp_height + clearance)


def test_an_unreachable_cap_is_still_reported_unreachable(config: SampleSortConfig) -> None:
    controller = ScriptedController(config, arm=None)  # type: ignore[arg-type]
    job = _job_at((0.40, 0.0))
    assert controller.pick_clearance(job) == config.workspace.approach_clearance
    assert not controller.is_feasible(job)


# ----------------------------------------------------------------------- loop


def test_pipeline_sorts_knocked_over_tubes_with_their_ids(
    config: SampleSortConfig, tmp_path: Path
) -> None:
    backend = build_backend(config, seed=3)
    with backend, SortLog(tmp_path / "log.db") as log:
        world = backend.world
        assert world is not None
        tubes = world.spawn_tubes(4)
        for index, tube in enumerate(tubes[:2]):
            world.knock_over(tube.body_id, math.radians(40 + 111 * index))
        report = SortPipeline(config, backend, sort_log=log).run()
        records = log.query()
        standing_in_racks = [
            t for t in world.tubes if t.is_sorted and world.tube_is_upright(t.body_id)
        ]

    assert report.righted == 2
    assert report.fallen_left == []
    assert len(standing_in_racks) == 4
    assert {r.sample_id for r in records if r.success} == {t.sample_id for t in tubes}
    assert "fallen tubes" in "\n".join(report.summary_lines())


def test_dry_run_reports_fallen_tubes_without_moving(config: SampleSortConfig) -> None:
    backend = build_backend(config, seed=3)
    with backend:
        world = backend.world
        assert world is not None
        tube = world.spawn_tubes(1)[0]
        world.knock_over(tube.body_id, math.radians(40))
        report = SortPipeline(config, backend).run(dry_run=True)
        assert report.rightings == []
        assert not world.tube_is_upright(tube.body_id)
