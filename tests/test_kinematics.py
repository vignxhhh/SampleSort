"""Tests for the analytic forward and inverse kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from samplesort.config import SampleSortConfig
from samplesort.planning.kinematics import (
    TOP_DOWN_PITCH,
    ElbowBranch,
    Pose,
    UnreachableError,
    forward_kinematics,
    inverse_kinematics,
    is_reachable,
    link_positions,
    pose_error,
)


def _random_valid_joints(config: SampleSortConfig, rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(low, high) for low, high in config.arm.joint_limits])


def test_home_pose_is_above_the_table(config: SampleSortConfig) -> None:
    pose = forward_kinematics(config.arm, config.arm.home_position)
    assert pose.z > config.workspace.table_height + 0.10
    assert pose.pitch == pytest.approx(TOP_DOWN_PITCH, abs=1e-3)


def test_observe_pose_clears_the_pickup_zone(config: SampleSortConfig) -> None:
    pose = forward_kinematics(config.arm, config.arm.observe_position)
    # The arm must fold back short of the pickup zone so the camera sees it all.
    assert pose.x < config.workspace.pickup_zone.x_min


def test_forward_kinematics_rejects_wrong_length(config: SampleSortConfig) -> None:
    with pytest.raises(ValueError, match="expected 5 joint values"):
        forward_kinematics(config.arm, [0.0, 0.0])


def test_zero_configuration_is_fully_extended(config: SampleSortConfig) -> None:
    pose = forward_kinematics(config.arm, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert pose.x == pytest.approx(config.arm.max_reach)
    assert pose.y == pytest.approx(0.0)
    assert pose.z == pytest.approx(config.arm.base_height)


def test_base_yaw_rotates_the_tool_about_z(config: SampleSortConfig) -> None:
    straight = forward_kinematics(config.arm, [0.0, 0.3, -0.6, 0.2, 0.0])
    turned = forward_kinematics(config.arm, [math.pi / 2, 0.3, -0.6, 0.2, 0.0])
    assert turned.x == pytest.approx(straight.y, abs=1e-9)
    assert turned.y == pytest.approx(straight.x, abs=1e-9)
    assert turned.z == pytest.approx(straight.z)


def test_ik_round_trips_to_machine_precision(config: SampleSortConfig) -> None:
    rng = np.random.default_rng(12345)
    checked = 0
    for _ in range(500):
        q = _random_valid_joints(config, rng)
        pose = forward_kinematics(config.arm, q)
        try:
            solution = inverse_kinematics(config.arm, pose)
        except UnreachableError:
            continue
        assert pose_error(forward_kinematics(config.arm, solution), pose) < 1e-9
        checked += 1
    assert checked > 100, "IK round-trip covered too few configurations to be meaningful"


def test_ik_matches_fk_on_a_known_pose(config: SampleSortConfig) -> None:
    target = Pose(0.20, 0.05, 0.12, pitch=TOP_DOWN_PITCH)
    q = inverse_kinematics(config.arm, target)
    achieved = forward_kinematics(config.arm, q)
    assert achieved.x == pytest.approx(target.x, abs=1e-9)
    assert achieved.y == pytest.approx(target.y, abs=1e-9)
    assert achieved.z == pytest.approx(target.z, abs=1e-9)
    assert achieved.pitch == pytest.approx(target.pitch, abs=1e-9)


def test_ik_respects_joint_limits(config: SampleSortConfig) -> None:
    target = Pose(0.22, -0.08, 0.10)
    q = inverse_kinematics(config.arm, target)
    for value, (low, high) in zip(q, config.arm.joint_limits, strict=True):
        assert low - 1e-9 <= value <= high + 1e-9


def test_ik_carries_roll_through(config: SampleSortConfig) -> None:
    q = inverse_kinematics(config.arm, Pose(0.20, 0.0, 0.12, roll=0.7))
    assert q[4] == pytest.approx(0.7)


def test_far_pose_is_unreachable(config: SampleSortConfig) -> None:
    with pytest.raises(UnreachableError, match="can only reach"):
        inverse_kinematics(config.arm, Pose(2.0, 0.0, 0.1))


def test_pose_inside_the_dead_zone_is_unreachable(config: SampleSortConfig) -> None:
    # Equal link lengths give no dead zone, so build a stubby asymmetric arm.
    stubby = config.arm.model_copy(update={"link_lengths": [0.30, 0.10, 0.02]})
    with pytest.raises(UnreachableError, match="dead zone"):
        inverse_kinematics(stubby, Pose(0.02, 0.0, stubby.base_height))


def test_unreachable_reports_limit_violations_separately(config: SampleSortConfig) -> None:
    # Geometrically fine, but pinning every joint to zero makes it unsolvable.
    pinned = config.arm.model_copy(update={"joint_limits": [(-0.01, 0.01)] * 5})
    with pytest.raises(UnreachableError, match="violates the joint limits"):
        inverse_kinematics(pinned, Pose(0.20, 0.05, 0.12))


def test_both_elbow_branches_reach_the_same_pose(config: SampleSortConfig) -> None:
    target = Pose(0.20, 0.0, 0.12)
    up = inverse_kinematics(config.arm, target, prefer=ElbowBranch.UP)
    down = inverse_kinematics(config.arm, target, prefer=ElbowBranch.DOWN)
    assert pose_error(forward_kinematics(config.arm, up), target) < 1e-9
    assert pose_error(forward_kinematics(config.arm, down), target) < 1e-9


def test_elbow_up_is_preferred_when_it_is_legal(config: SampleSortConfig) -> None:
    q = inverse_kinematics(config.arm, Pose(0.20, 0.0, 0.12), prefer=ElbowBranch.UP)
    assert q[2] <= 0.0, "elbow-up should give a non-positive elbow angle"


def test_every_pickup_and_rack_pose_is_reachable(config: SampleSortConfig) -> None:
    ws = config.workspace
    heights = [ws.table_grasp_height, ws.table_grasp_height + ws.approach_clearance]
    zone = ws.pickup_zone
    corners = [
        (zone.x_min, zone.y_min),
        (zone.x_min, zone.y_max),
        (zone.x_max, zone.y_min),
        (zone.x_max, zone.y_max),
    ]
    for x, y in corners:
        for z in heights:
            assert is_reachable(config.arm, Pose(x, y, z)), f"pickup corner ({x}, {y}) at z={z}"

    rack_heights = [ws.rack_grasp_height, ws.rack_grasp_height + ws.approach_clearance]
    for rack in ws.racks:
        for x, y in rack.slots:
            for z in rack_heights:
                assert is_reachable(config.arm, Pose(x, y, z)), f"{rack.id} slot ({x}, {y})"


def test_link_positions_form_a_connected_chain(config: SampleSortConfig) -> None:
    q = np.array(config.arm.home_position)
    points = link_positions(config.arm, q)
    assert points.shape == (5, 3)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(points[1], [0.0, 0.0, config.arm.base_height], atol=1e-12)

    lengths = [float(np.linalg.norm(b - a)) for a, b in zip(points[1:], points[2:], strict=False)]
    np.testing.assert_allclose(lengths, config.arm.link_lengths, atol=1e-12)

    tcp = forward_kinematics(config.arm, q)
    np.testing.assert_allclose(points[-1], tcp.position, atol=1e-12)


def test_pose_helpers() -> None:
    pose = Pose(0.1, 0.2, 0.3, pitch=-1.0, roll=0.5)
    assert pose.with_z(0.9).z == pytest.approx(0.9)
    assert pose.offset_z(0.05).z == pytest.approx(0.35)
    assert pose.offset_z(0.05).pitch == pytest.approx(-1.0)
    np.testing.assert_allclose(pose.position, [0.1, 0.2, 0.3])
