"""Smooth joint-space interpolation between waypoints."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from samplesort.config import ArmConfig
from samplesort.hal.arm import JointVector

logger = logging.getLogger(__name__)


def smoothstep(alpha: float) -> float:
    """Ease a normalised time in ``[0, 1]`` to zero velocity at both ends.

    Args:
        alpha: Normalised time, clamped into ``[0, 1]``.

    Returns:
        The eased value, also in ``[0, 1]``.
    """
    clamped = min(1.0, max(0.0, alpha))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def interpolate(start: JointVector, end: JointVector, steps: int) -> np.ndarray:
    """Build a smoothed joint path between two configurations.

    Args:
        start: Starting joint angles.
        end: Target joint angles.
        steps: Number of waypoints to emit, including ``end`` but excluding ``start``.

    Returns:
        A ``(steps, num_joints)`` array of joint angles.

    Raises:
        ValueError: If ``steps`` is not positive or the endpoints differ in length.
    """
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"start and end must have the same shape, got {a.shape} and {b.shape}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    alphas = np.array([smoothstep((i + 1) / steps) for i in range(steps)], dtype=float)
    return np.asarray(a[None, :] + (b - a)[None, :] * alphas[:, None], dtype=float)


def duration_for(config: ArmConfig, start: JointVector, end: JointVector) -> float:
    """Time a joint move so no joint exceeds the configured maximum velocity.

    A smoothstep profile peaks at 1.5x the average velocity, so the required time
    is scaled accordingly.

    Args:
        config: Arm configuration supplying ``max_joint_velocity``.
        start: Starting joint angles.
        end: Target joint angles.

    Returns:
        A duration in seconds, never below a 50 ms floor.
    """
    travel = float(np.max(np.abs(np.asarray(end, dtype=float) - np.asarray(start, dtype=float))))
    if travel <= 0.0:
        return 0.05
    peak_factor = 1.5
    return max(0.05, peak_factor * travel / config.max_joint_velocity)


def path_length(waypoints: Sequence[JointVector]) -> float:
    """Total joint-space path length across a sequence of waypoints.

    Args:
        waypoints: Ordered joint configurations.

    Returns:
        The summed Euclidean distance between consecutive waypoints, in radians.
    """
    points = [np.asarray(w, dtype=float) for w in waypoints]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:], strict=False)))


def resample(waypoints: Sequence[JointVector], steps_per_segment: int) -> np.ndarray:
    """Expand a coarse waypoint list into a smoothly interpolated path.

    Args:
        waypoints: At least two ordered joint configurations.
        steps_per_segment: Interpolation steps to insert between each pair.

    Returns:
        A ``(1 + (len(waypoints) - 1) * steps_per_segment, num_joints)`` array
        starting at the first waypoint.

    Raises:
        ValueError: If fewer than two waypoints are supplied.
    """
    if len(waypoints) < 2:
        raise ValueError("resample needs at least two waypoints")
    path = [np.asarray(waypoints[0], dtype=float)]
    for a, b in zip(waypoints, waypoints[1:], strict=False):
        path.extend(interpolate(a, b, steps_per_segment))
    return np.asarray(np.vstack(path), dtype=float)


def max_joint_step(path: np.ndarray) -> float:
    """Largest single-joint change between consecutive waypoints, in radians."""
    if len(path) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(path, axis=0))))


def is_monotonic(path: np.ndarray, *, tolerance: float = 1e-9) -> bool:
    """Whether every joint moves monotonically along the path.

    A smoothstep interpolation between two endpoints never overshoots, so this
    holds for any single-segment path and is a useful invariant to assert on.

    Args:
        path: A ``(steps, num_joints)`` array.
        tolerance: Slack absorbing floating-point noise.
    """
    if len(path) < 2:
        return True
    deltas = np.diff(path, axis=0)
    rising = np.all(deltas >= -tolerance, axis=0)
    falling = np.all(deltas <= tolerance, axis=0)
    return bool(np.all(rising | falling))


def angular_distance(a: JointVector, b: JointVector) -> float:
    """Euclidean distance in radians between two joint configurations."""
    return float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)))
