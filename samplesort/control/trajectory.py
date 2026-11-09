"""Smooth joint-space interpolation between waypoints.

Both arm backends and the scripted controller drive motion through these three
functions: :func:`smoothstep` shapes the time profile, :func:`interpolate` turns
a pair of configurations into eased setpoints, and :func:`duration_for` times a
move so no joint exceeds its configured maximum velocity.
"""

from __future__ import annotations

import logging

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
