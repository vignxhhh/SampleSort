"""Shared assertion helpers for inspecting generated joint paths.

These live with the tests rather than in ``samplesort.control.trajectory``
because nothing in the shipped library needs them — they exist to state
invariants about the paths the arms generate.
"""

from __future__ import annotations

import numpy as np


def max_joint_step(path: np.ndarray) -> float:
    """Largest single-joint change between consecutive waypoints, in radians."""
    if len(path) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(path, axis=0))))


def is_monotonic(path: np.ndarray, *, tolerance: float = 1e-9) -> bool:
    """Whether every joint moves monotonically along the path.

    Smoothstep interpolation between two endpoints never overshoots, so this
    holds for any single-segment path.

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
