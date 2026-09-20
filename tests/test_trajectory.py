"""Tests for joint-space trajectory interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from samplesort.config import SampleSortConfig
from samplesort.control.trajectory import duration_for, interpolate, smoothstep
from tests.assertions import is_monotonic, max_joint_step


def test_smoothstep_endpoints_and_midpoint() -> None:
    assert smoothstep(0.0) == pytest.approx(0.0)
    assert smoothstep(1.0) == pytest.approx(1.0)
    assert smoothstep(0.5) == pytest.approx(0.5)


def test_smoothstep_clamps_out_of_range_input() -> None:
    assert smoothstep(-3.0) == pytest.approx(0.0)
    assert smoothstep(4.0) == pytest.approx(1.0)


def test_smoothstep_has_zero_slope_at_the_ends() -> None:
    eps = 1e-6
    assert smoothstep(eps) / eps < 1e-3
    assert (1.0 - smoothstep(1.0 - eps)) / eps < 1e-3


def test_interpolate_shape_and_endpoint() -> None:
    path = interpolate([0.0, 0.0], [1.0, -2.0], 10)
    assert path.shape == (10, 2)
    np.testing.assert_allclose(path[-1], [1.0, -2.0])


def test_interpolate_is_monotonic_and_bounded() -> None:
    path = interpolate([0.0, 1.0], [1.0, -1.0], 25)
    assert is_monotonic(path)
    assert path[:, 0].min() >= 0.0 and path[:, 0].max() <= 1.0
    assert path[:, 1].min() >= -1.0 and path[:, 1].max() <= 1.0


def test_interpolate_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="same shape"):
        interpolate([0.0], [0.0, 1.0], 5)
    with pytest.raises(ValueError, match="at least 1"):
        interpolate([0.0], [1.0], 0)


def test_interpolating_a_zero_move_stays_put() -> None:
    path = interpolate([0.5, -0.5], [0.5, -0.5], 5)
    np.testing.assert_allclose(path, np.tile([0.5, -0.5], (5, 1)))


def test_more_steps_means_smaller_increments() -> None:
    coarse = max_joint_step(interpolate([0.0], [1.0], 5))
    fine = max_joint_step(interpolate([0.0], [1.0], 50))
    assert fine < coarse


def test_duration_scales_with_travel(config: SampleSortConfig) -> None:
    short = duration_for(config.arm, [0.0] * 5, [0.1, 0, 0, 0, 0])
    long = duration_for(config.arm, [0.0] * 5, [1.0, 0, 0, 0, 0])
    assert long > short
    assert long == pytest.approx(1.5 * 1.0 / config.arm.max_joint_velocity)


def test_duration_has_a_floor(config: SampleSortConfig) -> None:
    assert duration_for(config.arm, [0.0] * 5, [0.0] * 5) == pytest.approx(0.05)


def test_duration_respects_max_velocity(config: SampleSortConfig) -> None:
    start, end = [0.0] * 5, [0.0, 0.8, 0.0, 0.0, 0.0]
    seconds = duration_for(config.arm, start, end)
    path = interpolate(start, end, int(seconds * 240))
    peak_velocity = max_joint_step(path) * 240
    assert peak_velocity <= config.arm.max_joint_velocity * 1.05


def test_is_monotonic_detects_a_reversal() -> None:
    assert not is_monotonic(np.array([[0.0], [1.0], [0.5]]))
    assert is_monotonic(np.array([[0.0]]))
