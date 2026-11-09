"""Abstract robot-arm interface shared by the simulated and real backends."""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence

import numpy as np

from samplesort.config import ArmConfig

logger = logging.getLogger(__name__)

#: Any sequence of joint angles: a list, tuple or NumPy array.
JointVector = Sequence[float] | np.ndarray


class ArmError(RuntimeError):
    """Raised when an arm command is invalid or the hardware refuses it."""


class JointLimitError(ArmError):
    """Raised when a commanded joint configuration violates the configured limits."""


class ArmInterface(abc.ABC):
    """Minimal contract every arm backend implements.

    The rest of SampleSort only ever talks to this interface, so swapping the
    PyBullet arm for a real SO-101 is a config change rather than a code change.
    """

    def __init__(self, config: ArmConfig) -> None:
        """Store the arm configuration.

        Args:
            config: Validated arm geometry, joint limits and wiring.
        """
        self.config = config
        self._connected = False

    # ---------------------------------------------------------------- lifecycle

    @abc.abstractmethod
    def connect(self) -> None:
        """Bring the arm up and make it ready to accept commands."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release the arm and any resources it holds."""

    @property
    def is_connected(self) -> bool:
        """Whether :meth:`connect` has succeeded and :meth:`disconnect` has not run."""
        return self._connected

    # ------------------------------------------------------------------ motion

    @abc.abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Return the current joint angles in radians, shape ``(num_joints,)``."""

    @abc.abstractmethod
    def move_to_joints(self, q: JointVector, duration: float | None = None) -> None:
        """Drive the arm to a joint configuration.

        Args:
            q: Target joint angles in radians, one per arm joint.
            duration: Seconds the motion should take. ``None`` uses the configured
                default move duration.

        Raises:
            JointLimitError: If ``q`` violates the configured joint limits.
            ArmError: If the arm is not connected.
        """

    @abc.abstractmethod
    def open_gripper(self) -> None:
        """Open the gripper to its configured open width."""

    @abc.abstractmethod
    def close_gripper(self) -> None:
        """Close the gripper onto whatever sits between the fingers."""

    def go_home(self, duration: float | None = None) -> None:
        """Move to the configured home pose.

        Args:
            duration: Seconds the motion should take, or ``None`` for the default.
        """
        self.move_to_joints(self.config.home_position, duration)

    # ------------------------------------------------------------------ helpers

    def clamp_to_limits(self, q: JointVector) -> np.ndarray:
        """Clamp a joint vector into the configured limits.

        Args:
            q: Joint angles in radians.

        Returns:
            A new array with every joint inside its limits.
        """
        array = np.asarray(q, dtype=float)
        lows = np.array([low for low, _ in self.config.joint_limits])
        highs = np.array([high for _, high in self.config.joint_limits])
        return np.asarray(np.clip(array, lows, highs), dtype=float)

    def validate_joints(self, q: JointVector, *, tolerance: float = 1e-6) -> np.ndarray:
        """Check a joint vector against the configured limits.

        Args:
            q: Joint angles in radians.
            tolerance: Slack allowed before a value counts as out of range, which
                absorbs floating-point noise from the IK solver.

        Returns:
            The validated joint vector as a float array.

        Raises:
            JointLimitError: If the vector has the wrong length or any joint is
                outside its limits by more than ``tolerance``.
        """
        array = np.asarray(q, dtype=float)
        expected = self.config.num_joints
        if array.shape != (expected,):
            raise JointLimitError(
                f"expected {expected} joint values, got shape {tuple(array.shape)}"
            )
        if not np.all(np.isfinite(array)):
            raise JointLimitError(f"joint vector contains non-finite values: {array.tolist()}")
        for name, value, (low, high) in zip(
            self.config.joint_names, array, self.config.joint_limits, strict=True
        ):
            if value < low - tolerance or value > high + tolerance:
                raise JointLimitError(
                    f"joint '{name}' target {value:.4f} rad is outside its "
                    f"limits [{low:.4f}, {high:.4f}]"
                )
        return array

    def _require_connected(self) -> None:
        """Raise if the arm has not been connected yet."""
        if not self._connected:
            raise ArmError(f"{type(self).__name__} is not connected; call connect() first")

    def __enter__(self) -> ArmInterface:
        """Connect on entry to a ``with`` block."""
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Disconnect on exit from a ``with`` block."""
        self.disconnect()
