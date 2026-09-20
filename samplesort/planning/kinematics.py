"""Forward and inverse kinematics for the SampleSort 5-DOF arm.

The arm is a yaw joint followed by three coplanar pitch joints and a tool roll.
That structure has a closed-form solution, so the IK here is analytic rather than
numerical: it is exact, fast, branch-aware (elbow-up vs elbow-down) and easy to
test by round-tripping through :func:`forward_kinematics`.

Conventions, matching the generated URDF:

* ``q = [base_yaw, shoulder, elbow, wrist_pitch, wrist_roll]`` in radians.
* Pitch joints use the axis ``(0, -1, 0)``, so a positive angle raises the link
  and the elevation of link *i* is the running sum ``q1 + ... + qi``.
* ``pitch`` is the elevation of the tool axis: ``-pi/2`` points straight down,
  which is what a top-down tube grasp uses.
* All positions are in the arm base frame, in metres.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from samplesort.config import ArmConfig

logger = logging.getLogger(__name__)

#: Tool pitch for a straight-down, top-down grasp.
TOP_DOWN_PITCH = -math.pi / 2.0


class ElbowBranch(str, Enum):
    """Which of the two analytic IK solutions to prefer."""

    UP = "up"
    DOWN = "down"


class UnreachableError(ValueError):
    """Raised when a pose lies outside the arm's workspace or violates joint limits."""


@dataclass(frozen=True)
class Pose:
    """A tool pose the arm can be asked to reach.

    Attributes:
        x: Base-frame X in metres.
        y: Base-frame Y in metres.
        z: Base-frame Z in metres.
        pitch: Elevation of the tool axis in radians; ``-pi/2`` points down.
        roll: Rotation about the tool axis in radians.
    """

    x: float
    y: float
    z: float
    pitch: float = TOP_DOWN_PITCH
    roll: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """The pose's XYZ position as a 3-element array."""
        return np.array([self.x, self.y, self.z], dtype=float)

    def with_z(self, z: float) -> Pose:
        """Return a copy of this pose at a different height."""
        return Pose(self.x, self.y, z, self.pitch, self.roll)

    def offset_z(self, dz: float) -> Pose:
        """Return a copy of this pose raised by ``dz`` metres."""
        return self.with_z(self.z + dz)


def forward_kinematics(config: ArmConfig, q: np.ndarray | list[float]) -> Pose:
    """Compute the tool pose for a joint configuration.

    Args:
        config: Arm geometry.
        q: Five joint angles in radians.

    Returns:
        The resulting tool centre point pose.

    Raises:
        ValueError: If ``q`` does not hold exactly five values.
    """
    angles = np.asarray(q, dtype=float)
    if angles.shape != (config.num_joints,):
        raise ValueError(f"expected {config.num_joints} joint values, got {angles.shape}")

    l1, l2, l3 = config.link_lengths
    yaw, q1, q2, q3, roll = (float(v) for v in angles)

    a1 = q1
    a2 = q1 + q2
    a3 = q1 + q2 + q3

    radius = l1 * math.cos(a1) + l2 * math.cos(a2) + l3 * math.cos(a3)
    height = config.base_height + l1 * math.sin(a1) + l2 * math.sin(a2) + l3 * math.sin(a3)

    return Pose(
        x=radius * math.cos(yaw),
        y=radius * math.sin(yaw),
        z=height,
        pitch=a3,
        roll=roll,
    )


def link_positions(config: ArmConfig, q: np.ndarray | list[float]) -> np.ndarray:
    """Return the base-frame XYZ of every joint along the chain.

    Args:
        config: Arm geometry.
        q: Five joint angles in radians.

    Returns:
        A ``(5, 3)`` array: base, shoulder, elbow, wrist and tool centre point.
    """
    angles = np.asarray(q, dtype=float)
    l1, l2, l3 = config.link_lengths
    yaw, q1, q2, q3 = (float(v) for v in angles[:4])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    points = [(0.0, 0.0), (0.0, config.base_height)]
    radius, height = 0.0, config.base_height
    for length, elevation in ((l1, q1), (l2, q1 + q2), (l3, q1 + q2 + q3)):
        radius += length * math.cos(elevation)
        height += length * math.sin(elevation)
        points.append((radius, height))

    return np.array([[r * cos_yaw, r * sin_yaw, h] for r, h in points], dtype=float)


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle into ``[-pi, pi]``."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _within_limits(config: ArmConfig, q: np.ndarray, tolerance: float) -> bool:
    """Whether every joint in ``q`` sits inside its configured limit."""
    return all(
        low - tolerance <= value <= high + tolerance
        for value, (low, high) in zip(q, config.joint_limits, strict=True)
    )


def inverse_kinematics(
    config: ArmConfig,
    pose: Pose,
    *,
    prefer: ElbowBranch = ElbowBranch.UP,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Solve for the joint angles that put the tool centre point at ``pose``.

    Both analytic elbow branches are computed; the preferred one is returned when
    it respects the joint limits, otherwise the other one is. The base yaw is
    resolved so that the tool reaches the target from the near side.

    Args:
        config: Arm geometry and joint limits.
        pose: Target tool pose.
        prefer: Which elbow branch to try first.
        tolerance: Slack allowed when checking joint limits.

    Returns:
        Five joint angles in radians.

    Raises:
        UnreachableError: If the pose is outside the workspace, or every branch
            violates the joint limits.
    """
    l1, l2, l3 = config.link_lengths

    yaw = math.atan2(pose.y, pose.x)
    radius = math.hypot(pose.x, pose.y)

    # Wrist centre, walking back along the tool axis from the TCP.
    wrist_r = radius - l3 * math.cos(pose.pitch)
    wrist_z = (pose.z - config.base_height) - l3 * math.sin(pose.pitch)

    reach = math.hypot(wrist_r, wrist_z)
    max_reach, min_reach = l1 + l2, abs(l1 - l2)
    if reach > max_reach + 1e-9:
        raise UnreachableError(
            f"pose ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) needs a wrist reach of "
            f"{reach:.4f} m but the arm can only reach {max_reach:.4f} m"
        )
    if reach < min_reach - 1e-9:
        raise UnreachableError(
            f"pose ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) is inside the arm's "
            f"{min_reach:.4f} m dead zone"
        )

    cos_elbow = (reach**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    elbow_magnitude = math.acos(max(-1.0, min(1.0, cos_elbow)))

    # Our sign convention makes a negative elbow angle the "elbow up" posture.
    branches = {ElbowBranch.UP: -elbow_magnitude, ElbowBranch.DOWN: elbow_magnitude}
    order = [prefer, ElbowBranch.DOWN if prefer is ElbowBranch.UP else ElbowBranch.UP]

    attempted: list[np.ndarray] = []
    for branch in order:
        q2 = branches[branch]
        q1 = math.atan2(wrist_z, wrist_r) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        q3 = pose.pitch - q1 - q2
        solution = np.array(
            [_wrap_to_pi(yaw), _wrap_to_pi(q1), _wrap_to_pi(q2), _wrap_to_pi(q3), pose.roll],
            dtype=float,
        )
        attempted.append(solution)
        if _within_limits(config, solution, tolerance):
            return solution

    formatted = "; ".join(np.array2string(s, precision=3) for s in attempted)
    raise UnreachableError(
        f"pose ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) is geometrically reachable but "
        f"every elbow branch violates the joint limits: {formatted}"
    )


def is_reachable(config: ArmConfig, pose: Pose, *, prefer: ElbowBranch = ElbowBranch.UP) -> bool:
    """Whether :func:`inverse_kinematics` can solve ``pose`` within the joint limits."""
    try:
        inverse_kinematics(config, pose, prefer=prefer)
    except UnreachableError:
        return False
    return True


def pose_error(a: Pose, b: Pose) -> float:
    """Euclidean distance in metres between two poses' positions."""
    return float(np.linalg.norm(a.position - b.position))
