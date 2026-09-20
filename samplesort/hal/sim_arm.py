"""PyBullet implementation of :class:`~samplesort.hal.arm.ArmInterface`."""

from __future__ import annotations

import logging

import numpy as np

from samplesort.config import ArmConfig
from samplesort.hal.arm import ArmError, ArmInterface, JointVector
from samplesort.sim._bullet import pb
from samplesort.sim.world import SIM_TIMESTEP, SimWorld

logger = logging.getLogger(__name__)

#: Joint motor force used for position control, in newton-metres.
_JOINT_FORCE = 40.0
#: Finger motor force, in newtons.
_FINGER_FORCE = 25.0
#: Maximum force the grasp constraint can carry before it breaks.
_GRASP_CONSTRAINT_FORCE = 60.0


class SimArm(ArmInterface):
    """Position-controlled arm inside a :class:`~samplesort.sim.world.SimWorld`.

    Grasping is modelled with a fixed constraint rather than friction-based
    finger contact. Closing the gripper succeeds only when a tube's cap is inside
    the configured ``grasp_tolerance_xy`` / ``grasp_tolerance_z`` box around the
    tool centre point, so positioning errors still produce realistic failures
    while the outcome stays deterministic for benchmarking.

    Args:
        config: Arm geometry and limits.
        world: The simulation world that owns the arm body.
    """

    def __init__(self, config: ArmConfig, world: SimWorld) -> None:
        """Bind the arm to a simulation world (see the class docstring for args)."""
        super().__init__(config)
        self.world = world
        self._grasp_constraint: int | None = None
        self._grasped_body: int | None = None
        self._gripper_open = True

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> None:
        """Ensure the world is running and park the arm at its home pose."""
        if not self.world.is_connected:
            self.world.connect()
        self._connected = True
        self.world.reset_arm(self.config.home_position)
        self.open_gripper()
        logger.debug("SimArm connected")

    def disconnect(self) -> None:
        """Release any grasped object and mark the arm as disconnected."""
        if self._connected and self.world.is_connected:
            self.release()
        self._connected = False
        logger.debug("SimArm disconnected")

    # ------------------------------------------------------------------ motion

    def get_joint_positions(self) -> np.ndarray:
        """Return the current joint angles in radians."""
        self._require_connected()
        client = self.world.bullet
        states = client.getJointStates(self.world.arm_id, self.world.joint_indices)
        return np.array([state[0] for state in states], dtype=float)

    def move_to_joints(self, q: JointVector, duration: float | None = None) -> None:
        """Drive the arm to ``q`` over ``duration`` seconds of simulated time.

        Setpoints are eased with a smooth-step profile so the arm accelerates and
        decelerates instead of jerking between waypoints.

        Args:
            q: Target joint angles in radians.
            duration: Motion duration in seconds, or ``None`` for the configured default.

        Raises:
            JointLimitError: If ``q`` violates the configured joint limits.
            ArmError: If the arm is not connected.
        """
        self._require_connected()
        target = self.validate_joints(q)
        start = self.get_joint_positions()
        seconds = self.config.default_move_duration if duration is None else float(duration)
        seconds = max(seconds, SIM_TIMESTEP)

        steps = max(1, int(round(seconds / SIM_TIMESTEP)))
        for step in range(1, steps + 1):
            alpha = step / steps
            # Smooth-step easing: zero velocity at both ends.
            eased = alpha * alpha * (3.0 - 2.0 * alpha)
            setpoint = start + (target - start) * eased
            self._command_joints(setpoint)
            self.world.step(1)

        # Hold the final target so the arm does not drift while other work happens.
        self._command_joints(target)

    def _command_joints(self, q: np.ndarray) -> None:
        """Send one position-control setpoint to every arm joint."""
        client = self.world.bullet
        client.setJointMotorControlArray(
            self.world.arm_id,
            self.world.joint_indices,
            pb.POSITION_CONTROL,
            targetPositions=[float(v) for v in q],
            forces=[_JOINT_FORCE] * len(self.world.joint_indices),
        )

    # ----------------------------------------------------------------- gripper

    def open_gripper(self) -> None:
        """Open the fingers and drop anything currently held."""
        self._require_connected()
        self.release()
        self._set_finger_target(0.0)
        self.world.step(30)
        self._gripper_open = True

    def close_gripper(self) -> None:
        """Close the fingers and attach a tube if one is within grasp tolerance."""
        self._require_connected()
        travel = (self.config.gripper.open_width - self.config.gripper.closed_width) / 2.0
        self._set_finger_target(travel)
        self.world.step(30)
        self._gripper_open = False
        self._try_attach()

    def _set_finger_target(self, travel: float) -> None:
        """Command both prismatic fingers to the same inward travel, in metres."""
        client = self.world.bullet
        client.setJointMotorControlArray(
            self.world.arm_id,
            self.world.finger_indices,
            pb.POSITION_CONTROL,
            targetPositions=[travel, travel],
            forces=[_FINGER_FORCE] * len(self.world.finger_indices),
        )

    @property
    def gripper_is_open(self) -> bool:
        """Whether the last gripper command was an open."""
        return self._gripper_open

    @property
    def grasped_body(self) -> int | None:
        """PyBullet body id of the currently held tube, if any."""
        return self._grasped_body

    def has_object(self) -> bool:
        """Whether the gripper is currently holding a tube."""
        return self._grasped_body is not None

    # ------------------------------------------------------------- attach/detach

    def nearest_graspable(self) -> tuple[int, float] | None:
        """Find the closest tube whose cap lies inside the grasp tolerance box.

        Returns:
            ``(body_id, horizontal_distance)`` for the best candidate, or ``None``
            when nothing is close enough.
        """
        tool_xyz, _ = self.world.tool_pose()
        grip = self.config.gripper
        best: tuple[int, float] | None = None
        for tube in self.world.tubes:
            cap = self.world.tube_cap_position(tube.body_id)
            horizontal = float(np.hypot(cap[0] - tool_xyz[0], cap[1] - tool_xyz[1]))
            vertical = float(abs(cap[2] - tool_xyz[2]))
            in_tolerance = (
                horizontal <= grip.grasp_tolerance_xy and vertical <= grip.grasp_tolerance_z
            )
            if in_tolerance and (best is None or horizontal < best[1]):
                best = (tube.body_id, horizontal)
        return best

    def _try_attach(self) -> None:
        """Create a fixed constraint to the nearest graspable tube, if there is one."""
        candidate = self.nearest_graspable()
        if candidate is None:
            logger.debug("close_gripper found no tube within grasp tolerance")
            return

        body_id, distance = candidate
        client = self.world.bullet
        tool_pos, tool_orn = client.getLinkState(
            self.world.arm_id, self.world.tool_link_index, computeForwardKinematics=1
        )[4:6]
        body_pos, body_orn = client.getBasePositionAndOrientation(body_id)

        inv_pos, inv_orn = client.invertTransform(tool_pos, tool_orn)
        rel_pos, rel_orn = client.multiplyTransforms(inv_pos, inv_orn, body_pos, body_orn)

        self._grasp_constraint = client.createConstraint(
            parentBodyUniqueId=self.world.arm_id,
            parentLinkIndex=self.world.tool_link_index,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=list(rel_pos),
            parentFrameOrientation=list(rel_orn),
            childFramePosition=[0.0, 0.0, 0.0],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        client.changeConstraint(self._grasp_constraint, maxForce=_GRASP_CONSTRAINT_FORCE)
        self._grasped_body = body_id
        logger.debug("grasped tube %d at %.4f m from the TCP", body_id, distance)

    def release(self) -> None:
        """Remove the grasp constraint so the held tube falls free.

        Safe to call when nothing is held, when the constraint has already gone,
        or after the world has been torn down.
        """
        if self._grasp_constraint is not None and self.world.is_connected:
            try:
                self.world.bullet.removeConstraint(self._grasp_constraint)
            except pb.error:  # pragma: no cover - constraint already gone
                logger.debug("grasp constraint %s was already removed", self._grasp_constraint)
        self._grasp_constraint = None
        self._grasped_body = None

    def tool_position(self) -> np.ndarray:
        """Return the tool centre point in world coordinates.

        Raises:
            ArmError: If the arm is not connected.
        """
        if not self.world.is_connected:
            raise ArmError("simulation world is not connected")
        position, _ = self.world.tool_pose()
        return position
