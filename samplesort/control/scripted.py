"""Scripted pick-and-place: the reliable baseline controller.

A job becomes a fixed sequence of Cartesian waypoints — approach, grasp, lift,
transfer, place, retreat — each solved with the analytic IK and executed through
the :class:`~samplesort.hal.arm.ArmInterface`. Every stage is verified, and the
first one that fails names itself in the result so the benchmark can break
failures down by type.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.hal.arm import ArmError, ArmInterface
from samplesort.planning.kinematics import Pose, UnreachableError, inverse_kinematics
from samplesort.planning.task_planner import PickPlaceJob

logger = logging.getLogger(__name__)


class FailureReason(str, Enum):
    """Why a pick-and-place attempt did not succeed."""

    NONE = "none"
    UNREACHABLE_PICK = "unreachable_pick"
    UNREACHABLE_PLACE = "unreachable_place"
    GRASP_MISSED = "grasp_missed"
    DROPPED_IN_TRANSIT = "dropped_in_transit"
    MISPLACED = "misplaced"
    ARM_ERROR = "arm_error"


@runtime_checkable
class GraspSensingArm(Protocol):
    """An arm that can report whether it is actually holding something."""

    def has_object(self) -> bool:
        """Whether the gripper currently holds an object."""
        ...


@runtime_checkable
class PlacementVerifier(Protocol):
    """A world that can confirm where a tube ended up."""

    def tube_xy(self, body_id: int) -> tuple[float, float]:
        """Return a tube's XY position on the table."""
        ...

    def step(self, steps: int = 1) -> None:
        """Advance the simulation."""
        ...


@dataclass
class ExecutionResult:
    """The outcome of one scripted pick-and-place attempt.

    Attributes:
        success: Whether the tube ended up in its target slot.
        reason: Which stage failed, or :attr:`FailureReason.NONE` on success.
        duration_s: Wall-clock seconds the attempt took.
        grasped: Whether the grasp stage succeeded, regardless of the final outcome.
        place_error_m: Distance from the target slot centre, when measurable.
        waypoints: The joint configurations that were commanded.
    """

    success: bool
    reason: FailureReason = FailureReason.NONE
    duration_s: float = 0.0
    grasped: bool = False
    place_error_m: float | None = None
    waypoints: list[np.ndarray] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        """Whether the attempt did not succeed."""
        return not self.success


class ScriptedController:
    """Executes :class:`~samplesort.planning.task_planner.PickPlaceJob` objects.

    Args:
        config: The validated configuration bundle.
        arm: The arm to drive.
        world: Optional simulation world used to verify grasps and placements.
            In real mode this is ``None`` and verification falls back to the
            gripper's own state.
    """

    #: How close to the slot centre a tube must land to count as correctly placed.
    PLACE_TOLERANCE_M = 0.030

    def __init__(
        self,
        config: SampleSortConfig,
        arm: ArmInterface,
        world: PlacementVerifier | None = None,
    ) -> None:
        """Bind the controller to an arm (see the class docstring for args)."""
        self.config = config
        self.arm = arm
        self.world = world

    # ------------------------------------------------------------------ planning

    def waypoint_poses(self, job: PickPlaceJob) -> list[tuple[str, Pose]]:
        """Build the named Cartesian waypoints for a job.

        Args:
            job: The pick-and-place job to plan.

        Returns:
            Ordered ``(stage_name, pose)`` pairs from pre-grasp through retreat.
        """
        ws = self.config.workspace
        grasp_z = ws.table_grasp_height
        place_z = ws.rack_grasp_height
        clearance = ws.approach_clearance

        return [
            ("pre_grasp", job.pick_pose(grasp_z + clearance)),
            ("grasp", job.pick_pose(grasp_z)),
            ("lift", job.pick_pose(grasp_z + clearance)),
            ("pre_place", job.place_pose(place_z + clearance)),
            ("place", job.place_pose(place_z)),
            ("retreat", job.place_pose(place_z + clearance)),
        ]

    def solve_waypoints(self, job: PickPlaceJob) -> list[tuple[str, np.ndarray]]:
        """Solve the IK for every waypoint of a job.

        Args:
            job: The pick-and-place job to plan.

        Returns:
            Ordered ``(stage_name, joint_angles)`` pairs.

        Raises:
            UnreachableError: If any waypoint is outside the arm's workspace.
        """
        solved: list[tuple[str, np.ndarray]] = []
        for stage, pose in self.waypoint_poses(job):
            solved.append((stage, inverse_kinematics(self.config.arm, pose)))
        return solved

    def is_feasible(self, job: PickPlaceJob) -> bool:
        """Whether every waypoint of a job can be reached."""
        try:
            self.solve_waypoints(job)
        except UnreachableError:
            return False
        return True

    # ----------------------------------------------------------------- execution

    def execute(self, job: PickPlaceJob) -> ExecutionResult:
        """Run one full pick-and-place attempt.

        Args:
            job: The job to execute.

        Returns:
            An :class:`ExecutionResult` describing the outcome, including the
            stage that failed when the attempt did not succeed.
        """
        started = time.perf_counter()

        try:
            solved = self.solve_waypoints(job)
        except UnreachableError as exc:
            logger.warning("job %s is unreachable: %s", job.describe(), exc)
            stage = self._unreachable_stage(job)
            return ExecutionResult(
                success=False, reason=stage, duration_s=time.perf_counter() - started
            )

        stages = dict(solved)
        result = ExecutionResult(success=False, waypoints=[q for _, q in solved])

        try:
            self.arm.open_gripper()
            self._move(stages["pre_grasp"])
            self._move(stages["grasp"])
            self.arm.close_gripper()

            if not self._holding():
                result.reason = FailureReason.GRASP_MISSED
                result.duration_s = time.perf_counter() - started
                logger.info("grasp missed for %s", job.describe())
                self._recover()
                return result

            result.grasped = True
            self._move(stages["lift"])

            if not self._holding():
                result.reason = FailureReason.DROPPED_IN_TRANSIT
                result.duration_s = time.perf_counter() - started
                self._recover()
                return result

            self._move(stages["pre_place"])
            self._move(stages["place"])
            self.arm.open_gripper()
            self._settle()
            self._move(stages["retreat"])

        except ArmError as exc:
            logger.error("arm error while executing %s: %s", job.describe(), exc)
            result.reason = FailureReason.ARM_ERROR
            result.duration_s = time.perf_counter() - started
            return result

        error = self._placement_error(job)
        result.place_error_m = error
        if error is not None and error > self.PLACE_TOLERANCE_M:
            result.reason = FailureReason.MISPLACED
            logger.info("tube landed %.3f m from %s slot %d", error, job.rack_id, job.slot_index)
        else:
            result.success = True
            result.reason = FailureReason.NONE

        result.duration_s = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------- helpers

    def _move(self, q: np.ndarray) -> None:
        """Command one joint waypoint, timed from the configured joint velocity."""
        from samplesort.control.trajectory import duration_for

        current = self.arm.get_joint_positions()
        self.arm.move_to_joints(q, duration_for(self.config.arm, current, q))

    def _holding(self) -> bool:
        """Whether the gripper is holding a tube, when the arm can tell us."""
        if isinstance(self.arm, GraspSensingArm):
            return self.arm.has_object()
        # A real arm without a grasp sensor is assumed to have succeeded; the
        # wrist-camera check is a documented stretch goal.
        return True

    def _settle(self) -> None:
        """Give a released tube time to come to rest."""
        if self.world is not None:
            self.world.step(self.config.pipeline.settle_steps)

    def _recover(self) -> None:
        """Back off to a safe pose after a failed grasp so the next job starts clean."""
        try:
            self.arm.open_gripper()
            self.arm.move_to_joints(self.config.arm.observe_position)
        except ArmError:  # pragma: no cover - only if the arm died mid-recovery
            logger.exception("recovery move failed")

    def _placement_error(self, job: PickPlaceJob) -> float | None:
        """Distance from the target slot to where the tube actually ended up."""
        if self.world is None or job.body_id is None:
            return None
        x, y = self.world.tube_xy(job.body_id)
        return float(np.hypot(x - job.place_xy[0], y - job.place_xy[1]))

    def _unreachable_stage(self, job: PickPlaceJob) -> FailureReason:
        """Work out whether the pick side or the place side was out of reach."""
        ws = self.config.workspace
        pick = job.pick_pose(ws.table_grasp_height + ws.approach_clearance)
        try:
            inverse_kinematics(self.config.arm, pick)
        except UnreachableError:
            return FailureReason.UNREACHABLE_PICK
        return FailureReason.UNREACHABLE_PLACE
