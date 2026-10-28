"""Scripted pick-and-place: the reliable baseline controller.

A job becomes a fixed sequence of Cartesian waypoints — approach, grasp, lift,
transfer, place, retreat — each solved with the analytic IK and executed through
the :class:`~samplesort.hal.arm.ArmInterface`. Every stage is verified, and the
first one that fails names itself in the result so the benchmark can break
failures down by type.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.hal.arm import ArmError, ArmInterface
from samplesort.hal.scanner import IdScannerInterface
from samplesort.perception.types import FallenTube
from samplesort.planning.kinematics import (
    TOP_DOWN_PITCH,
    Pose,
    UnreachableError,
    inverse_kinematics,
    tool_rotation,
    top_down_roll_for_axis,
)
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
    BODY_SIDE_UNKNOWN = "body_side_unknown"
    NO_RIGHTING_STATION = "no_righting_station"


#: Answers "which way along this axis does the held tube's body point?" given
#: the cap's position and a horizontal axis: +1, -1, or ``None`` if unsure.
BodySideProbe = Callable[[np.ndarray, np.ndarray], "float | None"]

#: Roll used for "tool Z pointing down" once horizontal. Exactly pi sits on the
#: joint limit; a hair inside it is the same pose for every practical purpose.
_ROLL_HALF_TURN = math.pi - 1e-4

#: Commanded clearance under a lying tube when it is set back down. Like
#: :data:`_STANDING_RELEASE_M` it allows for the ~6 mm the arm sags when
#: carrying; at 3 mm the tube was pressed into the table and left tilted, its
#: labelled end face showing, and the next look read it as standing.
_RIGHTING_DROP_M = 0.010

#: Commanded height of an upright tube's foot above the table at release. Larger
#: than it looks: carrying a tube at the reach edge the arm sags ~6 mm below its
#: commanded pose, and the gripped cap sits ~2 mm below the TCP, so this leaves
#: ~3 mm of real clearance (measured). At 3 mm commanded, the foot met the table
#: while still held and the tube was pushed over.
_STANDING_RELEASE_M = 0.012

#: How far the gripper backs away, along its own axis, after setting a tube down.
_RIGHTING_RETREAT_M = 0.045

#: Largest vertical step when lowering a held, hanging tube onto the station.
_LOWERING_STEP_M = 0.01

#: Slow-down while carrying a fallen tube. It is gripped by the cap, far from its
#: centre of mass. At full speed the simulated grip lags and jolts the arm
#: (measured: 278 mm TCP error on a lift); at 2.5x the arm is steady but the tube
#: still twists 2-22 degrees in the grip; at 5x the twist is 2-9 degrees. Grip
#: force made no difference, so this is a speed limit, not a force limit.
_CARRY_SLOWDOWN = 5.0


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
        sample_id: The ID read from the cap by the wrist scanner, when one was.
    """

    success: bool
    reason: FailureReason = FailureReason.NONE
    duration_s: float = 0.0
    grasped: bool = False
    place_error_m: float | None = None
    waypoints: list[np.ndarray] = field(default_factory=list)
    sample_id: str | None = None

    @property
    def failed(self) -> bool:
        """Whether the attempt did not succeed."""
        return not self.success


@dataclass
class RightingResult:
    """The outcome of standing one fallen tube back up.

    Attributes:
        success: Whether the manoeuvre completed. Whether the tube is actually
            standing is confirmed by the next overhead frame, not assumed here.
        reason: Which stage failed, or :attr:`FailureReason.NONE`.
        duration_s: Wall-clock seconds the attempt took.
        body_side: ``+1``/``-1`` for the side the body was found on in the tool
            frame, or ``None`` if it could not be decided.
    """

    success: bool
    reason: FailureReason = FailureReason.NONE
    duration_s: float = 0.0
    body_side: float | None = None


class ScriptedController:
    """Executes :class:`~samplesort.planning.task_planner.PickPlaceJob` objects.

    Args:
        config: The validated configuration bundle.
        arm: The arm to drive.
        world: Optional simulation world used to verify grasps and placements.
            In real mode this is ``None`` and verification falls back to the
            gripper's own state.
        scanner: Optional wrist-camera scanner. When present and
            ``classes.qr_enabled`` is set, the cap's sample ID is read from the
            pre-grasp pose, where the camera is centred over the tube.
    """

    #: How close to the slot centre a tube must land to count as correctly placed.
    PLACE_TOLERANCE_M = 0.030

    def __init__(
        self,
        config: SampleSortConfig,
        arm: ArmInterface,
        world: PlacementVerifier | None = None,
        scanner: IdScannerInterface | None = None,
    ) -> None:
        """Bind the controller to an arm (see the class docstring for args)."""
        self.config = config
        self.arm = arm
        self.world = world
        self.scanner = scanner

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
        pick_clearance = self.pick_clearance(job)

        poses = [
            ("pre_grasp", job.pick_pose(grasp_z + pick_clearance)),
            ("grasp", job.pick_pose(grasp_z)),
            ("lift", job.pick_pose(grasp_z + pick_clearance)),
            ("pre_place", job.place_pose(place_z + clearance)),
            ("place", job.place_pose(place_z)),
            ("retreat", job.place_pose(place_z + clearance)),
        ]
        if ws.near_righting_station(*job.pick_xy):
            # The station sits beyond the racks, so a direct swing from it to a
            # rack crosses other racks' slots (measured: it toppled a tube already
            # in Rack C). Route via the pickup zone, which is empty by the time a
            # righted tube is sorted, so the rest of the move is an ordinary one.
            via = Pose(*ws.pickup_zone.center, grasp_z + clearance)
            poses.insert(3, ("via", via))
        return poses

    def pick_clearance(self, job: PickPlaceJob) -> float:
        """How high above the grasp pose to approach and lift a tube.

        Normally ``workspace.approach_clearance``. Near the edge of the reach the
        arm can get down to a cap but not to the point straight above it — the
        righting station is such a place, and a righted tube measured a few
        millimetres further out than it stands was given up on in testing. There
        the approach is lowered in 5 mm steps, but never below the cap height
        plus 20 mm, until it can be reached. If the grasp itself is out of reach,
        nothing is lowered and the job fails as unreachable.
        """
        ws = self.config.workspace
        grasp_z = ws.table_grasp_height
        try:
            inverse_kinematics(self.config.arm, job.pick_pose(grasp_z))
        except UnreachableError:
            return ws.approach_clearance
        floor = ws.tube.cap_height + 0.020
        clearance = ws.approach_clearance
        while clearance >= floor - 1e-9:
            try:
                inverse_kinematics(self.config.arm, job.pick_pose(grasp_z + clearance))
            except UnreachableError:
                clearance -= 0.005
                continue
            if clearance < ws.approach_clearance:
                logger.info(
                    "approaching %s from %.0f mm instead of %.0f mm to stay in reach",
                    job.describe(),
                    clearance * 1000,
                    ws.approach_clearance * 1000,
                )
            return clearance
        return ws.approach_clearance

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
            result.sample_id = self._scan_sample_id()
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
            if "via" in stages:
                self._move(stages["via"])

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

    # ------------------------------------------------------------------ righting

    def lying_grip_offset(self) -> float:
        """How far along the approach axis a lying tube's cap centre sits from the TCP.

        The palm block hangs half its depth below the fingertips, so reaching
        all the way down to a lying cap presses it into the table — measured in
        sim: the cap ended 5 mm below the surface and twisted 87 degrees in the
        grip on lifting. Holding it from just above the cap's top keeps the palm
        clear while staying inside the grasp tolerance.
        """
        grip = self.config.arm.gripper
        return self.config.workspace.tube.cap_radius + grip.palm_depth / 2.0 + 0.001

    def righting_poses(self, fallen: FallenTube) -> dict[str, Pose]:
        """The fixed waypoints of a righting manoeuvre, before the body side is known.

        Raises:
            ValueError: If no righting station is configured.
        """
        ws = self.config.workspace
        if ws.righting_station is None:
            raise ValueError("no righting station is configured")
        tube = ws.tube
        cx, cy = fallen.table_xy
        safe = ws.table_height + tube.total_height + ws.approach_clearance
        grip_roll = top_down_roll_for_axis(cx, cy, fallen.axis_yaw)
        # Inspect it right where it was picked up: that spot is now empty, near
        # the middle of the camera's view, and every standing tube has already
        # been sorted. Turn it to lie across the arm so both ends stand clear of
        # the arm in the overhead camera's view.
        across_roll = top_down_roll_for_axis(cx, cy, math.atan2(cy, cx) + math.pi / 2.0)
        return {
            "pre_grasp": Pose(cx, cy, safe, TOP_DOWN_PITCH, grip_roll),
            # A lying cap's centre sits one cap radius above the table; the TCP
            # stays lying_grip_offset() above that.
            "grasp": Pose(
                cx,
                cy,
                ws.table_height + tube.cap_radius + self.lying_grip_offset(),
                TOP_DOWN_PITCH,
                grip_roll,
            ),
            "lift": Pose(cx, cy, safe, TOP_DOWN_PITCH, grip_roll),
            "inspect": Pose(cx, cy, safe, TOP_DOWN_PITCH, across_roll),
        }

    def right_tube(self, fallen: FallenTube, body_side: BodySideProbe) -> RightingResult:
        """Pick a fallen tube up by its cap and stand it upright at the station.

        The sequence: grip the cap from above with the fingers across it, lift,
        carry it to the station, hold it across the arm and ask ``body_side``
        which end is the body, roll so that end will point down, pitch the tool
        level so the tube swings upright, lower it to just above the table,
        release, and back away along the tool axis.

        Args:
            fallen: The tube to right.
            body_side: Probe that inspects the held tube (see :data:`BodySideProbe`).

        Returns:
            A :class:`RightingResult`.
        """
        started = time.perf_counter()
        result = RightingResult(success=False)
        ws = self.config.workspace
        if ws.righting_station is None:
            result.reason = FailureReason.NO_RIGHTING_STATION
            return result

        try:
            poses = self.righting_poses(fallen)
            solved = {
                name: inverse_kinematics(self.config.arm, pose) for name, pose in poses.items()
            }
        except UnreachableError as exc:
            logger.warning("cannot right %s: %s", fallen.describe(), exc)
            result.reason = FailureReason.UNREACHABLE_PICK
            result.duration_s = time.perf_counter() - started
            return result

        try:
            self.arm.open_gripper()
            self._move(solved["pre_grasp"])
            self._move(solved["grasp"])
            self.arm.close_gripper()
            if not self._holding():
                result.reason = FailureReason.GRASP_MISSED
                self._recover()
                return self._finish(result, started)

            self._move(solved["lift"], slowdown=_CARRY_SLOWDOWN)
            self._move(solved["inspect"], slowdown=_CARRY_SLOWDOWN)
            if not self._holding():
                result.reason = FailureReason.DROPPED_IN_TRANSIT
                self._recover()
                return self._finish(result, started)

            inspect = poses["inspect"]
            frame = tool_rotation(inspect)
            cap_centre = inspect.position + frame[:, 0] * self.lying_grip_offset()
            side = body_side(cap_centre, frame[:, 2])
            result.body_side = side
            if side is None:
                logger.warning("could not tell which end of %s is the body", fallen.describe())
                self._set_down_lying(poses)
                result.reason = FailureReason.BODY_SIDE_UNKNOWN
                return self._finish(result, started)

            self._stand_up(side)
        except (ArmError, UnreachableError) as exc:
            logger.error("righting %s failed: %s", fallen.describe(), exc)
            result.reason = FailureReason.ARM_ERROR
            self._recover()
            return self._finish(result, started)

        result.success = True
        return self._finish(result, started)

    def _stand_up(self, side: float) -> None:
        """Roll, pitch level, lower and release a tube held across the arm at the station.

        Level with roll 0 the tool's Z axis points up, and with roll pi it points
        down. The body must end up below the cap, so the roll is pi when the body
        lies along the tool's +Z and 0 when it lies along -Z.
        """
        ws = self.config.workspace
        tube = ws.tube
        assert ws.righting_station is not None
        sx, sy = ws.righting_station
        safe = ws.table_height + tube.total_height + ws.approach_clearance
        roll = _ROLL_HALF_TURN if side > 0 else 0.0

        # Level, the tool axis points straight out from the arm and the held cap
        # sits lying_grip_offset() beyond the TCP along it; stand the TCP that
        # far short of the station so the tube itself lands on the station.
        radial = np.array([sx, sy]) / max(float(np.hypot(sx, sy)), 1e-9)
        tx, ty = np.array([sx, sy]) - radial * self.lying_grip_offset()
        tx, ty = float(tx), float(ty)

        self._move_to(Pose(sx, sy, safe, TOP_DOWN_PITCH, roll), carrying=True)
        for pitch in np.linspace(TOP_DOWN_PITCH, 0.0, 5)[1:]:
            self._move_to(Pose(tx, ty, safe, float(pitch), roll), carrying=True)

        # Level, the cap is beside the TCP, so the base hangs cap-half-height
        # plus the body length below it.
        place_z = ws.table_height + tube.cap_height / 2.0 + tube.body_height + _STANDING_RELEASE_M
        # Lower in short Cartesian steps. One joint-space move from ``safe`` does
        # not keep the wrist level on the way down, and the hanging tube's foot
        # swung into the table and was set down at 30-60 degrees (measured: 9 of
        # 15 single-tube trials ended with the tube lying down again).
        steps = max(1, math.ceil((safe - place_z) / _LOWERING_STEP_M))
        for z in np.linspace(safe, place_z, steps + 1)[1:]:
            self._move_to(Pose(tx, ty, float(z), 0.0, roll), carrying=True)
        self.arm.open_gripper()
        self._settle()

        bx, by = np.array([tx, ty]) - radial * _RIGHTING_RETREAT_M
        self._move_to(Pose(float(bx), float(by), place_z, 0.0, roll))
        self._move_to(Pose(float(bx), float(by), safe, 0.0, roll))
        self._move_to(Pose(float(bx), float(by), safe, TOP_DOWN_PITCH, 0.0))

    def _set_down_lying(self, poses: dict[str, Pose]) -> None:
        """Put an unresolved tube down at the station as it was held, and back off."""
        inspect = poses["inspect"]
        ws = self.config.workspace
        down = inspect.with_z(
            ws.table_height + ws.tube.cap_radius + self.lying_grip_offset() + _RIGHTING_DROP_M
        )
        self._move_to(down, carrying=True)
        self.arm.open_gripper()
        self._settle()
        self._move_to(inspect)

    def _move_to(self, pose: Pose, *, carrying: bool = False) -> None:
        """Solve and command one Cartesian waypoint, slower when carrying a fallen tube."""
        slowdown = _CARRY_SLOWDOWN if carrying else 1.0
        self._move(inverse_kinematics(self.config.arm, pose), slowdown=slowdown)

    @staticmethod
    def _finish(result: RightingResult, started: float) -> RightingResult:
        result.duration_s = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------- helpers

    def _scan_sample_id(self) -> str | None:
        """Read the cap's sample ID from the pre-grasp pose, when scanning is on."""
        if self.scanner is None or not self.config.classes.qr_enabled:
            return None
        if not self.scanner.is_connected:
            return None
        return self.scanner.scan()

    def _move(self, q: np.ndarray, *, slowdown: float = 1.0) -> None:
        """Command one joint waypoint, timed from the configured joint velocity."""
        from samplesort.control.trajectory import duration_for

        current = self.arm.get_joint_positions()
        self.arm.move_to_joints(q, duration_for(self.config.arm, current, q) * slowdown)

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
        pick = job.pick_pose(ws.table_grasp_height + self.pick_clearance(job))
        try:
            inverse_kinematics(self.config.arm, pick)
        except UnreachableError:
            return FailureReason.UNREACHABLE_PICK
        return FailureReason.UNREACHABLE_PLACE
