"""The main sort loop: perceive, plan, execute, log, repeat.

One iteration retracts the arm to its observe pose, captures a frame, detects and
classifies every tube in the pickup zone, plans jobs against the current rack
state, and executes them. Failed grasps are retried once and then skipped; every
attempt is written to the sort log either way.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field, replace

import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.control.scripted import ExecutionResult, FailureReason, ScriptedController
from samplesort.hal.factory import Backend
from samplesort.hal.sim_camera import SimCamera
from samplesort.logging_db.sort_log import SortLog, SortRecord
from samplesort.perception.calibration import (
    DEFAULT_CALIBRATION_NAME,
    Calibration,
    CalibrationError,
    calibration_from_camera_pose,
)
from samplesort.perception.detector import TubeDetector
from samplesort.perception.types import Detection
from samplesort.planning.rack_state import RackState
from samplesort.planning.task_planner import PickPlaceJob, PlanResult, TaskPlanner

logger = logging.getLogger(__name__)


@dataclass
class JobOutcome:
    """What happened to one planned job.

    Attributes:
        job: The job that was attempted.
        result: The controller's result for the final attempt.
        attempts: How many times the job was tried, including retries.
    """

    job: PickPlaceJob
    result: ExecutionResult
    attempts: int = 1

    @property
    def success(self) -> bool:
        """Whether the job ultimately succeeded."""
        return self.result.success


@dataclass
class PipelineReport:
    """Summary of one complete pipeline run.

    Attributes:
        run_id: Identifier shared by every log row this run produced.
        mode: Which controller was used.
        iterations: How many perceive-plan-execute cycles ran.
        outcomes: One entry per attempted job.
        detections_seen: Total detections across every iteration.
        skipped: ``(class_label, reason)`` for each detection that was not planned.
        duration_s: Wall-clock seconds for the whole run.
        dry_run: Whether motion was suppressed.
    """

    run_id: str
    mode: str
    iterations: int = 0
    outcomes: list[JobOutcome] = field(default_factory=list)
    detections_seen: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    duration_s: float = 0.0
    dry_run: bool = False

    @property
    def attempted(self) -> int:
        """How many jobs were attempted."""
        return len(self.outcomes)

    @property
    def succeeded(self) -> int:
        """How many jobs succeeded."""
        return sum(1 for outcome in self.outcomes if outcome.success)

    @property
    def success_rate(self) -> float:
        """Fraction of attempted jobs that succeeded, or 0.0 if none were attempted."""
        return self.succeeded / self.attempted if self.attempted else 0.0

    @property
    def grasp_success_rate(self) -> float:
        """Fraction of attempts whose grasp stage worked, across all retries."""
        total = sum(outcome.attempts for outcome in self.outcomes)
        if not total:
            return 0.0
        grasped = sum(1 for outcome in self.outcomes if outcome.result.grasped)
        return grasped / total

    @property
    def mean_duration_s(self) -> float:
        """Mean seconds per successful job, or 0.0 when none succeeded."""
        successes = [o.result.duration_s for o in self.outcomes if o.success]
        return sum(successes) / len(successes) if successes else 0.0

    def failures_by_reason(self) -> dict[str, int]:
        """How many jobs failed, broken down by failure reason."""
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            if not outcome.success:
                key = outcome.result.reason.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def summary_lines(self) -> list[str]:
        """Human-readable summary lines for CLI output."""
        lines = [
            f"run {self.run_id} ({self.mode}{', dry-run' if self.dry_run else ''})",
            f"  iterations        : {self.iterations}",
            f"  detections seen   : {self.detections_seen}",
            f"  jobs attempted    : {self.attempted}",
            f"  jobs succeeded    : {self.succeeded} ({self.success_rate:.0%})",
            f"  mean time per job : {self.mean_duration_s:.2f} s",
            f"  total duration    : {self.duration_s:.2f} s",
        ]
        failures = self.failures_by_reason()
        if failures:
            breakdown = ", ".join(f"{name}={count}" for name, count in failures.items())
            lines.append(f"  failures          : {breakdown}")
        if self.skipped:
            lines.append(f"  skipped detections: {len(self.skipped)}")
        return lines


class SortPipeline:
    """Runs the perceive-plan-execute-log loop until the pickup zone is clear.

    Args:
        config: The validated configuration bundle.
        backend: A connected arm and camera.
        sort_log: Where attempts are recorded. One is opened from
            ``config.database_path`` when omitted.
        controller: The controller to execute jobs with. Defaults to the scripted
            controller; the learned controller is injected here in phase 7.
        calibration: Pixel-to-table mapping. Derived or loaded when omitted.
    """

    def __init__(
        self,
        config: SampleSortConfig,
        backend: Backend,
        *,
        sort_log: SortLog | None = None,
        controller: ScriptedController | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        """Wire the pipeline together (see the class docstring for args)."""
        self.config = config
        self.backend = backend
        self._owns_log = sort_log is None
        self.sort_log = sort_log or SortLog(config.database_path)
        self.calibration = calibration or self._resolve_calibration()
        self.detector = TubeDetector(config, self.calibration)
        self.rack_state = RackState(config.workspace, config.classes)
        self.planner = TaskPlanner(config, self.rack_state)
        self.controller = controller or ScriptedController(config, backend.arm, backend.world)
        self._noise_rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------- calibration

    def _resolve_calibration(self) -> Calibration:
        """Load a saved calibration, or derive one from the sim camera pose.

        Raises:
            CalibrationError: In real mode when no saved calibration exists.
        """
        saved = self.config.config_dir / DEFAULT_CALIBRATION_NAME
        if saved.is_file():
            logger.info("using saved calibration from %s", saved)
            return Calibration.load(saved)

        camera = self.backend.camera
        if isinstance(camera, SimCamera):
            if not camera.is_connected:
                camera.connect()
            logger.debug("deriving calibration analytically from the sim camera pose")
            return calibration_from_camera_pose(
                self.config.camera, self.config.workspace, camera.world_to_pixel
            )

        raise CalibrationError(
            f"no calibration found at {saved}. Run 'samplesort calibrate' with the "
            f"printed ArUco board in view before sorting on real hardware."
        )

    # ------------------------------------------------------------------ helpers

    def perceive(self) -> list[Detection]:
        """Retract the arm clear of the camera and detect tubes in the pickup zone.

        Returns:
            Detections inside the pickup zone, highest confidence first.
        """
        self.backend.arm.move_to_joints(self.config.arm.observe_position)
        if self.backend.world is not None:
            self.backend.world.step(self.config.pipeline.settle_steps)
        detections = self.detector.detect_in_pickup_zone(self.backend.camera.read())
        return [self._jitter(d) for d in detections]

    def _jitter(self, detection: Detection) -> Detection:
        """Add the configured localisation noise to a detection.

        A no-op unless ``pipeline.perception_noise_m`` is set. See
        :class:`~samplesort.config.PipelineConfig` for why the knob exists.
        """
        sigma = self.config.pipeline.perception_noise_m
        if sigma <= 0.0:
            return detection
        offset = self._noise_rng.normal(0.0, sigma, size=2)
        return replace(
            detection,
            table_xy=(detection.table_xy[0] + offset[0], detection.table_xy[1] + offset[1]),
        )

    def plan(self, detections: list[Detection]) -> PlanResult:
        """Turn detections into ordered jobs against the current rack state.

        In simulation each job is additionally linked to the PyBullet body it
        refers to. That link is *instrumentation*, not perception: it is what lets
        the controller verify a grasp and a placement against ground truth, and
        what lets the benchmark measure whether a tube reached the rack its true
        class belongs in. Nothing in perception or planning consults it, and on
        real hardware it is simply absent.
        """
        plan = self.planner.plan(detections)
        if self.backend.world is None:
            return plan
        return PlanResult(
            jobs=[self._attach_ground_truth(job) for job in plan.jobs], skipped=plan.skipped
        )

    def _attach_ground_truth(self, job: PickPlaceJob) -> PickPlaceJob:
        """Link a job to the simulated tube nearest its pick position."""
        world = self.backend.world
        if world is None:
            return job

        nearest, best = None, float("inf")
        for tube in world.remaining_tubes():
            x, y = world.tube_xy(tube.body_id)
            distance = ((x - job.pick_xy[0]) ** 2 + (y - job.pick_xy[1]) ** 2) ** 0.5
            if distance < best:
                nearest, best = tube, distance

        if nearest is None or best > self.config.arm.gripper.grasp_tolerance_xy:
            return job
        return replace(job, body_id=nearest.body_id)

    def true_label_for(self, job: PickPlaceJob) -> str | None:
        """Ground-truth class of a job's tube, when running in simulation.

        Returns:
            The simulated tube's real class label, or ``None`` outside simulation
            or when the job was never linked to a body.
        """
        world = self.backend.world
        if world is None or job.body_id is None:
            return None
        try:
            return world.tube_by_id(job.body_id).label
        except KeyError:
            return None

    def _record(self, job: PickPlaceJob, result: ExecutionResult, run_id: str) -> None:
        """Write one attempt to the sort log."""
        self.sort_log.insert(
            SortRecord(
                class_label=job.class_label,
                pickup_xy=job.pick_xy,
                rack_id=job.rack_id,
                slot_index=job.slot_index,
                mode=self.config.pipeline.control_mode,
                success=result.success,
                duration_s=result.duration_s,
                sample_id=job.sample_id,
                failure_reason=None if result.success else result.reason.value,
                run_id=run_id,
            )
        )

    def _execute_with_retry(self, job: PickPlaceJob) -> JobOutcome:
        """Run a job, retrying a missed grasp up to the configured retry count."""
        retries = self.config.pipeline.grasp_retries
        result = self.controller.execute(job)
        attempts = 1

        while (
            not result.success
            and attempts <= retries
            and result.reason in (FailureReason.GRASP_MISSED, FailureReason.DROPPED_IN_TRANSIT)
        ):
            logger.info("retrying %s after %s", job.describe(), result.reason.value)
            result = self.controller.execute(job)
            attempts += 1

        return JobOutcome(job=job, result=result, attempts=attempts)

    # --------------------------------------------------------------------- run

    def run(self, *, dry_run: bool = False, run_id: str | None = None) -> PipelineReport:
        """Sort until the pickup zone is clear or the iteration cap is hit.

        Args:
            dry_run: Perceive and plan only; no motion and no log rows.
            run_id: Identifier to tag log rows with. Generated when omitted.

        Returns:
            A report covering every iteration, job and failure.
        """
        started = time.perf_counter()
        identifier = run_id or uuid.uuid4().hex[:12]
        report = PipelineReport(
            run_id=identifier, mode=self.config.pipeline.control_mode, dry_run=dry_run
        )

        for iteration in range(self.config.pipeline.max_iterations):
            detections = self.perceive()
            report.iterations = iteration + 1
            report.detections_seen += len(detections)

            if not detections:
                logger.info("pickup zone is clear after %d iteration(s)", report.iterations)
                break

            plan = self.plan(detections)
            report.skipped.extend(
                (skip.detection.class_label, skip.reason) for skip in plan.skipped
            )

            if plan.is_empty:
                logger.warning(
                    "%d detection(s) remain but none can be planned; stopping", len(detections)
                )
                break

            if dry_run:
                logger.info("dry run: planned %d job(s), not executing", len(plan.jobs))
                for job in plan.jobs:
                    logger.info("  would execute %s", job.describe())
                break

            for job in plan.jobs:
                outcome = self._execute_with_retry(job)
                report.outcomes.append(outcome)
                self._record(job, outcome.result, identifier)
                if not outcome.success:
                    # The slot was reserved optimistically; give it back.
                    self.rack_state.release(job.rack_id, job.slot_index)
                    self._mark_unsortable(job)
                elif self.backend.world is not None and job.body_id is not None:
                    tube = self.backend.world.tube_by_id(job.body_id)
                    tube.placed_rack, tube.placed_slot = job.rack_id, job.slot_index
        else:
            logger.warning(
                "hit the %d-iteration cap with tubes still in the pickup zone",
                self.config.pipeline.max_iterations,
            )

        if not dry_run:
            self.backend.arm.go_home()
        report.duration_s = time.perf_counter() - started
        return report

    def _mark_unsortable(self, job: PickPlaceJob) -> None:
        """Retire a tube the pipeline could not sort so the loop makes progress.

        Without this a tube the arm cannot grasp would be re-detected forever. In
        sim the tube is removed from the scene; on real hardware an operator is
        asked to clear it.
        """
        world = self.backend.world
        if world is None:
            logger.error(
                "could not sort the %s tube at (%.3f, %.3f); please remove it by hand",
                job.class_label,
                job.pick_xy[0],
                job.pick_xy[1],
            )
            return

        nearest = None
        if job.body_id is not None:
            try:
                nearest = world.tube_by_id(job.body_id)
            except KeyError:
                nearest = None
        if nearest is None:
            best = float("inf")
            for tube in world.remaining_tubes():
                x, y = world.tube_xy(tube.body_id)
                distance = ((x - job.pick_xy[0]) ** 2 + (y - job.pick_xy[1]) ** 2) ** 0.5
                if distance < best:
                    nearest, best = tube, distance
            if nearest is not None and best >= self.config.workspace.tube.min_separation:
                nearest = None

        if nearest is not None:
            logger.info("retiring unsortable %s tube %d", nearest.label, nearest.body_id)
            # Drop any grasp constraint first: removing a body that a constraint
            # still references leaves the constraint dangling, and PyBullet then
            # complains on the next release.
            release = getattr(self.backend.arm, "release", None)
            if callable(release):
                release()
            world.bullet.removeBody(nearest.body_id)
            world.tubes.remove(nearest)

    def close(self) -> None:
        """Close the sort log if this pipeline opened it."""
        if self._owns_log:
            self.sort_log.close()

    def __enter__(self) -> SortPipeline:
        """Return the pipeline on entry to a ``with`` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the pipeline on exit from a ``with`` block."""
        self.close()
