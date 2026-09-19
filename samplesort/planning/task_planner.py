"""Turn detections into ordered pick-and-place jobs.

:class:`PickPlaceJob` is the contract between perception/planning and the control
layer. :class:`TaskPlanner` converts a frame's worth of detections into a list of
them, allocating rack slots as it goes and ordering the work so the arm never
travels further than it has to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from samplesort.config import SampleSortConfig
from samplesort.perception.types import Detection
from samplesort.planning.kinematics import Pose
from samplesort.planning.rack_state import RackFullError, RackState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PickPlaceJob:
    """One tube's journey from the pickup zone into a rack slot.

    Attributes:
        class_label: The sample class the perception layer assigned.
        pick_xy: Table-frame XY of the tube to collect, in metres.
        rack_id: Destination rack.
        slot_index: Destination slot within that rack.
        place_xy: Table-frame XY of the destination slot, in metres.
        sample_id: Optional identifier decoded from a QR code.
        body_id: Simulation body id of the target tube, when known.
        confidence: Detector confidence for the originating detection.
    """

    class_label: str
    pick_xy: tuple[float, float]
    rack_id: str
    slot_index: int
    place_xy: tuple[float, float]
    sample_id: str | None = None
    body_id: int | None = None
    confidence: float = 1.0

    def pick_pose(self, z: float) -> Pose:
        """Tool pose above the pickup point at height ``z``."""
        return Pose(self.pick_xy[0], self.pick_xy[1], z)

    def place_pose(self, z: float) -> Pose:
        """Tool pose above the destination slot at height ``z``."""
        return Pose(self.place_xy[0], self.place_xy[1], z)

    def describe(self) -> str:
        """A one-line human-readable summary of the job."""
        return (
            f"{self.class_label} tube at ({self.pick_xy[0]:+.3f}, {self.pick_xy[1]:+.3f}) "
            f"-> {self.rack_id} slot {self.slot_index}"
        )


@dataclass(frozen=True)
class SkippedDetection:
    """A detection that could not be turned into a job.

    Attributes:
        detection: The detection that was skipped.
        reason: Why it was skipped, e.g. ``"rack_full"`` or ``"low_confidence"``.
    """

    detection: Detection
    reason: str


@dataclass(frozen=True)
class PlanResult:
    """The outcome of planning one frame.

    Attributes:
        jobs: The pick-and-place jobs to execute, in order.
        skipped: Detections that were deliberately not turned into jobs.
    """

    jobs: list[PickPlaceJob]
    skipped: list[SkippedDetection]

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing left to do."""
        return not self.jobs


class TaskPlanner:
    """Converts detections into ordered pick-and-place jobs.

    Jobs are ordered nearest-first from the arm's base: shorter reaches are both
    faster and more accurate, and clearing the near tubes first reduces the chance
    of knocking one over while reaching past it.

    Args:
        config: The validated configuration bundle.
        rack_state: Occupancy tracker that slots are allocated from.
    """

    def __init__(self, config: SampleSortConfig, rack_state: RackState) -> None:
        """Bind the planner to a rack state (see the class docstring for args)."""
        self.config = config
        self.rack_state = rack_state

    def plan(self, detections: list[Detection]) -> PlanResult:
        """Turn one frame's detections into jobs.

        Detections below the configured confidence threshold, outside the pickup
        zone, or destined for a full rack are skipped with a stated reason rather
        than silently dropped.

        Args:
            detections: What the detector found this frame.

        Returns:
            The ordered jobs plus every skipped detection and why.
        """
        zone = self.config.workspace.pickup_zone
        threshold = self.config.classes.detector.min_confidence

        candidates: list[Detection] = []
        skipped: list[SkippedDetection] = []
        for detection in detections:
            x, y = detection.table_xy
            if detection.confidence < threshold:
                skipped.append(SkippedDetection(detection, "low_confidence"))
            elif not zone.contains(x, y):
                skipped.append(SkippedDetection(detection, "outside_pickup_zone"))
            else:
                candidates.append(detection)

        # Nearest to the arm base first.
        candidates.sort(key=lambda d: d.distance_to(0.0, 0.0))

        jobs: list[PickPlaceJob] = []
        for detection in candidates:
            try:
                assignment = self.rack_state.occupy(detection.class_label)
            except RackFullError as exc:
                logger.warning("skipping %s: %s", detection.describe(), exc)
                skipped.append(SkippedDetection(detection, "rack_full"))
                continue
            jobs.append(
                PickPlaceJob(
                    class_label=detection.class_label,
                    pick_xy=detection.table_xy,
                    rack_id=assignment.rack_id,
                    slot_index=assignment.slot_index,
                    place_xy=assignment.xy,
                    sample_id=detection.sample_id,
                    confidence=detection.confidence,
                )
            )

        logger.debug("planned %d jobs, skipped %d", len(jobs), len(skipped))
        return PlanResult(jobs=jobs, skipped=skipped)
