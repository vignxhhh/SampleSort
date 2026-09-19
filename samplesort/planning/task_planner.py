"""Turn detections into ordered pick-and-place jobs.

Phase 3 defines the :class:`PickPlaceJob` contract that the control layer
executes. The :class:`TaskPlanner` that produces them lands in phase 5, once rack
state and perception exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from samplesort.planning.kinematics import Pose

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
