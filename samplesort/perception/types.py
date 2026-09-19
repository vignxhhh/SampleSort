"""Data types shared across the perception modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected sample tube.

    Attributes:
        pixel_xy: Centroid of the cap blob in image coordinates, ``(u, v)``.
        table_xy: The same point mapped onto the table plane in the arm base
            frame, in metres.
        class_label: The sample class assigned by the classifier.
        confidence: How cleanly the blob matched its class, in ``[0, 1]``.
        area_px: Blob area in pixels, useful for rejecting noise and merges.
        sample_id: Identifier decoded from a QR code, when one was read.
    """

    pixel_xy: tuple[float, float]
    table_xy: tuple[float, float]
    class_label: str
    confidence: float
    area_px: float = 0.0
    sample_id: str | None = None

    @property
    def position(self) -> np.ndarray:
        """The table-frame XY as a 2-element array."""
        return np.array(self.table_xy, dtype=float)

    def distance_to(self, x: float, y: float) -> float:
        """Distance in metres from this detection to a table-frame point."""
        return float(np.hypot(self.table_xy[0] - x, self.table_xy[1] - y))

    def describe(self) -> str:
        """A one-line human-readable summary."""
        return (
            f"{self.class_label} @ ({self.table_xy[0]:+.3f}, {self.table_xy[1]:+.3f}) m "
            f"conf={self.confidence:.2f}"
        )
