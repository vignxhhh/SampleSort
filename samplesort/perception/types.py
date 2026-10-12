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


@dataclass(frozen=True)
class FallenTube:
    """A tube lying on its side, seen from overhead.

    From above, a knocked-over tube's cap shows its side: a rectangle as long as
    the cap is wide and as wide as the cap is tall. Its short axis is the tube's
    axis. Which end the body is on is *not* decided here; the righting
    manoeuvre resolves that later, against a clean background.

    Attributes:
        pixel_xy: Centroid of the cap blob in image coordinates, ``(u, v)``.
        table_xy: The cap centre on the table plane in metres, corrected for a
            lying cap's height (its radius) rather than an upright cap's.
        class_label: The sample class from the cap colour.
        axis_yaw: Direction of the tube's axis in the table plane, in radians.
            A line, so it is only defined modulo pi.
        confidence: How cleanly the blob matched its class, in ``[0, 1]``.
        area_px: Blob area in pixels.
    """

    pixel_xy: tuple[float, float]
    table_xy: tuple[float, float]
    class_label: str
    axis_yaw: float
    confidence: float
    area_px: float = 0.0

    def describe(self) -> str:
        """A one-line human-readable summary."""
        return (
            f"fallen {self.class_label} @ ({self.table_xy[0]:+.3f}, {self.table_xy[1]:+.3f}) m "
            f"axis={np.degrees(self.axis_yaw):.0f} deg"
        )


@dataclass(frozen=True)
class Scene:
    """Everything one overhead frame shows: standing tubes and fallen ones."""

    upright: list[Detection]
    fallen: list[FallenTube]

    @property
    def is_clear(self) -> bool:
        """Whether there is nothing left to handle."""
        return not self.upright and not self.fallen
