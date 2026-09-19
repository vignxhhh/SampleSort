"""Cap-colour classification and the optional QR sample-ID reader.

The baseline classifier thresholds HSV ranges read from ``classes.yaml``. It is
deliberately simple: every threshold is config, nothing is learned, and the
behaviour is fully explainable — which matters for a chain-of-custody record.
Swapping it for a trained detector is a listed stretch goal.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from samplesort.config import ClassConfig, ClassesConfig

logger = logging.getLogger(__name__)


class ColorClassifier:
    """Assigns class labels to cap colours using configured HSV ranges.

    Args:
        config: The sample taxonomy and detector thresholds.
    """

    #: OpenCV packs the full hue circle into 0-179.
    HUE_PERIOD = 180

    def __init__(self, config: ClassesConfig) -> None:
        """Bind the classifier to a taxonomy (see the class docstring for args)."""
        self.config = config
        self._circular_ranges = {
            cls.label: self._merge_wraparound(cls.hue_ranges) for cls in config.classes
        }

    @classmethod
    def _merge_wraparound(cls, ranges: list[tuple[int, int]]) -> list[tuple[float, float]]:
        """Rejoin hue ranges that were split across the 0/179 boundary.

        ``classes.yaml`` has to express red as ``[[0, 8], [172, 179]]`` because a
        range cannot be written inverted. Scoring centrality on those halves would
        put pure red (hue 0) right at an edge. Merging them into a single
        ``[172, 188]`` interval, evaluated modulo 180, puts it at the centre where
        it belongs.

        Args:
            ranges: The configured hue intervals.

        Returns:
            Intervals whose upper bound may exceed 179 to express a wrap.
        """
        ordered = sorted((float(low), float(high)) for low, high in ranges)
        if len(ordered) < 2:
            return ordered
        first_low, first_high = ordered[0]
        last_low, last_high = ordered[-1]
        if first_low <= 0.0 and last_high >= cls.HUE_PERIOD - 1:
            merged = (last_low, first_high + cls.HUE_PERIOD)
            return [*ordered[1:-1], merged]
        return ordered

    @property
    def labels(self) -> list[str]:
        """Every class label the classifier can emit."""
        return self.config.labels

    # ------------------------------------------------------------------- masks

    def class_mask(self, hsv: np.ndarray, cls: ClassConfig) -> np.ndarray:
        """Build a binary mask of pixels matching one class.

        Args:
            hsv: An HSV image of shape ``(h, w, 3)``.
            cls: The class whose ranges to threshold against.

        Returns:
            A ``uint8`` mask where matching pixels are 255.
        """
        sat_low, sat_high = cls.saturation_range
        val_low, val_high = cls.value_range
        mask: np.ndarray = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hue_low, hue_high in cls.hue_ranges:
            lower = np.array([hue_low, sat_low, val_low], dtype=np.uint8)
            upper = np.array([hue_high, sat_high, val_high], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    def masks(self, hsv: np.ndarray) -> dict[str, np.ndarray]:
        """Build one mask per class.

        Args:
            hsv: An HSV image of shape ``(h, w, 3)``.

        Returns:
            A mapping from class label to its binary mask.
        """
        return {cls.label: self.class_mask(hsv, cls) for cls in self.config.classes}

    # -------------------------------------------------------------- classifying

    @staticmethod
    def _centrality(value: float, low: float, high: float) -> float:
        """How centrally ``value`` sits in ``[low, high]``: 1 at the centre, 0 at an edge.

        Used for hue, where drifting either way means the colour is turning into a
        neighbouring class.
        """
        if high <= low:
            return 1.0
        centre = (low + high) / 2.0
        half_width = (high - low) / 2.0
        return max(0.0, 1.0 - abs(value - centre) / half_width)

    @staticmethod
    def _headroom(value: float, low: float, high: float) -> float:
        """How far ``value`` clears the lower bound of ``[low, high]``, in ``[0, 1]``.

        Used for saturation and value, where the lower bound is the real
        threshold and the upper bound is a don't-care ceiling. A fully saturated,
        fully lit cap is the *best* case, so it must not be penalised for sitting
        at the top of its range the way :meth:`_centrality` would.
        """
        span = high - low
        if span <= 0.0:
            return 1.0
        return float(min(1.0, max(0.0, (value - low) / span)))

    def _score(self, cls: ClassConfig, hue: float, sat: float, val: float) -> float | None:
        """Score one class against an HSV triple, or ``None`` if it does not match."""
        matching: list[tuple[float, float, float]] = []
        for low, high in self._circular_ranges[cls.label]:
            for candidate in (hue, hue + self.HUE_PERIOD):
                if low <= candidate <= high:
                    matching.append((low, high, candidate))
        if not matching:
            return None
        sat_low, sat_high = cls.saturation_range
        val_low, val_high = cls.value_range
        if not (sat_low <= sat <= sat_high and val_low <= val <= val_high):
            return None

        hue_low, hue_high, wrapped_hue = max(
            matching, key=lambda r: self._centrality(r[2], r[0], r[1])
        )
        margins = (
            self._centrality(wrapped_hue, hue_low, hue_high),
            self._headroom(sat, sat_low, sat_high),
            self._headroom(val, val_low, val_high),
        )
        # Anything that matches at all scores at least 0.5; centrality lifts it.
        return 0.5 + 0.5 * float(min(margins))

    def classify_hsv(self, hue: float, sat: float, val: float) -> tuple[str, float] | None:
        """Classify a single HSV triple.

        Args:
            hue: Hue in OpenCV's 0-179 range.
            sat: Saturation in 0-255.
            val: Value in 0-255.

        Returns:
            The best ``(label, confidence)`` match, or ``None`` if the colour does
            not fall inside any configured class.
        """
        best: tuple[str, float] | None = None
        for cls in self.config.classes:
            score = self._score(cls, hue, sat, val)
            if score is not None and (best is None or score > best[1]):
                best = (cls.label, score)
        return best

    def classify_bgr(self, blue: int, green: int, red: int) -> tuple[str, float] | None:
        """Classify a single BGR colour by converting it to HSV first."""
        pixel = np.array([[[blue, green, red]]], dtype=np.uint8)
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        return self.classify_hsv(float(hsv[0]), float(hsv[1]), float(hsv[2]))

    def classify_region(self, hsv: np.ndarray, mask: np.ndarray) -> tuple[str, float] | None:
        """Classify a masked image region by its dominant colour.

        The median hue, saturation and value of the masked pixels are used rather
        than the mean, so a few stray edge pixels cannot drag the result across a
        class boundary.

        Args:
            hsv: An HSV image of shape ``(h, w, 3)``.
            mask: A boolean or ``uint8`` mask selecting the region.

        Returns:
            The best ``(label, confidence)`` match, or ``None``.
        """
        selected = hsv[mask.astype(bool)]
        if selected.size == 0:
            return None
        median = np.median(selected.reshape(-1, 3), axis=0)
        return self.classify_hsv(float(median[0]), float(median[1]), float(median[2]))


class QRReader:
    """Reads printed sample IDs from QR codes (spec section 10 stretch goal).

    Args:
        enabled: When ``False``, :meth:`read` always returns an empty list, so the
            pipeline can carry the reader unconditionally.
    """

    def __init__(self, enabled: bool = False) -> None:
        """Create a reader (see the class docstring for args)."""
        self.enabled = enabled
        self._detector = cv2.QRCodeDetector() if enabled else None

    def read(self, frame: np.ndarray) -> list[tuple[str, tuple[float, float]]]:
        """Decode every QR code visible in a frame.

        Args:
            frame: A BGR image.

        Returns:
            One ``(decoded_text, (u, v))`` pair per readable code, where ``(u, v)``
            is the centre of its bounding quad in pixels.
        """
        if not self.enabled or self._detector is None:
            return []
        try:
            ok, decoded, points, _ = self._detector.detectAndDecodeMulti(frame)
        except cv2.error as exc:  # pragma: no cover - malformed frames only
            logger.warning("QR decoding failed: %s", exc)
            return []
        if not ok or points is None:
            return []

        results: list[tuple[str, tuple[float, float]]] = []
        for text, quad in zip(decoded, points, strict=True):
            if not text:
                continue
            centre = np.mean(np.asarray(quad, dtype=float).reshape(-1, 2), axis=0)
            results.append((str(text), (float(centre[0]), float(centre[1]))))
        return results
