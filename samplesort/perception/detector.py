"""Find sample tubes in an overhead frame and locate them on the table.

The baseline pipeline is HSV masking followed by contour analysis: one mask per
configured class, morphological cleanup, contour extraction, then geometric and
area filtering. Surviving blobs are mapped through the calibration homography
into table-frame metres and returned as
:class:`~samplesort.perception.types.Detection` objects.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.perception.calibration import Calibration
from samplesort.perception.classifier import ColorClassifier, QRReader
from samplesort.perception.types import Detection

logger = logging.getLogger(__name__)

#: How close a decoded QR code must be to a detection to be attached to it.
QR_ASSOCIATION_RADIUS_PX = 60.0


class TubeDetector:
    """Detects coloured tube caps and maps them onto the table plane.

    Args:
        config: The validated configuration bundle.
        calibration: The pixel-to-table mapping to use. Detections carry a
            ``table_xy`` of ``(nan, nan)`` if this is ``None``.
    """

    def __init__(self, config: SampleSortConfig, calibration: Calibration | None = None) -> None:
        """Build a detector (see the class docstring for args)."""
        self.config = config
        self.calibration = calibration
        self.classifier = ColorClassifier(config.classes)
        self.qr_reader = QRReader(enabled=config.classes.qr_enabled)
        self.tuning = config.classes.detector

    # --------------------------------------------------------------- image prep

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Blur a BGR frame and convert it to HSV.

        Args:
            frame: A BGR ``uint8`` image.

        Returns:
            The blurred HSV image.

        Raises:
            ValueError: If the frame is not a 3-channel image.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected a 3-channel BGR frame, got shape {frame.shape}")
        kernel = self.tuning.blur_kernel
        blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0) if kernel > 1 else frame
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        """Open then close a mask to drop speckle and fill small holes."""
        size = self.tuning.morph_kernel
        if size <= 1:
            return mask
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, element)

    # ---------------------------------------------------------------- detection

    @staticmethod
    def _circularity(contour: np.ndarray) -> float:
        """Isoperimetric ratio of a contour: 1.0 for a perfect circle, less otherwise."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            return 0.0
        return float(min(1.0, 4.0 * np.pi * area / (perimeter**2)))

    def _to_table(self, u: float, v: float) -> tuple[float, float]:
        """Map a pixel onto the table plane, correcting for the cap's height."""
        if self.calibration is None:
            return (float("nan"), float("nan"))
        cap_top = self.config.workspace.tube.total_height
        return self.calibration.pixel_to_table(u, v, object_height=cap_top)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Find every tube cap in a frame.

        Args:
            frame: A BGR ``uint8`` overhead image.

        Returns:
            Detections sorted by descending confidence. Blobs below the
            configured area, circularity or confidence thresholds are dropped.
        """
        hsv = self.preprocess(frame)
        detections: list[Detection] = []

        for cls in self.config.classes.classes:
            mask = self._clean(self.classifier.class_mask(hsv, cls))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if not self.tuning.min_area_px <= area <= self.tuning.max_area_px:
                    continue

                moments = cv2.moments(contour)
                if moments["m00"] <= 0.0:
                    continue
                u = float(moments["m10"] / moments["m00"])
                v = float(moments["m01"] / moments["m00"])

                blob: np.ndarray = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(blob, [contour], -1, (255,), thickness=cv2.FILLED)
                verdict = self.classifier.classify_region(hsv, blob)
                if verdict is None:
                    continue
                label, colour_confidence = verdict
                if label != cls.label:
                    # The blob's dominant colour disagrees with the mask that found
                    # it; let the other class's own pass claim it instead.
                    continue

                # Caps are circular from overhead, so shape is real evidence.
                confidence = float(colour_confidence * self._circularity(contour))
                if confidence < self.tuning.min_confidence:
                    continue

                detections.append(
                    Detection(
                        pixel_xy=(u, v),
                        table_xy=self._to_table(u, v),
                        class_label=label,
                        confidence=confidence,
                        area_px=area,
                    )
                )

        detections = self._attach_sample_ids(frame, detections)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("detected %d tubes", len(detections))
        return detections

    def _attach_sample_ids(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Pair decoded QR codes with their nearest detection, when enabled."""
        codes = self.qr_reader.read(frame)
        if not codes:
            return detections

        enriched = list(detections)
        for text, (u, v) in codes:
            best_index, best_distance = -1, QR_ASSOCIATION_RADIUS_PX
            for index, detection in enumerate(enriched):
                if detection.sample_id is not None:
                    continue
                distance = float(np.hypot(detection.pixel_xy[0] - u, detection.pixel_xy[1] - v))
                if distance < best_distance:
                    best_index, best_distance = index, distance
            if best_index >= 0:
                current = enriched[best_index]
                enriched[best_index] = Detection(
                    pixel_xy=current.pixel_xy,
                    table_xy=current.table_xy,
                    class_label=current.class_label,
                    confidence=current.confidence,
                    area_px=current.area_px,
                    sample_id=text,
                )
        return enriched

    def detect_in_pickup_zone(self, frame: np.ndarray) -> list[Detection]:
        """Detect tubes and keep only those inside the configured pickup zone.

        Args:
            frame: A BGR ``uint8`` overhead image.

        Returns:
            The subset of :meth:`detect` results that lie inside the pickup zone.
            Detections with no valid table position are dropped.
        """
        zone = self.config.workspace.pickup_zone
        inside: list[Detection] = []
        for detection in self.detect(frame):
            x, y = detection.table_xy
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if zone.contains(x, y):
                inside.append(detection)
        return inside

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw detection markers on a copy of the frame, for debugging and docs.

        Args:
            frame: A BGR ``uint8`` image.
            detections: The detections to draw.

        Returns:
            A new annotated BGR image; the input is left untouched.
        """
        canvas: np.ndarray = frame.copy()
        for detection in detections:
            u, v = (int(round(detection.pixel_xy[0])), int(round(detection.pixel_xy[1])))
            radius = max(6, int(round(np.sqrt(max(detection.area_px, 1.0) / np.pi))))
            cv2.circle(canvas, (u, v), radius, (0, 0, 0), 2)
            cv2.putText(
                canvas,
                f"{detection.class_label} {detection.confidence:.2f}",
                (u + radius + 4, v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas
