"""Find sample tubes in an overhead frame and locate them on the table.

The baseline pipeline is HSV masking followed by contour analysis: one mask per
configured class, morphological cleanup, contour extraction, then geometric and
area filtering. Surviving blobs are mapped through the calibration homography
into table-frame metres and returned as
:class:`~samplesort.perception.types.Detection` objects.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.perception.calibration import Calibration
from samplesort.perception.classifier import ColorClassifier, QRReader
from samplesort.perception.fallen import rack_mask
from samplesort.perception.types import Detection, FallenTube, Scene

logger = logging.getLogger(__name__)

#: How close a decoded QR code must be to a detection to be attached to it.
QR_ASSOCIATION_RADIUS_PX = 60.0


#: Caps at least this elongated are checked for a body lying beside them.
_BODY_CHECK_ASPECT_MIN = 1.10

#: Fraction of probe points that must look like tube body for a body to count.
_BODY_BESIDE_FRACTION = 0.75


def _body_like(frame: np.ndarray) -> np.ndarray:
    """Pixels that could be translucent tube body: unsaturated, unlike the table."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    table = float(np.median(gray))
    mask: np.ndarray = (hsv[:, :, 1] < 40) & (np.abs(gray - table) > 10.0)
    return mask


def _body_beside(body_like: np.ndarray, centre: tuple[float, float], cap_width_px: float) -> bool:
    """Whether a tube body extends straight out from the blob in some direction.

    Probes rays every 10 degrees from 0.6 to 1.8 cap widths out — past the
    cap's own half-length and within the body's length, so no calibration is
    needed. All directions are tried because a nearly square outline does not
    say which way the tube's axis runs.
    """
    rows, cols = body_like.shape
    for angle in np.radians(np.arange(0.0, 360.0, 10.0)):
        dx, dy = float(np.cos(angle)), float(np.sin(angle))
        hits = total = 0
        for k in np.linspace(0.6, 1.8, 9):
            col = int(round(centre[0] + k * cap_width_px * dx))
            row = int(round(centre[1] + k * cap_width_px * dy))
            if 0 <= row < rows and 0 <= col < cols:
                hits += bool(body_like[row, col])
                total += 1
        if total and hits / total >= _BODY_BESIDE_FRACTION:
            return True
    return False


@dataclass(frozen=True)
class _Blob:
    """One class-coloured blob, before it is judged upright or fallen."""

    u: float
    v: float
    area: float
    label: str
    colour_confidence: float
    circularity: float
    aspect: float
    #: Unit vector along the blob's short side in image coordinates.
    short_axis: tuple[float, float]
    #: Whether a tube body extends from the blob in some direction.
    body_beside: bool = False


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

    def _blobs(self, frame: np.ndarray) -> list[_Blob]:
        """Every class-coloured blob that passes the area and colour checks."""
        hsv = self.preprocess(frame)
        body_like = _body_like(frame)
        if self.calibration is not None:
            # Rack plates are grey and holed, like tube body; a standing tube in a
            # rack near the image edge looks slightly oval and would otherwise
            # pass for a fallen one.
            body_like &= ~rack_mask(body_like.shape, self.calibration, self.config.workspace.racks)
        blobs: list[_Blob] = []

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

                (_, _), (width, height), angle = cv2.minAreaRect(contour)
                long_side, short_side = max(width, height), max(min(width, height), 1e-6)
                corners = cv2.boxPoints(((u, v), (width, height), angle))
                edge_a, edge_b = corners[1] - corners[0], corners[2] - corners[1]
                short_edge = edge_a if np.linalg.norm(edge_a) < np.linalg.norm(edge_b) else edge_b
                short_axis = short_edge / max(float(np.linalg.norm(short_edge)), 1e-9)
                blobs.append(
                    _Blob(
                        u=u,
                        v=v,
                        area=area,
                        label=label,
                        colour_confidence=float(colour_confidence),
                        circularity=self._circularity(contour),
                        aspect=float(long_side / short_side),
                        short_axis=(float(short_axis[0]), float(short_axis[1])),
                        body_beside=_body_beside(body_like, (u, v), long_side),
                    )
                )
        return blobs

    def _is_fallen(self, blob: _Blob) -> bool:
        """An upright cap is a disc from above; a fallen one shows its side.

        The outline alone is marginal for some lying tubes: seen obliquely, one
        measured 1.22 against a 1.25 threshold. So a slightly elongated cap with
        a tube body lying beside it along its axis also counts as fallen.
        Standing caps measured at most 1.06, so the body check never sees them.
        """
        if blob.aspect >= self.tuning.fallen_aspect_min:
            return True
        return blob.aspect >= _BODY_CHECK_ASPECT_MIN and blob.body_beside

    def _upright_from(self, frame: np.ndarray, blobs: list[_Blob]) -> list[Detection]:
        detections: list[Detection] = []
        for blob in blobs:
            if self._is_fallen(blob):
                continue
            # Caps are circular from overhead, so shape is real evidence.
            confidence = float(blob.colour_confidence * blob.circularity)
            if confidence < self.tuning.min_confidence:
                continue
            detections.append(
                Detection(
                    pixel_xy=(blob.u, blob.v),
                    table_xy=self._to_table(blob.u, blob.v),
                    class_label=blob.label,
                    confidence=confidence,
                    area_px=blob.area,
                )
            )
        detections = self._attach_sample_ids(frame, detections)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _fallen_from(self, blobs: list[_Blob]) -> list[FallenTube]:
        fallen: list[FallenTube] = []
        lying_height = self.config.workspace.tube.cap_radius
        for blob in blobs:
            if not self._is_fallen(blob) or blob.colour_confidence < self.tuning.min_confidence:
                continue
            table_xy = (float("nan"), float("nan"))
            axis_yaw = float("nan")
            if self.calibration is not None:
                table_xy = self.calibration.pixel_to_table(
                    blob.u, blob.v, object_height=lying_height
                )
                # Map a short step along the image axis to get the table direction.
                du, dv = blob.short_axis
                ahead = self.calibration.pixel_to_table(
                    blob.u + du * 6.0, blob.v + dv * 6.0, object_height=lying_height
                )
                axis_yaw = math.atan2(ahead[1] - table_xy[1], ahead[0] - table_xy[0])
            fallen.append(
                FallenTube(
                    pixel_xy=(blob.u, blob.v),
                    table_xy=table_xy,
                    class_label=blob.label,
                    axis_yaw=axis_yaw,
                    confidence=blob.colour_confidence,
                    area_px=blob.area,
                )
            )
        fallen.sort(key=lambda f: f.confidence, reverse=True)
        return fallen

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Find every upright tube cap in a frame.

        Args:
            frame: A BGR ``uint8`` overhead image.

        Returns:
            Detections sorted by descending confidence. Blobs below the
            configured area, circularity or confidence thresholds are dropped,
            and so are side-on caps of fallen tubes (see :meth:`detect_fallen`).
        """
        detections = self._upright_from(frame, self._blobs(frame))
        logger.debug("detected %d tubes", len(detections))
        return detections

    def detect_fallen(self, frame: np.ndarray) -> list[FallenTube]:
        """Find tubes lying on their side.

        Args:
            frame: A BGR ``uint8`` overhead image.

        Returns:
            Fallen tubes sorted by descending confidence.
        """
        return self._fallen_from(self._blobs(frame))

    def detect_scene(self, frame: np.ndarray) -> Scene:
        """Upright and fallen tubes from one pass over a frame, filtered to the work area.

        Upright tubes count when they are in the pickup zone or at the righting
        station, where a righted tube waits to be sorted. Fallen tubes count in
        the pickup zone and at the station (a tube left lying there blocks it).
        """
        blobs = self._blobs(frame)
        ws = self.config.workspace

        def in_area(xy: tuple[float, float], margin: float = 0.0) -> bool:
            x, y = xy
            if not (np.isfinite(x) and np.isfinite(y)):
                return False
            return ws.pickup_zone.contains(x, y, margin=margin) or ws.near_righting_station(x, y)

        # A tube that falls over at the zone's edge can leave its cap up to one
        # tube length outside it (measured: a cap 4 mm past the edge was dropped
        # before this margin existed).
        fallen_margin = ws.tube.total_height
        return Scene(
            upright=[d for d in self._upright_from(frame, blobs) if in_area(d.table_xy)],
            fallen=[f for f in self._fallen_from(blobs) if in_area(f.table_xy, fallen_margin)],
        )

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
