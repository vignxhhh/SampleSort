"""Decide which end of a held, horizontal tube is the body.

A fallen tube is gripped by its cap, which only needs the tube's axis — the
overhead camera measures that to within a few degrees. Which way the body
points cannot be read reliably where the tube fell: neighbouring caps and the
grey bodies of standing tubes beside them look like a tube body too (95% in a
measured sweep of cluttered layouts). So the question is asked later, at the
righting station, with the tube held horizontally above a clear table.

The body is translucent plastic: unsaturated, and shaded darker at its edges and
brighter along its highlight. The table is flat and uniform; caps are saturated.
So "unsaturated and noticeably different from the table" marks body pixels, and
the side of the gripper with more of them is the body side.

Two things measured in sim shape the probe. The arm's own links sit between the
camera and the inner part of the held body, so probing starts well out along it.
And the tube twists a few degrees in the grip while it is carried, so each side
is probed along a fan of directions and the best-matching one counts.
"""

from __future__ import annotations

import cv2
import numpy as np

from samplesort.config import TubeConfig
from samplesort.perception.calibration import Calibration

#: Pixels more saturated than this are a cap (or something else coloured), not a body.
_MAX_BODY_SATURATION = 40

#: How far a pixel's brightness must sit from the table's to count as body.
_MIN_TABLE_CONTRAST = 10.0

#: The body-side score difference below which the answer is "cannot tell".
MIN_SIDE_MARGIN = 0.2

#: Directions probed either side of the nominal axis, to absorb twist in the grip.
_FAN_DEG = (-24.0, -16.0, -8.0, 0.0, 8.0, 16.0, 24.0)


def body_side_scores(
    frame: np.ndarray,
    calibration: Calibration,
    cap_xyz: np.ndarray,
    axis: np.ndarray,
    tube: TubeConfig,
) -> tuple[float, float]:
    """Score how body-like each side of a held cap looks.

    Args:
        frame: A BGR overhead frame taken while the tube is held.
        calibration: Maps 3-D points into the frame.
        cap_xyz: Position of the held cap's centre (the tool centre point).
        axis: Horizontal unit vector along the tube's axis.
        tube: Tube dimensions.

    Returns:
        ``(score_along_axis, score_against_axis)``: the fraction of probe points
        that look like tube body on each side, each in ``[0, 1]``.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    table = float(np.median(gray))
    body_like = (hsv[:, :, 1] < _MAX_BODY_SATURATION) & (np.abs(gray - table) > _MIN_TABLE_CONTRAST)

    nominal = np.asarray(axis, dtype=float)[:2]
    nominal = nominal / max(float(np.linalg.norm(nominal)), 1e-9)
    height = float(cap_xyz[2])

    # Start beyond the arm links that hide the inner body from above, and stop
    # short of the body's far end.
    start = tube.cap_height / 2.0 + 0.020
    stop = tube.cap_height / 2.0 + tube.body_height * 0.95
    rows, cols = body_like.shape
    scores: list[float] = []
    for sign in (1.0, -1.0):
        best = 0.0
        for degrees in _FAN_DEG:
            angle = np.radians(degrees)
            c, s = np.cos(angle), np.sin(angle)
            direction = np.array([c * nominal[0] - s * nominal[1], s * nominal[0] + c * nominal[1]])
            across = np.array([-direction[1], direction[0]])
            hits = total = 0
            for along in np.linspace(start, stop, 10):
                for offset in (-tube.body_radius * 0.4, 0.0, tube.body_radius * 0.4):
                    point = cap_xyz[:2] + direction * sign * along + across * offset
                    u, v = calibration.point_to_pixel(float(point[0]), float(point[1]), height)
                    col, row = int(round(u)), int(round(v))
                    if 0 <= row < rows and 0 <= col < cols:
                        hits += bool(body_like[row, col])
                        total += 1
            if total:
                best = max(best, hits / total)
        scores.append(best)
    return (scores[0], scores[1])


def body_side(
    frame: np.ndarray,
    calibration: Calibration,
    cap_xyz: np.ndarray,
    axis: np.ndarray,
    tube: TubeConfig,
) -> float | None:
    """Which way along ``axis`` the held tube's body extends.

    Returns:
        ``+1.0`` if the body lies along ``axis``, ``-1.0`` if against it, or
        ``None`` when the two sides are too close to call.
    """
    along, against = body_side_scores(frame, calibration, cap_xyz, axis, tube)
    if abs(along - against) < MIN_SIDE_MARGIN:
        return None
    return 1.0 if along > against else -1.0
