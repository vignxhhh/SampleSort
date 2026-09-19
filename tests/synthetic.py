"""Synthetic image renderers shared by the perception tests.

Everything the perception tests look at is drawn here from the shipped config, so
the suite carries no fixture images and needs no camera or simulator.
"""

from __future__ import annotations

import cv2
import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.perception.calibration import Calibration, board_corner_table_positions

#: Pixels per metre used by the synthetic renderers.
SYNTH_SCALE = 1200.0
#: Pixel of the table-frame origin in synthetic images.
SYNTH_ORIGIN = (320.0, 420.0)
#: Default synthetic frame size, matching the shipped camera config.
SYNTH_SIZE = (640, 480)

#: Unambiguous BGR swatches for each configured class.
CAP_BGR = {
    "red": (40, 40, 200),
    "blue": (200, 90, 40),
    "green": (60, 170, 60),
    "yellow": (50, 200, 220),
}
#: Radius in pixels of a synthetic cap, close to how the sim renders them.
CAP_RADIUS_PX = 9


def table_to_synth_pixel(x: float, y: float) -> tuple[float, float]:
    """Project a table point into a synthetic image.

    Mirrors SampleSort's mounting convention: table +X points up the image and
    table +Y points left.

    Args:
        x: Table-frame X in metres.
        y: Table-frame Y in metres.

    Returns:
        The ``(u, v)`` pixel coordinate.
    """
    u0, v0 = SYNTH_ORIGIN
    return (u0 - y * SYNTH_SCALE, v0 - x * SYNTH_SCALE)


def synth_calibration() -> Calibration:
    """A calibration that exactly inverts :func:`table_to_synth_pixel`."""
    u0, v0 = SYNTH_ORIGIN
    homography = np.array(
        [
            [0.0, -1.0 / SYNTH_SCALE, v0 / SYNTH_SCALE],
            [-1.0 / SYNTH_SCALE, 0.0, u0 / SYNTH_SCALE],
            [0.0, 0.0, 1.0],
        ]
    )
    return Calibration(homography=homography, image_size=SYNTH_SIZE, source="sim")


def synth_frame(
    caps: list[tuple[str, tuple[float, float]]], size: tuple[int, int] = SYNTH_SIZE
) -> np.ndarray:
    """Draw coloured caps at known table positions on a neutral background.

    Args:
        caps: ``(class_label, (x, y))`` pairs in table-frame metres.
        size: Output ``(width, height)`` in pixels.

    Returns:
        A BGR ``uint8`` frame.
    """
    width, height = size
    frame = np.full((height, width, 3), 215, dtype=np.uint8)
    for label, (x, y) in caps:
        u, v = table_to_synth_pixel(x, y)
        cv2.circle(frame, (int(round(u)), int(round(v))), CAP_RADIUS_PX, CAP_BGR[label], -1)
    return frame


def render_aruco_board(config: SampleSortConfig, size: tuple[int, int] = SYNTH_SIZE) -> np.ndarray:
    """Draw the configured ArUco board onto a blank frame at known positions.

    Args:
        config: The configuration bundle supplying the board layout.
        size: Output ``(width, height)`` in pixels.

    Returns:
        A BGR ``uint8`` frame showing every marker that fits in the image.
    """
    board = config.workspace.marker_board
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))
    width, height = size
    canvas = np.full((height, width), 255, dtype=np.uint8)

    for marker_id, corners in board_corner_table_positions(board).items():
        top_left = table_to_synth_pixel(*corners[0])
        bottom_right = table_to_synth_pixel(*corners[2])
        u0, v0 = int(round(top_left[0])), int(round(top_left[1]))
        u1, v1 = int(round(bottom_right[0])), int(round(bottom_right[1]))
        side = max(u1 - u0, v1 - v0)
        if side < 8 or u0 < 0 or v0 < 0 or u0 + side > width or v0 + side > height:
            continue
        canvas[v0 : v0 + side, u0 : u0 + side] = cv2.aruco.generateImageMarker(
            dictionary, marker_id, side
        )

    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
