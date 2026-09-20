#!/usr/bin/env python3
"""Render the ArUco calibration board described by ``workspace.yaml``.

The board is the physical reference that ties image pixels to table coordinates,
so it must be printed at exactly the scale the config declares. This script emits
a PNG at a chosen DPI and prints the physical size to check against a ruler.

Usage:
    python scripts/generate_aruco_board.py --output docs/aruco_board.png --dpi 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from samplesort.config import ConfigError, load_config  # noqa: E402
from samplesort.perception.calibration import board_corner_table_positions  # noqa: E402

MM_PER_INCH = 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=None, help="Config directory.")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/aruco_board.png"), help="Output PNG."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Print resolution.")
    parser.add_argument(
        "--margin-mm", type=float, default=10.0, help="White border around the board."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    board = config.workspace.marker_board
    pixels_per_metre = args.dpi / MM_PER_INCH * 1000.0
    margin_px = int(round(args.margin_mm / 1000.0 * pixels_per_metre))

    # `board_corner_table_positions` indexes columns along table +X and rows along
    # table +Y. The camera convention puts +X up the image and +Y to the left, so
    # the X extent becomes the image's height and the Y extent its width.
    x_extent_m = board.squares_x * board.square_length
    y_extent_m = board.squares_y * board.square_length
    width_px = int(round(y_extent_m * pixels_per_metre)) + 2 * margin_px
    height_px = int(round(x_extent_m * pixels_per_metre)) + 2 * margin_px

    canvas = np.full((height_px, width_px), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))

    # Marker ids and positions come from the same function the calibrator uses,
    # so the printed board and the solver can never disagree about the layout.
    origin_x, origin_y = board.origin
    for marker_id, corners in board_corner_table_positions(board).items():
        xs, ys = corners[:, 0], corners[:, 1]
        # Table +X runs up the image and +Y runs left, matching camera.yaml.
        left_m = (origin_y + y_extent_m) - float(ys.max())
        top_m = (origin_x + x_extent_m) - float(xs.max())
        side_px = int(round(board.marker_length * pixels_per_metre))
        u0 = margin_px + int(round(left_m * pixels_per_metre))
        v0 = margin_px + int(round(top_m * pixels_per_metre))
        canvas[v0 : v0 + side_px, u0 : u0 + side_px] = cv2.aruco.generateImageMarker(
            dictionary, marker_id, side_px
        )
        cv2.putText(
            canvas,
            str(marker_id),
            (u0, v0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0,),
            1,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)

    print(f"Wrote {args.output}")
    print(f"  dictionary : {board.dictionary}")
    print(f"  grid       : {board.squares_x} x {board.squares_y} squares")
    print(f"  square     : {board.square_length * 1000:.1f} mm")
    print(f"  marker     : {board.marker_length * 1000:.1f} mm")
    print(f"  board size : {x_extent_m * 1000:.1f} mm (X) x {y_extent_m * 1000:.1f} mm (Y)")
    print(f"  image      : {width_px} x {height_px} px at {args.dpi} DPI")
    print()
    print("Print at 100% scale (no 'fit to page'), then measure one marker with a")
    print(f"ruler: it must be exactly {board.marker_length * 1000:.1f} mm across.")
    print("A board printed at the wrong scale produces a calibration that is wrong")
    print("by that same factor, silently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
