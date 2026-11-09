#!/usr/bin/env python3
"""Compute the camera-to-table homography from a photo of the ArUco board.

Either grabs a frame from the configured camera or reads one from disk, fits the
homography, reports the residual, and saves the result where the pipeline looks
for it.

Usage:
    python scripts/calibrate_camera.py                      # capture and fit
    python scripts/calibrate_camera.py --image board.png    # fit a saved frame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from samplesort.config import ConfigError, load_config  # noqa: E402
from samplesort.logging_setup import setup_logging  # noqa: E402
from samplesort.perception.calibration import (  # noqa: E402
    DEFAULT_CALIBRATION_NAME,
    CalibrationError,
    calibrate_from_aruco,
)

#: Residual above which a calibration should not be trusted for grasping.
RESIDUAL_WARN_PX = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=None, help="Config directory.")
    parser.add_argument("--image", type=Path, default=None, help="Use this frame instead.")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the result.")
    parser.add_argument("--save-frame", type=Path, default=None, help="Also save the frame used.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")

    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"Could not read an image from {args.image}", file=sys.stderr)
            return 2
        print(f"Using {args.image}")
    else:
        from samplesort.hal.real_camera import RealCamera

        print(f"Capturing from camera index {config.camera.index}...")
        with RealCamera(config.camera) as camera:
            frame = camera.read()

    if args.save_frame is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_frame), frame)
        print(f"Saved the frame to {args.save_frame}")

    try:
        calibration = calibrate_from_aruco(
            frame,
            config.workspace.marker_board,
            config.camera,
            table_height=config.workspace.table_height,
        )
    except CalibrationError as exc:
        print(f"Calibration failed:\n{exc}", file=sys.stderr)
        return 1

    destination = args.output or (config.config_dir / DEFAULT_CALIBRATION_NAME)
    calibration.save(destination)

    print()
    print(f"Saved the calibration to {destination}")
    print(f"  source            : {calibration.source}")
    print(f"  reprojection error: {calibration.rms_error_px:.3f} px")
    print(f"  table-plane error : {calibration.rms_error_m * 1000:.2f} mm")
    print(f"  camera height     : {calibration.camera_height:.3f} m")
    print(f"  optical nadir     : {calibration.camera_nadir_xy}")
    print()

    if calibration.rms_error_px > RESIDUAL_WARN_PX:
        print(
            f"WARNING: a residual above {RESIDUAL_WARN_PX:.1f} px usually means the "
            "board was not flat, was printed at the wrong scale, or is rotated "
            "relative to the mounting convention in docs/hardware_setup.md."
        )
        return 1

    print("Residual looks good. Check it with:  samplesort run --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
