"""Tests for camera-to-table calibration.

The ArUco tests render a synthetic board image from the shipped board config, so
they exercise the real OpenCV detection path without needing a printed target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from samplesort.config import CameraConfig, SampleSortConfig
from samplesort.perception.calibration import (
    Calibration,
    CalibrationError,
    board_corner_table_positions,
    calibrate_from_aruco,
    calibration_from_camera_pose,
    optical_nadir,
)
from tests.synthetic import render_aruco_board, table_to_synth_pixel

# ------------------------------------------------------------------- Calibration


def _identity_calibration() -> Calibration:
    return Calibration(homography=np.eye(3), image_size=(640, 480))


def test_rejects_a_non_square_homography() -> None:
    with pytest.raises(CalibrationError, match="must be 3x3"):
        Calibration(homography=np.eye(2), image_size=(640, 480))


def test_rejects_a_singular_homography() -> None:
    with pytest.raises(CalibrationError, match="singular"):
        Calibration(homography=np.zeros((3, 3)), image_size=(640, 480))


def test_pixel_and_table_mappings_are_inverses() -> None:
    homography = np.array([[0.001, 0.0, -0.3], [0.0, -0.001, 0.25], [0.0, 0.0, 1.0]])
    calibration = Calibration(homography=homography, image_size=(640, 480))
    for u, v in [(0.0, 0.0), (320.0, 240.0), (639.0, 479.0)]:
        x, y = calibration.pixel_to_table(u, v)
        back = calibration.table_to_pixel(x, y)
        assert back == pytest.approx((u, v), abs=1e-9)


def test_batch_mapping_matches_the_single_point_path() -> None:
    homography = np.array([[0.001, 0.0, -0.3], [0.0, -0.001, 0.25], [0.0, 0.0, 1.0]])
    calibration = Calibration(homography=homography, image_size=(640, 480))
    pixels = np.array([[0.0, 0.0], [100.0, 200.0], [639.0, 479.0]])
    batch = calibration.pixels_to_table(pixels)
    for pixel, mapped in zip(pixels, batch, strict=True):
        assert calibration.pixel_to_table(*pixel) == pytest.approx(tuple(mapped), abs=1e-12)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = Calibration(
        homography=np.array([[0.001, 0.0, -0.3], [0.0, -0.001, 0.25], [0.0, 0.0, 1.0]]),
        image_size=(640, 480),
        rms_error_px=0.42,
        rms_error_m=0.0004,
        source="aruco",
        camera_height=0.7,
        camera_nadir_xy=(0.21, 0.0),
    )
    restored = Calibration.load(original.save(tmp_path / "cal.npz"))

    np.testing.assert_allclose(restored.homography, original.homography)
    assert restored.image_size == original.image_size
    assert restored.rms_error_px == pytest.approx(0.42)
    assert restored.rms_error_m == pytest.approx(0.0004)
    assert restored.source == "aruco"
    assert restored.camera_height == pytest.approx(0.7)
    assert restored.camera_nadir_xy == pytest.approx((0.21, 0.0))


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="not found"):
        Calibration.load(tmp_path / "absent.npz")


def test_load_garbage_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.npz"
    path.write_bytes(b"not an npz archive")
    with pytest.raises(CalibrationError, match="could not read calibration"):
        Calibration.load(path)


# ---------------------------------------------------------------------- parallax


def test_parallax_is_a_no_op_without_a_camera_pose() -> None:
    calibration = _identity_calibration()
    assert not calibration.corrects_parallax
    assert calibration.correct_parallax(0.2, 0.1, 0.06) == (0.2, 0.1)


def test_parallax_pulls_points_towards_the_nadir() -> None:
    calibration = Calibration(
        homography=np.eye(3), image_size=(640, 480), camera_height=0.7, camera_nadir_xy=(0.21, 0.0)
    )
    corrected = calibration.correct_parallax(0.30, 0.10, 0.064)
    assert 0.21 < corrected[0] < 0.30
    assert 0.0 < corrected[1] < 0.10


def test_a_point_at_the_nadir_does_not_move() -> None:
    calibration = Calibration(
        homography=np.eye(3), image_size=(640, 480), camera_height=0.7, camera_nadir_xy=(0.21, 0.0)
    )
    assert calibration.correct_parallax(0.21, 0.0, 0.064) == pytest.approx((0.21, 0.0))


def test_zero_height_is_a_no_op() -> None:
    calibration = Calibration(
        homography=np.eye(3), image_size=(640, 480), camera_height=0.7, camera_nadir_xy=(0.21, 0.0)
    )
    assert calibration.correct_parallax(0.30, 0.10, 0.0) == (0.30, 0.10)


def test_object_above_the_camera_raises() -> None:
    calibration = Calibration(
        homography=np.eye(3), image_size=(640, 480), camera_height=0.7, camera_nadir_xy=(0.21, 0.0)
    )
    with pytest.raises(CalibrationError, match="at or above the camera"):
        calibration.correct_parallax(0.3, 0.0, 0.8)


def test_optical_nadir_matches_the_configured_look_at(config: SampleSortConfig) -> None:
    nadir = optical_nadir(config.camera, config.workspace.table_height)
    assert nadir == pytest.approx((config.camera.look_at[0], config.camera.look_at[1]), abs=1e-9)


def test_sideways_camera_has_no_nadir(config: SampleSortConfig) -> None:
    sideways = config.camera.model_copy(
        update={"position": (0.0, 0.0, 0.3), "look_at": (1.0, 0.0, 0.3)}
    )
    with pytest.raises(CalibrationError, match="parallel to the table plane"):
        optical_nadir(sideways, 0.0)


# ------------------------------------------------------------------- board layout


def test_board_corners_are_square_and_correctly_sized(config: SampleSortConfig) -> None:
    board = config.workspace.marker_board
    corners = board_corner_table_positions(board)
    assert len(corners) >= 4

    for marker_id, quad in corners.items():
        assert quad.shape == (4, 2), marker_id
        sides = [float(np.linalg.norm(quad[i] - quad[(i + 1) % 4])) for i in range(4)]
        assert sides == pytest.approx([board.marker_length] * 4, abs=1e-12)


def test_board_markers_do_not_overlap(config: SampleSortConfig) -> None:
    board = config.workspace.marker_board
    centres = [q.mean(axis=0) for q in board_corner_table_positions(board).values()]
    for i, a in enumerate(centres):
        for b in centres[i + 1 :]:
            assert float(np.linalg.norm(a - b)) > board.marker_length


# ----------------------------------------------------------------- ArUco fitting


def test_calibrates_from_a_synthetic_board(config: SampleSortConfig) -> None:
    frame = render_aruco_board(config)
    calibration = calibrate_from_aruco(frame, config.workspace.marker_board, config.camera)

    assert calibration.source == "aruco"
    assert calibration.rms_error_px < 2.0, "synthetic board should fit to sub-pixel accuracy"
    assert calibration.rms_error_m < 2e-3

    # The fitted mapping must reproduce the renderer's own geometry.
    for quad in board_corner_table_positions(config.workspace.marker_board).values():
        for x, y in quad:
            u, v = table_to_synth_pixel(float(x), float(y))
            if not (0 <= u < 640 and 0 <= v < 480):
                continue
            mapped = calibration.pixel_to_table(u, v)
            assert mapped == pytest.approx((x, y), abs=2e-3)


def test_calibration_fails_loudly_on_a_blank_frame(config: SampleSortConfig) -> None:
    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    with pytest.raises(CalibrationError, match="no ArUco markers found"):
        calibrate_from_aruco(blank, config.workspace.marker_board, config.camera)


def test_unknown_dictionary_name_raises(config: SampleSortConfig) -> None:
    board = config.workspace.marker_board.model_copy(update={"dictionary": "DICT_NOT_REAL"})
    with pytest.raises(CalibrationError, match="unknown ArUco dictionary"):
        calibrate_from_aruco(np.zeros((480, 640, 3), dtype=np.uint8), board, config.camera)


# ------------------------------------------------------- analytic sim derivation


def test_analytic_calibration_needs_a_callable(config: SampleSortConfig) -> None:
    with pytest.raises(CalibrationError, match="must be callable"):
        calibration_from_camera_pose(config.camera, config.workspace, "not callable")  # type: ignore[arg-type]


def test_analytic_calibration_inverts_a_known_projection(config: SampleSortConfig) -> None:
    camera: CameraConfig = config.camera

    def project(point: np.ndarray) -> tuple[float, float]:
        return table_to_synth_pixel(float(point[0]), float(point[1]))

    calibration = calibration_from_camera_pose(camera, config.workspace, project)
    assert calibration.source == "sim"
    assert calibration.camera_height == pytest.approx(camera.position[2])
    assert calibration.camera_nadir_xy is not None

    for x, y in [(0.10, 0.05), (0.25, -0.12), (0.30, 0.0)]:
        u, v = table_to_synth_pixel(x, y)
        assert calibration.pixel_to_table(u, v) == pytest.approx((x, y), abs=1e-6)
