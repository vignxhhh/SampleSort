"""Camera-to-table calibration.

Two paths produce the same artefact — a 3x3 homography mapping image pixels to
table-plane XY in the arm base frame:

* **Real mode** detects a printed ArUco board whose square positions are known
  from ``workspace.yaml`` and fits the homography to those correspondences.
* **Sim mode** derives it analytically by projecting known table points through
  the virtual camera's view and projection matrices, which is exact.

Both are saved to and loaded from the same ``.npz`` file, so downstream code
never needs to know which one produced it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from samplesort.config import CameraConfig, MarkerBoardConfig, WorkspaceConfig

logger = logging.getLogger(__name__)

#: Default filename for a saved calibration, relative to the config directory.
DEFAULT_CALIBRATION_NAME = "homography.npz"


class CalibrationError(RuntimeError):
    """Raised when a calibration cannot be computed, saved or loaded."""


@dataclass
class Calibration:
    """A pixel-to-table-plane mapping plus the metadata needed to trust it.

    The homography is fitted on the table plane, which is where the printed ArUco
    board lies. Objects standing on the table are seen from an angle, so their
    tops project further from the camera's nadir than they really are. Supplying
    :attr:`camera_height` and :attr:`camera_nadir_xy` lets
    :meth:`pixel_to_table` undo that parallax for a known object height.

    Attributes:
        homography: 3x3 matrix mapping homogeneous pixel coordinates to table XY.
        image_size: The ``(width, height)`` the homography was fitted at.
        rms_error_px: Reprojection error of the fit, in pixels — the residual
            when the known table points are mapped back into the image.
        rms_error_m: Residual of the fit on the table plane, in metres.
        source: How the calibration was produced — ``"sim"`` or ``"aruco"``.
        camera_height: Camera height above the table plane, in metres.
        camera_nadir_xy: Where the optical axis meets the table plane, in metres.
    """

    homography: np.ndarray
    image_size: tuple[int, int]
    rms_error_px: float = 0.0
    rms_error_m: float = 0.0
    source: str = "sim"
    camera_height: float | None = None
    camera_nadir_xy: tuple[float, float] | None = None

    @property
    def corrects_parallax(self) -> bool:
        """Whether this calibration knows enough to undo height parallax."""
        return (
            self.camera_height is not None
            and self.camera_nadir_xy is not None
            and self.camera_height > 0.0
        )

    def correct_parallax(self, x: float, y: float, object_height: float) -> tuple[float, float]:
        """Undo the parallax shift for a feature at a known height.

        A feature ``h`` above the table plane, seen by a pinhole camera ``H``
        above that plane, projects onto the plane displaced away from the nadir
        by a factor ``H / (H - h)``. Scaling back by ``(H - h) / H`` recovers the
        feature's true footprint.

        Args:
            x: Table-plane X returned by the raw homography, in metres.
            y: Table-plane Y returned by the raw homography, in metres.
            object_height: Height of the observed feature above the table plane.

        Returns:
            The corrected ``(x, y)``. Returns the input unchanged when the
            calibration has no camera pose or the height is zero.

        Raises:
            CalibrationError: If the feature is at or above the camera itself.
        """
        if object_height == 0.0 or not self.corrects_parallax:
            return (x, y)
        assert self.camera_height is not None and self.camera_nadir_xy is not None
        if object_height >= self.camera_height:
            raise CalibrationError(
                f"object height {object_height:.3f} m is at or above the camera "
                f"({self.camera_height:.3f} m); parallax is undefined"
            )
        scale = (self.camera_height - object_height) / self.camera_height
        nadir_x, nadir_y = self.camera_nadir_xy
        return (nadir_x + (x - nadir_x) * scale, nadir_y + (y - nadir_y) * scale)

    def __post_init__(self) -> None:
        """Validate the homography's shape and invertibility."""
        self.homography = np.asarray(self.homography, dtype=float)
        if self.homography.shape != (3, 3):
            raise CalibrationError(f"homography must be 3x3, got {self.homography.shape}")
        if abs(float(np.linalg.det(self.homography))) < 1e-12:
            raise CalibrationError("homography is singular and cannot be inverted")

    # ------------------------------------------------------------------ mapping

    def pixel_to_table(
        self, u: float, v: float, *, object_height: float = 0.0
    ) -> tuple[float, float]:
        """Map one image pixel onto the table plane.

        Args:
            u: Column coordinate in pixels.
            v: Row coordinate in pixels.
            object_height: Height above the table plane of the feature that
                produced this pixel. Non-zero values trigger the parallax
                correction described on the class.

        Returns:
            The ``(x, y)`` table-frame position in metres.

        Raises:
            CalibrationError: If the pixel maps to a point at infinity.
        """
        point = self.homography @ np.array([u, v, 1.0], dtype=float)
        if abs(point[2]) < 1e-12:
            raise CalibrationError(f"pixel ({u}, {v}) maps to a point at infinity")
        x, y = float(point[0] / point[2]), float(point[1] / point[2])
        return self.correct_parallax(x, y, object_height)

    def table_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Map a table-frame point back into image coordinates.

        Args:
            x: Table-frame X in metres.
            y: Table-frame Y in metres.

        Returns:
            The ``(u, v)`` pixel coordinate.
        """
        inverse = np.linalg.inv(self.homography)
        point = inverse @ np.array([x, y, 1.0], dtype=float)
        if abs(point[2]) < 1e-12:
            raise CalibrationError(f"table point ({x}, {y}) maps to a pixel at infinity")
        return (float(point[0] / point[2]), float(point[1] / point[2]))

    def pixels_to_table(self, points: np.ndarray) -> np.ndarray:
        """Map an ``(N, 2)`` array of pixels onto the table plane in one call."""
        pixels = np.asarray(points, dtype=float).reshape(-1, 2)
        homogeneous = np.hstack([pixels, np.ones((len(pixels), 1))])
        mapped = homogeneous @ self.homography.T
        return np.asarray(mapped[:, :2] / mapped[:, 2:3], dtype=float)

    # -------------------------------------------------------------- persistence

    def save(self, path: Path | str) -> Path:
        """Write the calibration to a ``.npz`` file.

        Args:
            path: Destination file. Parent directories are created.

        Returns:
            The path that was written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        nadir = self.camera_nadir_xy if self.camera_nadir_xy is not None else (np.nan, np.nan)
        np.savez(
            target,
            homography=self.homography,
            image_size=np.asarray(self.image_size, dtype=int),
            rms_error_px=np.asarray(self.rms_error_px, dtype=float),
            rms_error_m=np.asarray(self.rms_error_m, dtype=float),
            source=np.asarray(self.source),
            camera_height=np.asarray(
                np.nan if self.camera_height is None else self.camera_height, dtype=float
            ),
            camera_nadir_xy=np.asarray(nadir, dtype=float),
        )
        logger.info("saved %s calibration to %s", self.source, target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> Calibration:
        """Read a calibration back from a ``.npz`` file.

        Raises:
            CalibrationError: If the file is missing or does not hold a calibration.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise CalibrationError(f"calibration file not found: {source_path}")
        try:
            data = np.load(source_path, allow_pickle=False)
            height = float(data["camera_height"]) if "camera_height" in data else float("nan")
            nadir_raw = (
                np.asarray(data["camera_nadir_xy"], dtype=float)
                if "camera_nadir_xy" in data
                else np.array([np.nan, np.nan])
            )
            nadir = (
                None
                if not np.all(np.isfinite(nadir_raw))
                else (float(nadir_raw[0]), float(nadir_raw[1]))
            )
            return cls(
                homography=data["homography"],
                image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
                rms_error_px=float(data["rms_error_px"]),
                rms_error_m=float(data["rms_error_m"]) if "rms_error_m" in data else 0.0,
                source=str(data["source"]),
                camera_height=None if not np.isfinite(height) else height,
                camera_nadir_xy=nadir,
            )
        except (KeyError, ValueError, OSError) as exc:
            raise CalibrationError(f"could not read calibration from {source_path}: {exc}") from exc


def optical_nadir(camera: CameraConfig, table_height: float) -> tuple[float, float]:
    """Where the camera's optical axis meets the table plane.

    Args:
        camera: Camera configuration supplying the position and look-at point.
        table_height: Z of the table plane in the base frame, in metres.

    Returns:
        The ``(x, y)`` table-frame position of the nadir, in metres.

    Raises:
        CalibrationError: If the optical axis is parallel to the table plane.
    """
    eye = np.asarray(camera.position, dtype=float)
    direction = np.asarray(camera.look_at, dtype=float) - eye
    if abs(direction[2]) < 1e-9:
        raise CalibrationError(
            "the camera's optical axis is parallel to the table plane; a top-down "
            "mount is required for the planar homography to be well defined"
        )
    t = (table_height - eye[2]) / direction[2]
    hit = eye + t * direction
    return (float(hit[0]), float(hit[1]))


def _fit_homography(
    pixels: np.ndarray, table_points: np.ndarray, image_size: tuple[int, int], source: str
) -> Calibration:
    """Least-squares fit a pixel-to-table homography and measure its residual."""
    if len(pixels) < 4:
        raise CalibrationError(f"need at least 4 correspondences, got {len(pixels)}")

    fitted, _ = cv2.findHomography(
        pixels.astype(np.float64), table_points.astype(np.float64), method=0
    )
    if fitted is None:
        raise CalibrationError("cv2.findHomography failed to fit the correspondences")

    # OpenCV's return dtype varies across versions; pin it to float64 so the
    # linear algebra below is well defined and typed.
    homography = np.asarray(fitted, dtype=np.float64)
    calibration = Calibration(homography=homography, image_size=image_size, source=source)

    # Residual on the table plane: how far the fitted mapping misses in metres.
    predicted = calibration.pixels_to_table(pixels)
    calibration.rms_error_m = float(
        np.sqrt(np.mean(np.sum((predicted - table_points) ** 2, axis=1)))
    )

    # Residual back in the image: map the known table points through the inverse
    # and compare against where the corners were actually observed. This is the
    # number a calibration technician cares about.
    inverse = np.linalg.inv(homography)
    homogeneous = np.hstack([table_points, np.ones((len(table_points), 1))])
    reprojected = homogeneous @ inverse.T
    reprojected = reprojected[:, :2] / reprojected[:, 2:3]
    calibration.rms_error_px = float(np.sqrt(np.mean(np.sum((reprojected - pixels) ** 2, axis=1))))
    return calibration


def calibration_from_camera_pose(
    camera: CameraConfig,
    workspace: WorkspaceConfig,
    project: Callable[[np.ndarray], tuple[float, float]],
) -> Calibration:
    """Derive the homography analytically from a known camera pose.

    Four corners of the table region are projected through the camera's view and
    projection matrices, then a homography is fitted to those exact
    correspondences. Because the projection is exact, the residual is numerical
    noise rather than measurement error.

    Args:
        camera: Camera configuration, used for the image size.
        workspace: Workspace configuration, used to pick well-spread sample points.
        project: A callable taking a world-frame XYZ array and returning ``(u, v)``.
            :meth:`samplesort.hal.sim_camera.SimCamera.world_to_pixel` satisfies this.

    Returns:
        The derived calibration, tagged with source ``"sim"``.

    Raises:
        CalibrationError: If the projection produced degenerate correspondences.
    """
    if not callable(project):
        raise CalibrationError("project must be callable")

    half_x, half_y = workspace.table_size[0] / 2.0, workspace.table_size[1] / 2.0
    # A well-spread grid across the reachable table area beats four tight corners.
    xs = np.linspace(0.02, min(half_x, 0.42), 4)
    ys = np.linspace(-min(half_y, 0.28), min(half_y, 0.28), 4)

    pixels: list[tuple[float, float]] = []
    table_points: list[tuple[float, float]] = []
    for x in xs:
        for y in ys:
            point = np.array([x, y, workspace.table_height], dtype=float)
            pixels.append(project(point))
            table_points.append((float(x), float(y)))

    calibration = _fit_homography(
        np.asarray(pixels, dtype=float),
        np.asarray(table_points, dtype=float),
        camera.resolution,
        source="sim",
    )
    calibration.camera_height = float(camera.position[2]) - workspace.table_height
    calibration.camera_nadir_xy = optical_nadir(camera, workspace.table_height)
    return calibration


def _aruco_dictionary(name: str) -> int:
    """Resolve an ArUco dictionary name to its OpenCV constant.

    Raises:
        CalibrationError: If OpenCV does not know the name.
    """
    constant = getattr(cv2.aruco, name, None)
    if constant is None:
        raise CalibrationError(
            f"unknown ArUco dictionary '{name}'; expected something like 'DICT_4X4_50'"
        )
    return int(constant)


def board_corner_table_positions(board: MarkerBoardConfig) -> dict[int, np.ndarray]:
    """Compute each marker's four table-frame corners for a printed board.

    The board is laid out as a chessboard whose alternating squares carry markers,
    with ids assigned in row-major order. Marker *i* is centred on its square and
    inset by ``(square_length - marker_length) / 2``.

    Corners are returned in OpenCV's image order — top-left, top-right,
    bottom-right, bottom-left — under SampleSort's mounting convention: the
    overhead camera is oriented so that table **+X points up** the image and table
    **+Y points left**, matching ``camera.yaml``'s ``up_axis: [1, 0, 0]``. Rotating
    the printed board relative to that convention will make the fit fail loudly
    with a large residual rather than silently producing a rotated frame.
    ``docs/hardware_setup.md`` describes how to lay the board down.

    Args:
        board: The marker board configuration.

    Returns:
        A mapping from marker id to a ``(4, 2)`` array of table-frame corners.
    """
    inset = (board.square_length - board.marker_length) / 2.0
    size = board.marker_length
    origin_x, origin_y = board.origin

    corners: dict[int, np.ndarray] = {}
    marker_id = 0
    for row in range(board.squares_y):
        for col in range(board.squares_x):
            if (row + col) % 2 != 0:
                continue  # markers only sit on alternating squares
            x0 = origin_x + col * board.square_length + inset
            y0 = origin_y + row * board.square_length + inset
            corners[marker_id] = np.array(
                [
                    [x0 + size, y0 + size],  # top-left in the image
                    [x0 + size, y0],  # top-right
                    [x0, y0],  # bottom-right
                    [x0, y0 + size],  # bottom-left
                ],
                dtype=float,
            )
            marker_id += 1
    return corners


def calibrate_from_aruco(
    image: np.ndarray,
    board: MarkerBoardConfig,
    camera: CameraConfig,
    *,
    table_height: float = 0.0,
) -> Calibration:
    """Fit the pixel-to-table homography from a photo of the printed board.

    Args:
        image: A BGR frame showing the board flat on the table.
        board: The marker board configuration describing what was printed.
        camera: Camera configuration, used for the image size and mounting pose.
        table_height: Z of the table plane in the base frame, in metres.

    Returns:
        The fitted calibration, tagged with source ``"aruco"``.

    Raises:
        CalibrationError: If too few markers were detected to fit a homography.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(_aruco_dictionary(board.dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)

    if marker_ids is None or len(marker_ids) == 0:
        raise CalibrationError(
            "no ArUco markers found; check lighting, focus and that the printed "
            f"board really uses {board.dictionary}"
        )

    expected = board_corner_table_positions(board)
    pixels: list[tuple[float, float]] = []
    table_points: list[tuple[float, float]] = []
    for corner_set, marker_id in zip(marker_corners, marker_ids.flatten(), strict=True):
        known = expected.get(int(marker_id))
        if known is None:
            logger.warning("ignoring unexpected marker id %d", int(marker_id))
            continue
        for pixel, table_point in zip(corner_set.reshape(4, 2), known, strict=True):
            pixels.append((float(pixel[0]), float(pixel[1])))
            table_points.append((float(table_point[0]), float(table_point[1])))

    if len(pixels) < 4:
        raise CalibrationError(
            f"only {len(pixels)} usable marker corners were found; at least 4 are needed"
        )

    calibration = _fit_homography(
        np.asarray(pixels), np.asarray(table_points), camera.resolution, source="aruco"
    )
    calibration.camera_height = float(camera.position[2]) - table_height
    calibration.camera_nadir_xy = optical_nadir(camera, table_height)
    logger.info(
        "calibrated from %d markers (%d corners), residual %.4f px",
        len(marker_ids),
        len(pixels),
        calibration.rms_error_px,
    )
    return calibration
