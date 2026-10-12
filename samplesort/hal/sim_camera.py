"""Virtual overhead camera that renders the PyBullet scene."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from samplesort.config import CameraConfig
from samplesort.hal.camera import CameraError, CameraInterface
from samplesort.sim._bullet import pb
from samplesort.sim.world import SimWorld

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Lighting:
    """Scene lighting plus the camera's response to it, for robustness testing.

    The first group is passed to PyBullet's renderer; the second is applied to
    the rendered image the way a real camera's exposure, white balance and
    sensor noise would be.

    Attributes:
        direction: Direction the light comes from, in the world frame.
        color: RGB light colour in ``[0, 1]``; tints the whole scene.
        ambient: Ambient coefficient.
        diffuse: Diffuse coefficient.
        specular: Specular coefficient.
        shadow: Whether to cast shadows.
        exposure: Multiplier on pixel values (under- or over-exposure).
        white_balance: Per-channel RGB gains (a camera's colour cast).
        noise_std: Standard deviation of Gaussian sensor noise, in 8-bit levels.
        vignette: Brightness lost at the frame corners, in ``[0, 1)``: a lens's
            falloff, or a single lamp lighting the middle of the table more than
            its edges. Unlike exposure or white balance it varies across the
            frame, so no single global correction can undo it.
        seed: Seeds the sensor noise so a frame is reproducible.
    """

    direction: tuple[float, float, float] = (0.4, -0.6, 1.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient: float = 0.6
    diffuse: float = 0.35
    specular: float = 0.05
    shadow: bool = False
    exposure: float = 1.0
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    noise_std: float = 0.0
    vignette: float = 0.0
    seed: int = 0

    def apply_camera_response(self, rgb: np.ndarray) -> np.ndarray:
        """Apply exposure, white balance and noise to a ``uint8`` RGB image."""
        image = rgb.astype(np.float32) * self.exposure
        image *= np.asarray(self.white_balance, dtype=np.float32)[None, None, :]
        if self.vignette > 0.0:
            rows, cols = image.shape[:2]
            ys, xs = np.mgrid[0:rows, 0:cols].astype(np.float32)
            # Squared distance from the centre, 1.0 at the corners.
            r2 = ((xs + 0.5 - cols / 2) ** 2 + (ys + 0.5 - rows / 2) ** 2) / (
                (cols / 2) ** 2 + (rows / 2) ** 2
            )
            image *= (1.0 - self.vignette * r2)[:, :, None]
        if self.noise_std > 0.0:
            rng = np.random.default_rng(self.seed)
            image += rng.normal(0.0, self.noise_std, size=image.shape).astype(np.float32)
        response: np.ndarray = np.clip(image, 0.0, 255.0).astype(np.uint8)
        return response


class SimCamera(CameraInterface):
    """Renders a top-down RGB view of the simulated workspace.

    The view and projection matrices come straight from
    :class:`~samplesort.config.CameraConfig`, so the rendered geometry matches the
    real overhead rig described by the same file. Because the pose is known
    exactly, the pixel-to-table homography can be derived analytically instead of
    from an ArUco board — see :meth:`projection_matrices`.

    Args:
        config: Camera resolution, pose and field of view.
        world: The simulation world to render.
    """

    def __init__(self, config: CameraConfig, world: SimWorld) -> None:
        """Bind the camera to a simulation world (see the class docstring for args)."""
        super().__init__(config)
        self.world = world
        self._view_matrix: tuple[float, ...] | None = None
        self._projection_matrix: tuple[float, ...] | None = None
        #: ``None`` renders with PyBullet's default lighting, as every result
        #: before the lighting study was produced; set a :class:`Lighting` to vary it.
        self.lighting: Lighting | None = None

    def connect(self) -> None:
        """Compute the view and projection matrices and mark the camera ready."""
        if not self.world.is_connected:
            self.world.connect()
        view, projection = self._compute_matrices()
        self._view_matrix = view
        self._projection_matrix = projection
        self._connected = True
        logger.debug("SimCamera connected at %s", self.config.position)

    def disconnect(self) -> None:
        """Drop the cached matrices."""
        self._view_matrix = None
        self._projection_matrix = None
        self._connected = False

    def _compute_matrices(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Build the OpenGL view and projection matrices from the configured pose."""
        width, height = self.config.resolution
        view = pb.computeViewMatrix(
            cameraEyePosition=list(self.config.position),
            cameraTargetPosition=list(self.config.look_at),
            cameraUpVector=list(self.config.up_axis),
        )
        projection = pb.computeProjectionMatrixFOV(
            fov=self.config.vertical_fov_deg,
            aspect=width / height,
            nearVal=self.config.near,
            farVal=self.config.far,
        )
        return tuple(view), tuple(projection)

    def projection_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(view, projection)`` matrices as 4x4 column-major arrays.

        Raises:
            CameraError: If the camera has not been connected.
        """
        self._require_connected()
        assert self._view_matrix is not None and self._projection_matrix is not None
        view = np.asarray(self._view_matrix, dtype=float).reshape(4, 4, order="F")
        projection = np.asarray(self._projection_matrix, dtype=float).reshape(4, 4, order="F")
        return view, projection

    def world_to_pixel(self, point: np.ndarray) -> tuple[float, float]:
        """Project a world-frame XYZ point into pixel coordinates.

        Args:
            point: A 3-element world position in metres.

        Returns:
            The ``(u, v)`` pixel coordinate, with ``v`` measured downwards from the
            top of the image as OpenCV expects.

        Raises:
            CameraError: If the camera is not connected or the point is behind it.
        """
        view, projection = self.projection_matrices()
        width, height = self.config.resolution
        homogeneous = np.append(np.asarray(point, dtype=float)[:3], 1.0)
        clip = projection @ (view @ homogeneous)
        if abs(clip[3]) < 1e-12:
            raise CameraError(f"point {point} projects to a degenerate clip coordinate")
        ndc = clip[:3] / clip[3]
        u = (ndc[0] * 0.5 + 0.5) * width
        v = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
        return (float(u), float(v))

    def read(self) -> np.ndarray:
        """Render one BGR frame of the current scene.

        Returns:
            A ``uint8`` array of shape ``(height, width, 3)`` in BGR channel order.

        Raises:
            CameraError: If the camera is not connected or the render failed.
        """
        self._require_connected()
        if not self.world.is_connected:
            raise CameraError("simulation world is not connected")
        width, height = self.config.resolution

        light: dict[str, object] = {}
        if self.lighting is not None:
            light = {
                "lightDirection": list(self.lighting.direction),
                "lightColor": list(self.lighting.color),
                "lightAmbientCoeff": self.lighting.ambient,
                "lightDiffuseCoeff": self.lighting.diffuse,
                "lightSpecularCoeff": self.lighting.specular,
                "shadow": int(self.lighting.shadow),
            }
        _, _, rgba, _, _ = self.world.bullet.getCameraImage(
            width=width,
            height=height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix,
            renderer=pb.ER_TINY_RENDERER,
            flags=pb.ER_NO_SEGMENTATION_MASK,
            **light,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        if self.lighting is not None:
            rgb = self.lighting.apply_camera_response(rgb)
        bgr = np.ascontiguousarray(rgb[:, :, [2, 1, 0]])
        return self._validate_frame(bgr)
