"""A wrist camera in simulation, rendered from the arm's tool frame."""

from __future__ import annotations

import numpy as np

from samplesort.config import WristCameraConfig
from samplesort.hal.scanner import IdScannerInterface, ScannerError
from samplesort.sim._bullet import pb
from samplesort.sim.world import SimWorld

#: Near clip plane. The camera sits just below the tool body, so this has to be
#: small enough not to slice off the cap it is looking at.
_NEAR_M = 0.002
_FAR_M = 1.0


class SimIdScanner(IdScannerInterface):
    """Renders what a camera on the gripper sees and decodes the QR label in it.

    The camera is placed ``config.standoff_m`` beyond the tool centre point
    along the tool's approach axis — just clear of the tool body — and looks
    along that axis, rolling with the wrist exactly as a mounted camera would.

    Args:
        config: Wrist camera settings.
        world: The simulation world to render.
    """

    def __init__(self, config: WristCameraConfig, world: SimWorld) -> None:
        """Bind to a world (see the class docstring for args)."""
        super().__init__(config)
        self.world = world

    def connect(self) -> None:
        """Mark the scanner ready; rendering needs no device."""
        self._connected = True

    def disconnect(self) -> None:
        """Mark the scanner closed."""
        self._connected = False

    def view_matrix(self) -> tuple[float, ...]:
        """The camera's view matrix for the arm's current pose."""
        position, orientation = self.world.tool_pose()
        rotation = np.asarray(pb.getMatrixFromQuaternion(orientation.tolist())).reshape(3, 3)
        approach = rotation[:, 0]  # the tool's +X points from the wrist towards the target
        up = rotation[:, 1]
        eye = position + approach * self.config.standoff_m
        target = eye + approach
        return tuple(pb.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist()))

    def capture(self) -> np.ndarray:
        """Render one BGR frame from the wrist camera."""
        if not self.world.is_connected:
            raise ScannerError("simulation world is not connected")
        width, height = self.config.resolution
        projection = pb.computeProjectionMatrixFOV(
            self.config.vertical_fov_deg, width / height, _NEAR_M, _FAR_M
        )
        _, _, rgba, _, _ = self.world.bullet.getCameraImage(
            width=width,
            height=height,
            viewMatrix=self.view_matrix(),
            projectionMatrix=projection,
            renderer=pb.ER_TINY_RENDERER,
            flags=pb.ER_NO_SEGMENTATION_MASK,
        )
        frame = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(frame[:, :, [2, 1, 0]])
