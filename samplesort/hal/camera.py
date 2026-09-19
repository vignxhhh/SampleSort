"""Abstract camera interface shared by the simulated and real backends."""

from __future__ import annotations

import abc
import logging

import numpy as np

from samplesort.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or a frame cannot be read."""


class CameraInterface(abc.ABC):
    """Minimal contract every camera backend implements.

    Both backends return BGR ``uint8`` images shaped ``(height, width, 3)`` so
    perception code never needs to know which one it is talking to.
    """

    def __init__(self, config: CameraConfig) -> None:
        """Store the camera configuration.

        Args:
            config: Validated camera resolution, pose and intrinsics settings.
        """
        self.config = config
        self._connected = False

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the camera and make it ready to stream frames."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the camera and release its resources."""

    @abc.abstractmethod
    def read(self) -> np.ndarray:
        """Capture one frame.

        Returns:
            A BGR ``uint8`` array of shape ``(height, width, 3)``.

        Raises:
            CameraError: If the camera is not connected or the frame grab failed.
        """

    @property
    def is_connected(self) -> bool:
        """Whether :meth:`connect` has succeeded and :meth:`disconnect` has not run."""
        return self._connected

    @property
    def resolution(self) -> tuple[int, int]:
        """Configured image size as ``(width, height)``."""
        return self.config.resolution

    def _require_connected(self) -> None:
        """Raise if the camera has not been connected yet."""
        if not self._connected:
            raise CameraError(f"{type(self).__name__} is not connected; call connect() first")

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Check that a frame has the expected dtype and shape.

        Args:
            frame: The captured image.

        Returns:
            The frame unchanged.

        Raises:
            CameraError: If the dtype or shape is wrong.
        """
        width, height = self.config.resolution
        if frame.dtype != np.uint8:
            raise CameraError(f"expected a uint8 frame, got {frame.dtype}")
        if frame.shape != (height, width, 3):
            raise CameraError(f"expected a frame of shape {(height, width, 3)}, got {frame.shape}")
        return frame

    def __enter__(self) -> CameraInterface:
        """Connect on entry to a ``with`` block."""
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Disconnect on exit from a ``with`` block."""
        self.disconnect()
