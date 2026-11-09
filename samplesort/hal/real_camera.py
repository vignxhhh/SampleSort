"""USB webcam capture through OpenCV."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from samplesort.config import CameraConfig
from samplesort.hal.camera import CameraError, CameraInterface

logger = logging.getLogger(__name__)

#: Frames read and discarded after opening, to let auto-exposure settle.
WARMUP_FRAMES = 5

#: Retries for a single dropped frame before giving up.
READ_RETRIES = 3


class RealCamera(CameraInterface):
    """An overhead USB camera read through :class:`cv2.VideoCapture`.

    Args:
        config: Camera index, resolution and frame rate.
        backend: OpenCV capture backend, e.g. ``cv2.CAP_V4L2``. ``None`` lets
            OpenCV choose.
    """

    def __init__(self, config: CameraConfig, backend: int | None = None) -> None:
        """Prepare the camera without opening the device yet."""
        super().__init__(config)
        self.backend = backend
        self._capture: cv2.VideoCapture | None = None

    def connect(self) -> None:
        """Open the device, apply the configured format and warm it up.

        Raises:
            CameraError: If the device cannot be opened, or refuses the
                configured resolution.
        """
        if self._connected:
            return

        capture = (
            cv2.VideoCapture(self.config.index, self.backend)
            if self.backend is not None
            else cv2.VideoCapture(self.config.index)
        )
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"could not open camera index {self.config.index}. Check that the "
                f"camera is plugged in, that nothing else is using it, and that "
                f"camera.yaml's index is right (`v4l2-ctl --list-devices` on Linux)."
            )

        width, height = self.config.resolution
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        actual = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != (width, height):
            capture.release()
            raise CameraError(
                f"camera {self.config.index} refused {width}x{height} and is "
                f"delivering {actual[0]}x{actual[1]}. Set camera.yaml's width and "
                f"height to a format the device supports — a calibration fitted at "
                f"one resolution is not valid at another."
            )

        self._capture = capture
        self._connected = True

        # Discard the first few frames; auto-exposure and white balance need a
        # moment, and a dark first frame would wreck the HSV thresholds.
        for _ in range(WARMUP_FRAMES):
            capture.read()

        logger.info("opened camera %d at %dx%d", self.config.index, width, height)

    def disconnect(self) -> None:
        """Release the capture device."""
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._connected = False
        logger.debug("released camera %d", self.config.index)

    def read(self) -> np.ndarray:
        """Grab one BGR frame.

        Args:
            None.

        Returns:
            A ``uint8`` array of shape ``(height, width, 3)`` in BGR order.

        Raises:
            CameraError: If the camera is not connected, or every retry produced
                a dropped frame.
        """
        self._require_connected()
        assert self._capture is not None

        for attempt in range(READ_RETRIES):
            ok, frame = self._capture.read()
            if ok and frame is not None:
                return self._validate_frame(frame)
            logger.warning(
                "dropped frame from camera %d (attempt %d/%d)",
                self.config.index,
                attempt + 1,
                READ_RETRIES,
            )

        raise CameraError(
            f"camera {self.config.index} returned no frame after {READ_RETRIES} "
            f"attempts; it may have been unplugged"
        )
