"""Wrist-camera sample-ID scanners.

A scanner answers one question: what sample ID is printed on the cap directly
below the gripper right now? The controller asks it from the pre-grasp pose,
where the camera is a few centimetres above the cap and a QR code that the
overhead camera cannot resolve fills a large part of the frame.

Both implementations share :meth:`IdScannerInterface.decode_nearest_centre`:
neighbouring tubes can be partly in view, and the one being picked is the one
the camera is centred on.
"""

from __future__ import annotations

import abc
import logging

import numpy as np

from samplesort.config import WristCameraConfig
from samplesort.perception.classifier import QRReader

logger = logging.getLogger(__name__)


class ScannerError(RuntimeError):
    """Raised when a scanner cannot capture a frame."""


class IdScannerInterface(abc.ABC):
    """Reads the sample ID of the tube under the gripper.

    Args:
        config: Wrist camera settings.
    """

    def __init__(self, config: WristCameraConfig) -> None:
        """Store the config (see the class docstring for args)."""
        self.config = config
        self.reader = QRReader(enabled=True)
        self._connected = False

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the camera."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release the camera."""

    @abc.abstractmethod
    def capture(self) -> np.ndarray:
        """Grab one BGR frame from the wrist camera.

        Raises:
            ScannerError: If no frame could be captured.
        """

    @property
    def is_connected(self) -> bool:
        """Whether :meth:`connect` has been called and not undone."""
        return self._connected

    def decode_nearest_centre(self, frame: np.ndarray) -> str | None:
        """Decode the QR code closest to the centre of ``frame``.

        Args:
            frame: A BGR image from the wrist camera.

        Returns:
            The decoded text, or ``None`` if no code could be read.
        """
        codes = self.reader.read(frame)
        if not codes:
            single = self.reader.read_one(frame)
            return single[0] if single is not None else None
        height, width = frame.shape[:2]
        centre = np.array([width / 2.0, height / 2.0])
        text, _ = min(codes, key=lambda code: float(np.linalg.norm(np.asarray(code[1]) - centre)))
        return text

    def scan(self) -> str | None:
        """Capture a frame and return the sample ID under the gripper, if readable.

        A failed capture is logged and reported as "no ID read" rather than
        raised: a missing ID should be recorded as missing, not stop the sort.
        """
        if not self._connected:
            raise ScannerError("scanner is not connected")
        try:
            frame = self.capture()
        except ScannerError as exc:
            logger.warning("wrist scan failed: %s", exc)
            return None
        sample_id = self.decode_nearest_centre(frame)
        if sample_id is None:
            logger.info("no readable sample ID under the gripper")
        return sample_id

    def __enter__(self) -> IdScannerInterface:
        """Connect on entry to a ``with`` block."""
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Disconnect on exit from a ``with`` block."""
        self.disconnect()


class RealIdScanner(IdScannerInterface):
    """A USB wrist camera read through OpenCV.

    Args:
        config: Wrist camera settings; ``config.index`` must be set.
        capture_factory: Builds the capture object from an index. Defaults to
            :class:`cv2.VideoCapture`; tests inject a fake.
    """

    def __init__(self, config: WristCameraConfig, capture_factory: object | None = None) -> None:
        """Bind to a camera index (see the class docstring for args)."""
        super().__init__(config)
        if config.index is None:
            raise ScannerError("camera.wrist.index must be set to use a real wrist scanner")
        self._factory = capture_factory
        self._capture: object | None = None

    def connect(self) -> None:
        """Open the capture device and request the configured resolution."""
        if self._connected:
            return
        import cv2

        factory = self._factory if self._factory is not None else cv2.VideoCapture
        capture = factory(self.config.index)  # type: ignore[operator]
        if not capture.isOpened():
            raise ScannerError(f"could not open wrist camera at index {self.config.index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture = capture
        self._connected = True

    def disconnect(self) -> None:
        """Release the capture device."""
        if self._capture is not None:
            self._capture.release()  # type: ignore[attr-defined]
        self._capture = None
        self._connected = False

    def capture(self) -> np.ndarray:
        """Read one frame from the device."""
        if self._capture is None:
            raise ScannerError("wrist camera is not open")
        ok, frame = self._capture.read()  # type: ignore[attr-defined]
        if not ok or frame is None:
            raise ScannerError("wrist camera returned no frame")
        return np.asarray(frame, dtype=np.uint8)
