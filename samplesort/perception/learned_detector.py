"""A small trained cap detector, the learned alternative to the HSV thresholds.

The network is CenterNet-shaped and deliberately tiny (~55k parameters, so it
trains on a CPU in minutes): a strided convolutional trunk, a few dilated
layers for context, and two 1x1 heads — one heatmap per sample class whose
peaks are cap centres, and a two-channel sub-cell offset. The overall output
stride is 8 pixels of the full 640x480 frame.

Every frame is normalised per channel before it reaches the network. That is
the classic gray-world assumption — the table dominates the view and is
neutral, so its average colour *should* be grey — and it removes most of a
lighting colour cast or exposure error before learning has to. Because that
step alone could explain a robustness gain, the detector study also runs the
HSV detector on gray-world-corrected frames: see
:mod:`samplesort.perception.detector_study`.

PyTorch is an optional dependency; it is imported only when a model is built.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from samplesort.config import SampleSortConfig
from samplesort.perception.calibration import Calibration
from samplesort.perception.types import Detection

logger = logging.getLogger(__name__)

#: The network sees the frame at half resolution.
INPUT_SCALE = 0.5

#: Downsampling inside the network, relative to its input.
NET_STRIDE = 4

#: Output cells per full-resolution pixel: one cell covers this many pixels.
OUTPUT_STRIDE = int(NET_STRIDE / INPUT_SCALE)

#: Heatmap peaks below this are not reported.
DEFAULT_THRESHOLD = 0.3


def gray_world(frame: np.ndarray, target_mean: float = 200.0) -> np.ndarray:
    """Correct a frame's colour cast and exposure under the gray-world assumption.

    Each channel is scaled so its mean equals ``target_mean``. Works because the
    neutral table fills most of the overhead view.

    Args:
        frame: A BGR ``uint8`` image.
        target_mean: The brightness every channel's mean is brought to.

    Returns:
        The corrected ``uint8`` image.
    """
    image = frame.astype(np.float32)
    means = image.reshape(-1, 3).mean(axis=0)
    gains = target_mean / np.maximum(means, 1.0)
    corrected: np.ndarray = np.clip(image * gains[None, None, :], 0.0, 255.0).astype(np.uint8)
    return corrected


def prepare_input(frame: np.ndarray) -> np.ndarray:
    """Turn a BGR frame into the network's ``(3, H, W)`` float input.

    Half resolution, RGB order, and per-channel standardisation (which also
    performs the gray-world correction).
    """
    small = cv2.resize(frame, None, fx=INPUT_SCALE, fy=INPUT_SCALE, interpolation=cv2.INTER_AREA)
    rgb = small[:, :, ::-1].astype(np.float32)
    mean = rgb.reshape(-1, 3).mean(axis=0)
    std = float(rgb.std()) + 1e-6
    return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1))


def build_model(num_classes: int) -> Any:
    """Build the cap-detection network.

    Args:
        num_classes: Number of sample classes (heatmap channels).

    Returns:
        A ``torch.nn.Module`` mapping ``(N, 3, H, W)`` to a dict with
        ``heatmap`` logits ``(N, C, H/4, W/4)`` and ``offset`` ``(N, 2, H/4, W/4)``.
    """
    from samplesort.learning._deps import require_torch

    torch = require_torch()
    nn = torch.nn

    def block(c_in: int, c_out: int, *, stride: int = 1, dilation: int = 1) -> Any:
        return nn.Sequential(
            nn.Conv2d(
                c_in, c_out, 3, stride=stride, padding=dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    class CapNet(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                block(3, 16, stride=2),
                block(16, 24),
                block(24, 32, stride=2),
                block(32, 32, dilation=2),
                block(32, 48, dilation=4),
                block(48, 48),
            )
            self.heatmap = nn.Conv2d(48, num_classes, 1)
            self.offset = nn.Conv2d(48, 2, 1)
            # Start every cell at ~10% "cap here", as CenterNet does, so the
            # focal loss is not swamped by easy negatives at the start.
            nn.init.constant_(self.heatmap.bias, -2.19)

        def forward(self, x: Any) -> dict[str, Any]:
            features = self.trunk(x)
            return {"heatmap": self.heatmap(features), "offset": self.offset(features)}

    return CapNet()


def encode_targets(
    caps: list[tuple[float, float, int]],
    num_classes: int,
    out_shape: tuple[int, int],
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build training targets for one frame.

    Args:
        caps: ``(u, v, class_index)`` per cap, in full-resolution pixels.
        num_classes: Heatmap channels.
        out_shape: ``(rows, cols)`` of the network output.
        sigma: Gaussian spread of each peak, in output cells.

    Returns:
        ``(heatmap, offset, mask)``: a ``(C, rows, cols)`` Gaussian heatmap, a
        ``(2, rows, cols)`` sub-cell offset and a ``(rows, cols)`` mask marking
        the cells that carry an offset target.
    """
    rows, cols = out_shape
    heatmap = np.zeros((num_classes, rows, cols), dtype=np.float32)
    offset = np.zeros((2, rows, cols), dtype=np.float32)
    mask = np.zeros((rows, cols), dtype=np.float32)
    ys, xs = np.mgrid[0:rows, 0:cols]
    for u, v, class_index in caps:
        cx, cy = u / OUTPUT_STRIDE, v / OUTPUT_STRIDE
        ix, iy = math.floor(cx), math.floor(cy)
        if not (0 <= ix < cols and 0 <= iy < rows):
            continue
        gaussian = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2.0 * sigma**2))
        heatmap[class_index] = np.maximum(heatmap[class_index], gaussian)
        offset[:, iy, ix] = (cx - ix, cy - iy)
        mask[iy, ix] = 1.0
    return heatmap, offset, mask


def decode_peaks(
    heatmap: np.ndarray, offset: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> list[tuple[float, float, int, float]]:
    """Find cap centres in a network output.

    Args:
        heatmap: ``(C, rows, cols)`` probabilities.
        offset: ``(2, rows, cols)`` sub-cell offsets.
        threshold: Minimum peak probability.

    Returns:
        ``(u, v, class_index, score)`` per peak, in full-resolution pixels.
        Where two classes peak in the same cell only the stronger is kept.
    """
    best = heatmap.max(axis=0)
    winner = heatmap.argmax(axis=0)
    # A cell is a peak if it is the maximum of its 3x3 neighbourhood.
    dilated = cv2.dilate(best, np.ones((3, 3), np.uint8))
    peaks = np.argwhere((best >= dilated) & (best >= threshold))
    found: list[tuple[float, float, int, float]] = []
    for iy, ix in peaks:
        u = (ix + float(offset[0, iy, ix])) * OUTPUT_STRIDE
        v = (iy + float(offset[1, iy, ix])) * OUTPUT_STRIDE
        found.append((u, v, int(winner[iy, ix]), float(best[iy, ix])))
    return found


@dataclass
class LearnedDetector:
    """Runs a trained :func:`build_model` network on overhead frames.

    Attributes:
        config: The configuration bundle (for class names and tube geometry).
        model: The trained network, in eval mode.
        calibration: Maps cap pixels onto the table; ``None`` leaves ``table_xy`` NaN.
        threshold: Minimum heatmap peak to report.
    """

    config: SampleSortConfig
    model: Any
    calibration: Calibration | None = None
    threshold: float = DEFAULT_THRESHOLD

    @classmethod
    def load(
        cls, config: SampleSortConfig, checkpoint: Path, calibration: Calibration | None = None
    ) -> LearnedDetector:
        """Load a checkpoint written by :func:`save_checkpoint`.

        Raises:
            ValueError: If the checkpoint was trained on different classes.
        """
        from samplesort.learning._deps import require_torch

        torch = require_torch()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        labels = list(state["labels"])
        if labels != config.classes.labels:
            raise ValueError(
                f"checkpoint classes {labels} do not match config {config.classes.labels}"
            )
        model = build_model(len(labels))
        model.load_state_dict(state["model"])
        model.eval()
        return cls(config=config, model=model, calibration=calibration)

    def predict(self, frame: np.ndarray) -> list[tuple[float, float, int, float]]:
        """Raw ``(u, v, class_index, score)`` peaks for one frame."""
        from samplesort.learning._deps import require_torch

        torch = require_torch()
        with torch.no_grad():
            batch = torch.from_numpy(prepare_input(frame))[None]
            out = self.model(batch)
            heatmap = torch.sigmoid(out["heatmap"])[0].numpy()
            offset = out["offset"][0].numpy()
        return decode_peaks(heatmap, offset, self.threshold)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect upright caps, highest score first."""
        labels = self.config.classes.labels
        cap_top = self.config.workspace.tube.total_height
        detections: list[Detection] = []
        for u, v, index, score in self.predict(frame):
            table_xy = (float("nan"), float("nan"))
            if self.calibration is not None:
                table_xy = self.calibration.pixel_to_table(u, v, object_height=cap_top)
            detections.append(
                Detection(
                    pixel_xy=(u, v), table_xy=table_xy, class_label=labels[index], confidence=score
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


def save_checkpoint(
    model: Any, labels: list[str], path: Path, meta: dict[str, Any] | None = None
) -> Path:
    """Save a trained model with the class order it was trained on.

    Args:
        model: The trained network.
        labels: Class labels, in heatmap-channel order.
        path: Destination file.
        meta: How the model was trained (frame count, time, loss history), kept
            so a study that reloads the checkpoint can still report it.
    """
    from samplesort.learning._deps import require_torch

    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "labels": list(labels), "meta": meta or {}}, path)
    return path


def checkpoint_meta(path: Path) -> dict[str, Any]:
    """The training metadata saved with a checkpoint (empty for older ones)."""
    from samplesort.learning._deps import require_torch

    torch = require_torch()
    state = torch.load(path, map_location="cpu", weights_only=True)
    return dict(state.get("meta", {}))
