"""Generate the printed QR labels that sit on simulated tube caps.

Each label is a flat square mesh with the QR code as its texture. Two details
matter for it to be readable in PyBullet's software renderer, both found by
rendering and decoding rather than assumed:

* The mesh carries both face windings. A single-sided quad is back-face culled
  from some viewpoints and simply disappears.
* The label must sit at least ~1 mm proud of the cap. Closer than that, it
  depth-fights with the cap's top face and the cap wins, so callers place it
  with :data:`LABEL_LIFT_M` of clearance — the thickness of a printed sticker.

The label's size is ``workspace.tube.label_side``; see ``workspace.yaml`` for
why it sits where it does between the two cameras' limits.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Height of the label above the cap's top face. Below ~1 mm the label loses the
#: depth test to the cap in ER_TINY_RENDERER and does not appear at all.
LABEL_LIFT_M = 0.0012

#: Pixels per QR module in the texture. Nearest-neighbour upscaling keeps the
#: module edges sharp when the renderer samples the texture.
_TEXTURE_SCALE = 12

_QUAD_OBJ = """\
# Unit square in the XY plane, UVs spanning the full texture.
# Both windings are present so the label is never back-face culled.
v -0.5 -0.5 0
v 0.5 -0.5 0
v 0.5 0.5 0
v -0.5 0.5 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
f 1/1/1 3/3/1 2/2/1
f 1/1/1 4/4/1 3/3/1
"""


def qr_image(text: str) -> np.ndarray:
    """Encode ``text`` as a black-on-white QR code, upscaled for use as a texture.

    Args:
        text: The payload, typically a sample ID.

    Returns:
        A ``uint8`` BGR image whose sides are a multiple of the module count.

    Raises:
        ValueError: If ``text`` is empty or OpenCV could not encode it.
    """
    if not text:
        raise ValueError("a QR label needs a non-empty payload")
    # Highest error correction: a real printed label gets scuffed, and a short
    # ID still fits in the smallest (21-module) code at this level.
    params = cv2.QRCodeEncoder.Params()
    params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_H
    encoded = cv2.QRCodeEncoder.create(params).encode(text)
    if encoded is None or encoded.size == 0:
        raise ValueError(f"could not encode {text!r} as a QR code")
    modules = np.asarray(encoded, dtype=np.uint8)
    size = (modules.shape[1] * _TEXTURE_SCALE, modules.shape[0] * _TEXTURE_SCALE)
    upscaled = cv2.resize(modules, size, interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR))


def label_dir() -> Path:
    """Directory the generated label assets are written to.

    Labels are per-sample and regenerated on every spawn, so they live in the
    temp directory rather than the package's asset folder.
    """
    path = Path(tempfile.gettempdir()) / "samplesort" / "labels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_label_assets(sample_id: str, directory: Path | None = None) -> tuple[Path, Path]:
    """Write the quad mesh and the QR texture for one sample ID.

    Args:
        sample_id: The ID to encode.
        directory: Where to write. Defaults to :func:`label_dir`.

    Returns:
        ``(mesh_path, texture_path)``.
    """
    target = directory if directory is not None else label_dir()
    target.mkdir(parents=True, exist_ok=True)

    mesh_path = target / "label_quad.obj"
    if not mesh_path.exists() or mesh_path.read_text(encoding="utf-8") != _QUAD_OBJ:
        mesh_path.write_text(_QUAD_OBJ, encoding="utf-8")

    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id)
    texture_path = target / f"label_{safe_name}.png"
    if not cv2.imwrite(str(texture_path), qr_image(sample_id)):
        raise OSError(f"could not write QR texture to {texture_path}")
    logger.debug("wrote QR label for %s to %s", sample_id, texture_path)
    return mesh_path, texture_path
