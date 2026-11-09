"""Assemble the frames from ``record_gif.mjs`` into an optimised GIF.

Usage: ``python make_gif.py <frames_dir> <out.gif> [width] [fps]``
"""

import sys
from pathlib import Path

from PIL import Image

frames_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
width = int(sys.argv[3]) if len(sys.argv) > 3 else 720
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0

images = []
for path in sorted(frames_dir.glob("f*.png")):
    image = Image.open(path).convert("RGB")
    height = round(image.height * width / image.width)
    images.append(image.resize((width, height), Image.LANCZOS))
# One shared palette keeps colours stable from frame to frame. It is built from
# a strip of frames across the run so that small, saturated caps get entries.
sample = images[:: max(1, len(images) // 8)]
strip = Image.new("RGB", (images[0].width, images[0].height * len(sample)))
for i, frame in enumerate(sample):
    strip.paste(frame, (0, i * frame.height))
palette = strip.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
frames[0].save(
    out, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=True
)
print(f"{out}: {len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB")
