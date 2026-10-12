from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def preprocess_sketch(image: Image.Image, size: int = 28) -> np.ndarray:
    """Convert a canvas image to a centered 28 x 28 white-background bitmap.

    The drawing canvas is expected to have a light background and dark strokes,
    matching the QuickDraw bitmap representation. The result is float32 in [0, 1].
    """
    if size < 8:
        raise ValueError("size must be at least 8 pixels")

    gray = ImageOps.grayscale(image.convert("RGBA").convert("RGB"))
    arr = np.asarray(gray, dtype=np.uint8)
    mask = arr < 245

    if mask.any():
        ys, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        crop = gray.crop((left, top, right, bottom))
        margin = max(2, round(size * 0.12))
        inner = size - 2 * margin
        crop.thumbnail((inner, inner), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (size, size), color=255)
        x = (size - crop.width) // 2
        y = (size - crop.height) // 2
        canvas.paste(crop, (x, y))
    else:
        canvas = Image.new("L", (size, size), color=255)

    out = np.asarray(canvas, dtype=np.float32) / 255.0
    return out[None, :, :]
