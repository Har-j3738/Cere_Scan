"""Visualization helpers for CereScan."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))).astype(np.float32)
    h = heatmap.astype(np.float32) / 255.0
    color = np.zeros((*h.shape, 3), dtype=np.float32)
    color[..., 0] = h
    color[..., 1] = np.maximum(0.0, 1.0 - np.abs(h * 2.0 - 1.0))
    color[..., 2] = 1.0 - h
    blended = base * (1.0 - alpha) + color * 255.0 * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))
