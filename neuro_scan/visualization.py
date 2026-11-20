"""Visualization helpers for CereScan."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay a normalized or 8-bit activation map on an MRI image.

    Parameters
    ----------
    image:
        Original MRI image.
    heatmap:
        2-D activation map. Values may be in [0, 1] or [0, 255].
    alpha:
        Blend strength for the visualization.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    base = image.convert("RGB")
    heat = np.asarray(heatmap, dtype=np.float32)

    if heat.ndim != 2:
        raise ValueError("heatmap must be a 2-D array.")

    if heat.max(initial=0.0) > 1.0:
        heat /= 255.0

    heat = np.clip(heat, 0.0, 1.0)
    heat_image = Image.fromarray(np.uint8(heat * 255.0), mode="L")
    heat_image = heat_image.resize(base.size)

    # Lightweight blue -> cyan -> yellow -> red style mapping without
    # requiring matplotlib in the inference path.
    h = np.asarray(heat_image, dtype=np.float32) / 255.0
    color = np.empty((*h.shape, 3), dtype=np.float32)
    color[..., 0] = np.clip(1.5 * h - 0.5, 0.0, 1.0)
    color[..., 1] = np.clip(1.5 - np.abs(2.0 * h - 1.0) * 1.5, 0.0, 1.0)
    color[..., 2] = np.clip(1.0 - 1.5 * h, 0.0, 1.0)

    base_array = np.asarray(base, dtype=np.float32)
    blended = base_array * (1.0 - alpha) + color * 255.0 * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0.0, 255.0)))
