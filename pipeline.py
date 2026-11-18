"""Compatibility preprocessing helpers for CereScan."""

from PIL import Image
import numpy as np

IMAGE_SIZE = (224, 224)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an image to a simple normalized RGB array."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0
