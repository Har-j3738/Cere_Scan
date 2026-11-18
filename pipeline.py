"""Core image preprocessing and temporary demo inference pipeline."""

from PIL import Image
import numpy as np


IMAGE_SIZE = (224, 224)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an uploaded image to the model-ready demo format."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def demo_predict(image_array: np.ndarray):
    """Return placeholder probabilities for UI development.

    This function is intentionally deterministic and is NOT a trained model.
    Replace it with real PyTorch inference in the model milestone.
    """
    predictions = {
        "Glioma": 0.78,
        "Meningioma": 0.16,
        "Pituitary": 0.06,
    }

    predicted_class = max(predictions, key=predictions.get)
    confidence = predictions[predicted_class]

    return predicted_class, confidence, predictions
