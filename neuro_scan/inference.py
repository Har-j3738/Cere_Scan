"""Model loading, inference, and Grad-CAM utilities for CereScan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from neuro_scan.model import CLASS_NAMES, load_checkpoint

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def preprocess(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into the model's normalized batch format."""
    return INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0)


def load_model(model_path: Path):
    """Load a trained CereScan checkpoint and select the available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_checkpoint(model_path, device), device


@torch.inference_mode()
def predict(model, image: Image.Image, device: torch.device):
    """Return predicted class, confidence, and all class probabilities."""
    tensor = preprocess(image).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax().item())
    return (
        CLASS_NAMES[index],
        float(probabilities[index].item()),
        probabilities.cpu().numpy(),
    )


def grad_cam(
    model,
    image: Image.Image,
    device: torch.device,
    class_index: int | None = None,
) -> np.ndarray:
    """Return a normalized Grad-CAM activation map in the range [0, 1].

    The map is an explanation aid for the model's selected class. It is not a
    validated tumor segmentation mask.
    """
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = model.layer4[-1].conv2

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor = preprocess(image).to(device)
        model.eval()
        model.zero_grad(set_to_none=True)

        logits = model(tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        logits[0, class_index].backward()

        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1).squeeze(0))
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)

        return cam.detach().cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()
