"""Inference and Grad-CAM utilities for CereScan."""

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


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(model, image: Image.Image, device: torch.device):
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    return CLASS_NAMES[index], float(probabilities[index]), probabilities.cpu().numpy()


def load_model(model_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_checkpoint(model_path, device), device


def grad_cam(model, image: Image.Image, device: torch.device, class_index: int | None = None):
    """Return an RGB Grad-CAM heatmap for the selected class."""
    activations = []
    gradients = []

    target_layer = model.layer4[-1]

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    f_hook = target_layer.register_forward_hook(forward_hook)
    b_hook = target_layer.register_full_backward_hook(backward_hook)

    tensor = preprocess(image).to(device)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)

    if class_index is None:
        class_index = int(logits.argmax(1).item())

    logits[0, class_index].backward()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max().clamp_min(1e-8)
    cam = cam.detach().cpu().numpy()

    f_hook.remove()
    b_hook.remove()

    cam_image = Image.fromarray(np.uint8(cam * 255)).resize(image.convert("RGB").size)
    return np.asarray(cam_image, dtype=np.uint8)
