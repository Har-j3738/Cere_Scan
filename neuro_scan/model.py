"""CereScan transfer-learning model."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


def build_model(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    if num_classes != len(CLASS_NAMES):
        raise ValueError(f"CereScan expects {len(CLASS_NAMES)} classes, got {num_classes}.")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def save_checkpoint(model: nn.Module, path, class_names=CLASS_NAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
        "architecture": "resnet18",
    }, path)


def load_checkpoint(path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    if list(class_names) != CLASS_NAMES:
        raise ValueError(f"Checkpoint classes {class_names} do not match current classes {CLASS_NAMES}.")
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
