"""U-Net segmentation primitives for datasets with real image/mask pairs.

No masks are synthesized by this module. Pixel measurements are reported in pixels
unless trustworthy spatial metadata is explicitly supplied by the caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PairedSegmentationDataset(Dataset):
    """Load paired raster MRI images and binary masks from a CSV with image,mask columns."""

    def __init__(self, manifest: Path, size: int = 256):
        import pandas as pd

        self.frame = pd.read_csv(manifest)
        if not {"image", "mask"}.issubset(self.frame.columns):
            raise ValueError("Segmentation manifest must contain image and mask columns.")
        if self.frame.empty:
            raise ValueError("Segmentation manifest contains no pairs.")
        self.size = size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path, mask_path = Path(row.image), Path(row["mask"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing segmentation pair: {image_path}, {mask_path}")
        image = Image.open(image_path).convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        x = TF.to_tensor(image)
        y = (TF.to_tensor(mask) > 0).float()
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    """Compact binary U-Net returning one-channel logits."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec1 = DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.enc4(self.pool(c))
        z = self.bottleneck(self.pool(d))
        z = self.dec4(torch.cat((self.up4(z), d), 1))
        z = self.dec3(torch.cat((self.up3(z), c), 1))
        z = self.dec2(torch.cat((self.up2(z), b), 1))
        z = self.dec1(torch.cat((self.up1(z), a), 1))
        return self.out(z)


def dice_coefficient(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    prediction = (prediction > 0.5).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * target).sum(dims)
    return ((2 * intersection + eps) / (prediction.sum(dims) + target.sum(dims) + eps)).mean()


def iou_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    prediction, target = (prediction > 0.5), (target > 0.5)
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & target).float().sum(dims)
    union = (prediction | target).float().sum(dims)
    return ((intersection + eps) / (union + eps)).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    dice_loss = 1 - ((2 * (probs * target).sum(dims) + 1e-7) /
                     (probs.sum(dims) + target.sum(dims) + 1e-7)).mean()
    return bce + dice_loss


def tumor_area(mask: np.ndarray) -> dict[str, float | int]:
    """Return pixel count and percentage of foreground pixels in a 2-D mask."""
    values = np.asarray(mask)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Expected a non-empty 2-D mask.")
    area = int(np.count_nonzero(values))
    return {"area_pixels": area, "area_percent": 100.0 * area / values.size}


def predict_mask(model: nn.Module, image: Image.Image, device: torch.device, size: int = 256,
                 threshold: float = 0.5) -> np.ndarray:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one.")
    gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(gray).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        return (torch.sigmoid(model(tensor))[0, 0].cpu().numpy() >= threshold)
