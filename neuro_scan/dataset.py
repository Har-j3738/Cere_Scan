"""Dataset utilities for the four-class CereScan MRI dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
LABELS = {index: name for index, name in enumerate(CLASS_NAMES)}
NAME_TO_LABEL = {name: index for index, name in LABELS.items()}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Record:
    path: Path
    label: int
    split: str


def normalize_mri(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(image, (1, 99))
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def to_rgb(image: np.ndarray) -> Image.Image:
    return Image.fromarray(normalize_mri(image), mode="L").convert("RGB")


def discover_images(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No MRI image files found under {root}")
    return files


def validate_class_tree(dataset_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name.lower().replace(" ", "_")
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            count = sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if count == 0:
                raise ValueError(f"No images found in {class_dir}")
            counts[f"{split}/{class_name}"] = count
    return counts


def build_records(dataset_root: Path) -> list[Record]:
    validate_class_tree(dataset_root)
    records: list[Record] = []
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_dir = dataset_root / split / class_name.lower().replace(" ", "_")
            label = NAME_TO_LABEL[class_name]
            for path in discover_images(class_dir):
                records.append(Record(path=path, label=label, split=split))
    return records
