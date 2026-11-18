"""Dataset utilities for the Jun Cheng brain tumor dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LABELS = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}


@dataclass(frozen=True)
class Record:
    path: Path
    label: int
    patient_id: str


def _unwrap(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _field(struct: Any, name: str) -> Any:
    struct = _unwrap(struct)
    if isinstance(struct, dict):
        return _unwrap(struct[name])
    if hasattr(struct, name):
        return _unwrap(getattr(struct, name))
    if isinstance(struct, np.void) and struct.dtype.names and name in struct.dtype.names:
        return _unwrap(struct[name])
    raise KeyError(f"MATLAB field '{name}' not found")


def load_mat_record(path: Path) -> tuple[np.ndarray, int, str]:
    try:
        import hdf5storage
        data = hdf5storage.loadmat(str(path), simplify_cells=True)
    except Exception as hdf_error:
        try:
            from scipy.io import loadmat
            data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        except Exception as scipy_error:
            raise RuntimeError(
                f"Could not read {path}. hdf5storage error: {hdf_error}; "
                f"scipy error: {scipy_error}"
            ) from scipy_error

    if "cjdata" not in data:
        raise KeyError(f"{path} does not contain expected 'cjdata' field")

    cjdata = data["cjdata"]
    image = np.asarray(_field(cjdata, "image"), dtype=np.float32)
    label = int(np.asarray(_field(cjdata, "label")).reshape(-1)[0])
    patient_id = str(np.asarray(_field(cjdata, "PID")).reshape(-1)[0])
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D MRI image in {path}, got {image.shape}")
    if label not in LABELS:
        raise ValueError(f"Unexpected label {label} in {path}")

    return image, label, patient_id


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


def discover_mat_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found under {root}")
    return files


def inspect_dataset(root: Path) -> list[Record]:
    records = []
    for index, path in enumerate(discover_mat_files(root), start=1):
        _, label, patient_id = load_mat_record(path)
        records.append(Record(path, label, patient_id))
        if index % 250 == 0:
            print(f"Audited {index} files...")
    return records
