"""Prepare the Jun Cheng Figshare dataset for CereScan training."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from neuro_scan.dataset import LABELS, discover_mat_files, load_mat_record, to_rgb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MAT_DIR = ROOT / "data" / "mat"
PROCESSED = ROOT / "data" / "processed"
IMAGE_DIR = PROCESSED / "images"


def extract_archives():
    archives = sorted(RAW.glob("*.zip"))
    if len(archives) != 4:
        raise SystemExit(f"Expected 4 dataset archives in {RAW}; found {len(archives)}.")
    MAT_DIR.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(MAT_DIR)
    print(f"Extracted {len(discover_mat_files(MAT_DIR))} MATLAB files.")


def build_metadata():
    rows = []
    files = discover_mat_files(MAT_DIR)
    for i, path in enumerate(files, 1):
        image, label, pid = load_mat_record(path)
        rows.append({
            "mat_path": str(path.relative_to(ROOT)),
            "label": label,
            "class_name": LABELS[label],
            "patient_id": pid,
            "height": image.shape[0],
            "width": image.shape[1],
        })
        if i % 250 == 0:
            print(f"Read {i}/{len(files)} files")
    frame = pd.DataFrame(rows).sort_values(["patient_id", "label", "mat_path"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "metadata.csv", index=False)
    print("\nClass counts:")
    print(frame["class_name"].value_counts().to_string())
    print(f"Total images: {len(frame)}")
    print(f"Unique patients: {frame.patient_id.nunique()}")
    return frame


def create_splits(frame):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    _, test_idx = next(splitter.split(np.zeros(len(frame)), frame.label, frame.patient_id))

    remaining = frame.drop(frame.index[test_idx]).copy()
    _, val_idx = next(
        splitter.split(np.zeros(len(remaining)), remaining.label, remaining.patient_id)
    )

    test_patients = set(frame.iloc[test_idx].patient_id)
    val_patients = set(remaining.iloc[val_idx].patient_id)

    result = frame.copy()
    result["split"] = "train"
    result.loc[result.patient_id.isin(test_patients), "split"] = "test"
    result.loc[result.patient_id.isin(val_patients), "split"] = "val"

    sets = {s: set(result[result.split == s].patient_id) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])

    result.to_csv(PROCESSED / "splits.csv", index=False)
    print("\nImage counts by split/class:")
    print(result.groupby(["split", "class_name"]).size().unstack(fill_value=0))
    print("\nPatient counts:")
    print(result.groupby("split").patient_id.nunique())
    return result


def convert_images(frame):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for i, row in enumerate(frame.itertuples(index=False), 1):
        destination = IMAGE_DIR / f"{i:04d}_{row.label}.png"
        image_paths.append(str(destination.relative_to(ROOT)))
        if not destination.exists():
            image, _, _ = load_mat_record(ROOT / row.mat_path)
            to_rgb(image).save(destination, format="PNG")
        if i % 250 == 0:
            print(f"Converted {i}/{len(frame)} images")

    result = frame.copy()
    result["image_path"] = image_paths
    result.to_csv(PROCESSED / "splits.csv", index=False)


def main():
    if not list(RAW.glob("*.zip")):
        raise SystemExit("Dataset archives not found. Run: python scripts/download_dataset.py")
    if not list(MAT_DIR.rglob("*.mat")):
        extract_archives()
    metadata = build_metadata()
    splits = create_splits(metadata)
    convert_images(splits)
    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
