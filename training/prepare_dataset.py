"""Validate and index the four-class BDNeuro-MRI dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuro_scan.dataset import CLASS_NAMES, build_records, validate_class_tree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET_ROOT = RAW / "BDNeuro-MRI"
PROCESSED = ROOT / "data" / "processed"


def find_dataset_root() -> Path:
    if (DATASET_ROOT / "train").is_dir():
        return DATASET_ROOT

    candidates = [p for p in RAW.rglob("train") if p.is_dir()]
    valid = [
        p.parent for p in candidates
        if all((p.parent / "test" / c.lower().replace(" ", "_")).is_dir() for c in CLASS_NAMES)
    ]
    if len(valid) == 1:
        return valid[0]
    raise SystemExit(
        "Could not locate the BDNeuro-MRI train/val/test folder tree. "
        "Extract the downloaded dataset under data/raw/BDNeuro-MRI/."
    )


def main():
    dataset_root = find_dataset_root()
    counts = validate_class_tree(dataset_root)
    print(f"Dataset root: {dataset_root}")
    print("\nClass/split counts:")
    for split in ("train", "val", "test"):
        total = 0
        for class_name in CLASS_NAMES:
            count = counts[f"{split}/{class_name}"]
            total += count
            print(f"  {split:5s} {class_name:12s}: {count}")
        print(f"  {split:5s} {'TOTAL':12s}: {total}")

    records = build_records(dataset_root)
    frame = pd.DataFrame([
        {
            "image_path": str(r.path.relative_to(ROOT)),
            "label": r.label,
            "class_name": CLASS_NAMES[r.label],
            "split": r.split,
        }
        for r in records
    ])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "splits.csv", index=False)
    summary = frame.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    summary.to_csv(PROCESSED / "class_distribution.csv")
    print(f"\nIndexed {len(frame)} images.")
    print(summary.to_string())
    print(f"\nWrote: {PROCESSED / 'splits.csv'}")


if __name__ == "__main__":
    main()
