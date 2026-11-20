"""Dataset acquisition and validation helper for CereScan.

BDNeuro-MRI is distributed through Mendeley Data. The dataset archive is
intentionally not downloaded into the Git repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATASET_URL = "https://data.mendeley.com/datasets/zwr4ntf94j"
EXPECTED_ROOT = Path("data/raw/BDNeuro-MRI")
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")


def check_dataset(root: Path = EXPECTED_ROOT) -> bool:
    """Validate the expected local image-folder structure."""
    missing = []

    if not root.exists():
        missing.append(str(root))
    else:
        for split in EXPECTED_SPLITS:
            for class_name in EXPECTED_CLASSES:
                directory = root / split / class_name
                if not directory.is_dir():
                    missing.append(str(directory))
                elif not any(directory.iterdir()):
                    missing.append(f"{directory} (empty)")

    if missing:
        print("Dataset check: FAILED")
        print("Missing or empty paths:")
        for path in missing:
            print(f"  - {path}")
        return False

    print(f"Dataset check: OK -> {root}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local BDNeuro-MRI dataset.")
    parser.add_argument("--root", type=Path, default=EXPECTED_ROOT)
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Print the official dataset page for manual download.",
    )
    args = parser.parse_args()

    if args.show_source:
        print(f"Official dataset page: {DATASET_URL}")
        print(f"Expected local directory: {args.root}")

    return 0 if check_dataset(args.root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
