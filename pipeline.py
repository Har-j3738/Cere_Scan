"""Command-line orchestration for the CereScan ML workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    """Run a project script with the current Python interpreter."""
    command = [sys.executable, str(ROOT / script), *args]
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CereScan pipeline stages.")
    parser.add_argument("--prepare", action="store_true", help="Validate and index the dataset.")
    parser.add_argument("--train", action="store_true", help="Train the ResNet18 classifier.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the held-out test split.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not any((args.prepare, args.train, args.evaluate)):
        parser.error("Choose at least one stage: --prepare, --train, or --evaluate.")

    if args.prepare:
        run("training/prepare_dataset.py")

    if args.train:
        run(
            "training/train.py",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        )

    if args.evaluate:
        run("training/evaluate.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
