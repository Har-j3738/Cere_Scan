"""Inspect and prepare the Figshare MATLAB dataset.

This first version is intentionally an audit tool. It does not train a model.
It reports archive contents and gives us a safe place to add patient-level
splitting logic after the raw files are downloaded.
"""

from pathlib import Path
import zipfile

RAW_DIR = Path("data/raw")


def inspect_archives() -> None:
    archives = sorted(RAW_DIR.glob("*.zip"))

    if not archives:
        raise SystemExit(
            "No dataset archives found. Run: python scripts/download_dataset.py"
        )

    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            files = zf.namelist()
            mat_files = [name for name in files if name.lower().endswith(".mat")]

            print(f"\n{archive.name}")
            print(f"  files: {len(files)}")
            print(f"  .mat files: {len(mat_files)}")

            if mat_files:
                print(f"  first file: {mat_files[0]}")


if __name__ == "__main__":
    inspect_archives()
