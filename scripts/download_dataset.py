"""Download the official Jun Cheng brain tumor dataset from Figshare.

The dataset is kept outside Git because the archives are large.
"""

from pathlib import Path
from urllib.request import Request, urlopen

DATASET_DIR = Path("data/raw")

FILES = [
    (
        "brainTumorDataPublic_1-766.zip",
        "https://figshare.com/ndownloader/files/2706818",
    ),
    (
        "brainTumorDataPublic_767-1532.zip",
        "https://figshare.com/ndownloader/files/2706815",
    ),
    (
        "brainTumorDataPublic_1533-2298.zip",
        "https://figshare.com/ndownloader/files/2706812",
    ),
    (
        "brainTumorDataPublic_2299-3064.zip",
        "https://figshare.com/ndownloader/files/2706809",
    ),
]


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Already exists: {destination}")
        return

    print(f"Downloading {destination.name} ...")

    request = Request(
        url,
        headers={"User-Agent": "CereScan/0.1"},
    )

    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)

            if total_bytes:
                percent = downloaded * 100 / total_bytes
                print(f"  {percent:5.1f}%", end="\r")

    print(f"  Saved: {destination}")


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES:
        download(url, DATASET_DIR / filename)

    print("\nDataset download complete.")
    print("The archives are intentionally ignored by Git.")


if __name__ == "__main__":
    main()
