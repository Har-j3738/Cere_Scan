"""Evaluate CereScan on the held-out patient-disjoint test set."""

from pathlib import Path
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from neuro_scan.model import load_checkpoint
from training.metrics import classification_metrics, save_confusion_matrix
from training.train import CereScanDataset, build_transforms

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "splits.csv"
MODEL_PATH = ROOT / "models" / "cerescan_resnet18.pth"
RESULTS = ROOT / "reports"
NAMES = ["Meningioma", "Glioma", "Pituitary"]


def main():
    if not CSV_PATH.exists() or not MODEL_PATH.exists():
        raise SystemExit("Prepare the dataset and train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(CSV_PATH)
    test_df = frame[frame.split == "test"]

    loader = DataLoader(
        CereScanDataset(test_df, build_transforms(False)),
        batch_size=32, shuffle=False, num_workers=0,
    )
    model = load_checkpoint(MODEL_PATH, device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    metrics = classification_metrics(y_true, y_pred)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(y_true, y_pred, NAMES, RESULTS / "confusion_matrix.png")

    print("\nCereScan test results")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=NAMES, zero_division=0))
    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nSaved reports to: {RESULTS}")


if __name__ == "__main__":
    main()
