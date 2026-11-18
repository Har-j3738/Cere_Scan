"""Evaluate CereScan on the held-out patient-disjoint test set."""

from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from neuro_scan.model import load_checkpoint
from training.train import CereScanDataset, build_transforms

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "splits.csv"
MODEL_PATH = ROOT / "models" / "cerescan_resnet18.pth"


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

    names = ["Meningioma", "Glioma", "Pituitary"]
    print("\nCereScan test results")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))
    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
