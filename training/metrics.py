"""Metrics and confusion-matrix utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, class_names, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title("CereScan Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
