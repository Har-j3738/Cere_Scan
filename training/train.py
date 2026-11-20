"""Train the CereScan four-class classifier with reproducible, class-balanced training."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from neuro_scan.model import CLASS_NAMES, build_model, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "splits.csv"
MODEL_PATH = ROOT / "models" / "cerescan_resnet18.pth"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CereScanDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(ROOT / row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def build_transforms(train: bool):
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(8)]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    weights = len(frame) / (len(CLASS_NAMES) * counts.replace(0, np.nan))
    weights = weights.fillna(0.0)
    return torch.tensor(weights.values, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, optimizer, device, training=True):
    model.train(training)
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit("Run: python training/prepare_dataset.py")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    frame = pd.read_csv(CSV_PATH)
    train_df = frame[frame.split == "train"].copy()
    val_df = frame[frame.split == "val"].copy()

    train_loader = DataLoader(
        CereScanDataset(train_df, build_transforms(True)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CereScanDataset(val_df, build_transforms(False)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(pretrained=True).to(device)
    weights = class_weights(train_df).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        if val_acc > best:
            best = val_acc
            save_checkpoint(model, MODEL_PATH)
            print(f"Saved best model -> {MODEL_PATH}")
    print(f"Best validation accuracy: {best:.3f}")


if __name__ == "__main__":
    main()
