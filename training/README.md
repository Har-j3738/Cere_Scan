# Training

CereScan uses a four-class transfer-learning pipeline.

```text
BDNeuro-MRI
      ↓
Published train / validation / test folders
      ↓
Dataset validation + indexing
      ↓
Resize + augmentation
      ↓
ResNet18 transfer learning
      ↓
Class-weighted CrossEntropyLoss
      ↓
AdamW
      ↓
Best validation checkpoint
      ↓
Held-out test evaluation
      ↓
Accuracy / precision / recall / F1 / confusion matrix
```

## Target classes

1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

## Commands

```bash
python scripts/download_dataset.py
python training/prepare_dataset.py
python training/train.py --epochs 12 --batch-size 16
python training/evaluate.py
```

The exact model performance is intentionally not hard-coded in this repository. Results must come from an actual training/evaluation run.
