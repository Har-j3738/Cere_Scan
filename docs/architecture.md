# CereScan Architecture

## MVP pipeline

1. User uploads an MRI image.
2. Streamlit receives the image.
3. Input is converted to RGB and resized to 224×224.
4. Pixel values are normalized to [0, 1].
5. The inference layer returns class probabilities.
6. The UI displays the prediction and probabilities.

## Important boundary

The current inference function is a deterministic placeholder. The next implementation milestone replaces it with a trained PyTorch classifier and a reproducible evaluation workflow.

## Model milestone

```
Dataset
  ↓
Patient-level train/validation/test split
  ↓
Preprocessing + augmentation
  ↓
CNN / transfer-learning model
  ↓
Training
  ↓
Validation
  ↓
Held-out test evaluation
  ↓
Saved model
  ↓
CereScan inference API
```
