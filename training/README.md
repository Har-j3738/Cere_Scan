# Training

This directory contains the CereScan model-training pipeline.

## Planned flow

```text
Official Figshare dataset
        ↓
MATLAB files
        ↓
Dataset audit
        ↓
Patient-disjoint split
        ↓
Image preprocessing
        ↓
Transfer-learning classifier
        ↓
Validation
        ↓
Held-out test set
        ↓
Metrics + confusion matrix
        ↓
Saved model
```

## First model

The first baseline will be a small transfer-learning image classifier rather than a neural network trained from scratch.

Target classes:

- meningioma
- glioma
- pituitary

The model must be evaluated on data separated by patient where the dataset metadata permits it.

Training code should never contain the dataset itself.
