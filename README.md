# CereScan — AI Brain MRI Classification

CereScan is a student research/educational prototype for **four-class brain MRI classification**.

## Required classes

1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

## Current architecture

```text
MRI upload
   ↓
Input validation + preprocessing
   ↓
ResNet18 transfer learning
   ↓
4-output Softmax
   ├── Glioma
   ├── Meningioma
   ├── Pituitary
   └── No Tumor
   ↓
Prediction + model probability
   ↓
Grad-CAM explanation
```

## Dataset

CereScan uses **BDNeuro-MRI**, a four-class dataset with 5,941 pre-processed T1-weighted contrast-enhanced MRI images.

Published split:

- Train: 4,160
- Validation: 892
- Test: 889

The published dataset reports exact/near-duplicate removal and zero image-content overlap across the supplied splits.

Dataset page:
https://data.mendeley.com/datasets/zwr4ntf94j

## Repository

```text
Cere_Scan/
├── app.py
├── pipeline.py
├── neuro_scan/
│   ├── dataset.py
│   ├── inference.py
│   ├── model.py
│   └── visualization.py
├── training/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
├── scripts/
│   └── download_dataset.py
├── tests/
└── docs/
```

## Linux setup

Use **Python 3.12** for the project environment because the CI pipeline is also tested on Python 3.12.

```bash
cd ~/Projects/Cere_Scan

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --default-timeout=300 --retries=10 -r requirements.txt
```

If the repository is already cloned:

```bash
cd ~/Projects/Cere_Scan
git pull origin main
```

## Download dataset

```bash
python scripts/download_dataset.py
```

The archive is kept outside Git.

## Prepare / audit dataset

```bash
python training/prepare_dataset.py
```

This validates the four-class folder structure and creates:

```text
data/processed/splits.csv
data/processed/class_distribution.csv
```

The published train/validation/test split is preserved; the script does not randomly reshuffle images.

## Train

```bash
python training/train.py --epochs 12 --batch-size 16
```

CUDA is used automatically when available. The best validation checkpoint is saved as:

```text
models/cerescan_resnet18.pth
```

Model artifacts are ignored by Git.

## Evaluate

```bash
python training/evaluate.py
```

This evaluates the held-out test split and produces:

- accuracy
- macro precision
- macro recall
- macro F1
- classification report
- confusion matrix

Reports are written to `reports/`.

**Do not put accuracy numbers in documentation until the model has actually been trained and evaluated.**

## Run CereScan

```bash
streamlit run app.py
```

Upload an MRI image and click **Analyze MRI**. The application performs real PyTorch inference and displays:

- predicted class
- model probability
- all four class probabilities
- Grad-CAM visualization

## Mathematics used

The project uses:

- image normalization
- convolutional feature extraction
- ReLU activation
- linear transformation `z = Wx + b`
- four-class Softmax
- cross-entropy loss
- backpropagation
- AdamW optimization
- class-weighted loss
- accuracy, precision, recall and F1
- confusion matrix
- Grad-CAM

## Important scientific boundary

CereScan is a **classification** system. It does not calculate tumor volume, tumor diameter, growth/regression, treatment response, or a validated segmentation boundary.

Grad-CAM is an explanation/localization aid, not a clinical tumor segmentation mask.

Softmax probability is a model output, **not clinical certainty**.

CereScan is **not a medical device** and must not be used to diagnose, treat, or make clinical decisions.

## Status

- [x] Four required classes
- [x] Four-class dataset integration
- [x] Dataset validation/indexing
- [x] ResNet18 transfer-learning model
- [x] Class-weighted training
- [x] Held-out test evaluation
- [x] Confusion matrix
- [x] Real Streamlit inference
- [x] Grad-CAM
- [x] GitHub Actions tests
- [ ] Train final model
- [ ] Record final experimental metrics
- [ ] Add model/version metadata
- [ ] Optional segmentation research
