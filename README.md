# CereScan — AI Brain MRI Classification

CereScan is a student research/educational prototype for three-class brain MRI tumor classification.

## Current architecture

```text
MRI upload
   ↓
Preprocessing
   ↓
ResNet18 transfer learning
   ↓
Class probabilities
   ↓
Prediction
   ↓
Grad-CAM explanation
```

Classes: Glioma, Meningioma, Pituitary tumor.

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
│   └── evaluate.py
├── scripts/
│   └── download_dataset.py
├── tests/
└── docs/
```

## Linux setup

```bash
cd ~/Projects

git clone https://github.com/Har-j3738/Cere_Scan.git
cd Cere_Scan

python3 -m venv .venv
source .venv/bin/activate
pip install --default-timeout=300 --retries=10 -r requirements.txt
```

If already cloned:

```bash
cd ~/Projects/Cere_Scan
git pull origin main
```

## Download dataset

```bash
python scripts/download_dataset.py
```

## Prepare dataset

```bash
python training/prepare_dataset.py
```

This extracts the MATLAB archives, reads image/label/patient metadata, creates patient-disjoint train/validation/test splits, and converts MRI images into model-ready RGB PNGs.

## Train

```bash
python training/train.py --epochs 8 --batch-size 16
```

CUDA is used automatically when available. The local checkpoint is saved as `models/cerescan_resnet18.pth` and is ignored by Git.

## Evaluate

```bash
python training/evaluate.py
```

Reports accuracy, macro precision, macro recall, macro F1, classification report, and confusion matrix.

## Run CereScan

```bash
streamlit run app.py
```

Upload an MRI and click **Analyze MRI**. The application performs real PyTorch inference and displays the prediction, class probabilities, and Grad-CAM visualization.

## Interpretation and safety

Softmax probability is the neural network output, not clinical certainty. Grad-CAM is an explanation/localization aid, not a validated tumor segmentation mask.

CereScan is **not a medical device** and must not be used to diagnose, treat, or make clinical decisions.

## Status

- [x] GitHub project
- [x] Dataset download tooling
- [x] MATLAB parser
- [x] Patient-disjoint splitting
- [x] ResNet18 model
- [x] Training pipeline
- [x] Test evaluation
- [x] Real Streamlit inference
- [x] Grad-CAM
- [x] GitHub Actions tests
- [ ] Confidence calibration
- [ ] Confusion-matrix UI
- [ ] Experiment/model version tracking
- [ ] Optional segmentation research
