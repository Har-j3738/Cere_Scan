# CereScan

**CereScan** is an educational and research-oriented AI prototype for brain MRI image classification and explainable analysis.

## Current MVP

The first milestone establishes the complete inference pipeline:

```
MRI Image
   ↓
Image Validation
   ↓
Preprocessing
   ↓
Deep Learning Classifier
   ↓
Class Probabilities
   ↓
Prediction + Explanation
```

The current repository contains a working Streamlit interface and a clearly marked demo inference layer. Demo probabilities are placeholders and must not be interpreted as real medical predictions.

## Planned development

- [x] Project repository and documentation
- [x] Working Streamlit prototype
- [x] MRI preprocessing pipeline
- [ ] Real PyTorch training pipeline
- [ ] Patient-independent dataset split
- [ ] Model evaluation: accuracy, precision, recall/sensitivity, specificity, F1
- [ ] Confusion matrix and evaluation dashboard
- [ ] Explainable AI with Grad-CAM
- [ ] Model versioning and reproducible experiments
- [ ] Optional future tumor segmentation
- [ ] Optional future tumor-volume estimation

## Tech stack

- Python
- Streamlit
- NumPy
- Pillow
- PyTorch (model phase)
- scikit-learn (evaluation phase)

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
Cere_Scan/
├── app.py
├── pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
├── training/
├── neuro_scan/
├── tests/
└── docs/
```

## Medical safety

CereScan is a student research/educational prototype. It is **not a medical device and must not be used to diagnose, treat, or make clinical decisions about a patient**. Model confidence is not equivalent to clinical certainty.

## Goal

Build a transparent, reproducible student project that demonstrates the complete machine-learning workflow rather than claiming clinical performance without appropriate validation.
