# Dataset

## Selected dataset: BDNeuro-MRI

CereScan uses **BDNeuro-MRI**, a four-class brain MRI dataset published on Mendeley Data.

Official dataset page:
https://data.mendeley.com/datasets/zwr4ntf94j

Current release used by this project:
- Version 9
- DOI: 10.17632/zwr4ntf94j.9
- License: CC BY 4.0

The dataset contains **5,941 pre-processed T1-weighted contrast-enhanced MRI images** in four classes:

| Class | Total |
|---|---:|
| Glioma | 2,191 |
| Meningioma | 1,478 |
| Pituitary | 1,485 |
| No Tumor | 787 |
| **Total** | **5,941** |

Published split:

| Split | Images |
|---|---:|
| Train | 4,160 |
| Validation | 892 |
| Test | 889 |

The dataset documentation reports exact and near-duplicate removal and zero image-content overlap between train/validation/test.

## Why this dataset

The college project requires four classes, so the previous three-class Jun Cheng dataset was not sufficient.

BDNeuro-MRI is a strong fit for the MVP because:

- it directly provides all four required classes;
- it provides train/validation/test folders;
- the published release documents duplicate removal and split-level leakage verification;
- it is small enough for transfer learning on a student GPU.

## Expected local structure

```text
data/raw/BDNeuro-MRI/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

The preparation script preserves these supplied splits.

## Scientific limitation

The final released images are anonymized and do not retain original patient identifiers. Therefore, CereScan can report the published split/leakage verification, but cannot independently reconstruct patient-level separation from the final image files.

Our evaluation should therefore be described as:

> Held-out test evaluation on the published leakage-checked BDNeuro-MRI split.

Do not describe it as an independently verified patient-disjoint clinical evaluation.

## Download

```bash
python scripts/download_dataset.py
python training/prepare_dataset.py
```

The dataset archive and extracted MRI files remain outside Git.

## Citation

> Hira, M. I. K., Hossain, M. S., Bithee, M. M. A., Sara, U., Hasan, M. M., Towsif, A., & Ahmed, M. K. (2026). *BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class Deep Learning Classification*. Mendeley Data, V9. DOI: 10.17632/zwr4ntf94j.9.

Always follow the dataset's attribution and licensing requirements when using or redistributing it.
