# CereScan data directory

MRI data is intentionally excluded from Git.

Expected local structure:

```text
data/
├── README.md
└── raw/
    └── BDNeuro-MRI/
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
            ├── no_tumor/
```

Download and prepare:

```bash
python scripts/download_dataset.py
python training/prepare_dataset.py
```

The dataset archive and extracted MRI images remain local and are ignored by Git.
