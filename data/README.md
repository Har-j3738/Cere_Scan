# CereScan data directory

Dataset files are intentionally excluded from Git.

Expected local structure:

```text
data/
├── README.md
└── raw/
    ├── brainTumorDataPublic_1-766.zip
    ├── brainTumorDataPublic_767-1532.zip
    ├── brainTumorDataPublic_1533-2298.zip
    └── brainTumorDataPublic_2299-3064.zip
```

Run:

```bash
python scripts/download_dataset.py
```

The archives are downloaded from the official Figshare dataset record.

Do not commit the downloaded archives or extracted dataset to Git.
