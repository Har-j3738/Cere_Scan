# Dataset

## Selected dataset

CereScan will use the **Jun Cheng Brain Tumor Dataset** hosted on Figshare for the first classification milestone.

Official dataset record:
https://doi.org/10.6084/m9.figshare.1512427

The current Figshare record is version 8 and lists the dataset under a **CC BY 4.0** licence. It contains 3,064 T1-weighted contrast-enhanced MRI images from 233 patients and three tumor classes:

| Class | Images |
|---|---:|
| Meningioma | 708 |
| Glioma | 1,426 |
| Pituitary tumor | 930 |
| **Total** | **3,064** |

The original files are provided as MATLAB `.mat` archives. The dataset also contains patient IDs and manually delineated tumor masks/borders.

## Why this dataset

It is small enough for local experimentation on the CereScan development machine while still providing patient identifiers and tumor annotations that allow us to build a more rigorous experimental pipeline.

For the deadline MVP we will use the **classification labels**. The supplied tumor masks will be treated as future/optional data for segmentation experiments rather than mixing classification and segmentation into the first milestone.

## Critical data-splitting rule

The dataset contains multiple 2D slices from the same patients. We must not randomly split individual slices into train and test sets if that causes the same patient to appear in both sets.

CereScan will therefore keep the patient identifier available during dataset preparation and aim for **patient-disjoint train/validation/test splits**.

## Download policy

The dataset itself is **not stored in this Git repository**.

Reasons:

- the archive is hundreds of MB;
- GitHub is not a dataset storage service;
- committing the data would make cloning the code unnecessarily large;
- reproducible projects should document the source and download process rather than silently bundling the data.

Use `scripts/download_dataset.py` to download the official archives into the local `data/raw/` directory.

## Attribution

When using the dataset, cite the original Jun Cheng dataset and the associated research papers. Preserve the CC BY 4.0 attribution requirements.

## Scope limitation

This dataset contains three tumor categories. It does **not** provide a healthy/no-tumor class in the original collection. CereScan therefore must not advertise a fourth "no tumor" class unless a separately documented dataset is added and the experimental protocol is changed.
