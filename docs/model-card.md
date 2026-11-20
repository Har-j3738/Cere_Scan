# CereScan Model Card

## Model

**Architecture:** ResNet18 with ImageNet pretrained weights and a four-class classification head.

**Task:** Brain MRI image classification.

**Classes:** Glioma, Meningioma, Pituitary, No Tumor.

## Intended use

Educational and research prototyping, experimentation, and demonstration of a computer-vision workflow.

## Not intended for

- clinical diagnosis;
- treatment decisions;
- patient triage;
- measuring tumor volume;
- estimating tumor growth or regression;
- replacing a radiologist or other qualified clinician.

## Training data

CereScan uses the documented BDNeuro-MRI dataset release. See [dataset documentation](dataset.md) for version, class counts, licensing, and split information.

## Preprocessing

Input images are converted to RGB, resized to 224×224, converted to tensors, and normalized using ImageNet mean and standard deviation.

Training augmentation currently includes random horizontal flipping and small random rotations.

## Evaluation

The repository contains code for held-out test evaluation using accuracy, macro precision, macro recall, macro F1, classification report, and confusion matrix.

Performance values are intentionally not specified here until a CereScan training run produces them.

## Limitations and risks

Dataset performance may not generalize to clinical scanners, acquisition protocols, populations, or real-world class distributions.

The supplied dataset is preprocessed and anonymized. The project cannot independently reconstruct patient-level separation from the final image files.

Grad-CAM is an explanation aid and should not be interpreted as a validated tumor segmentation.

## Versioning

Model checkpoints are generated artifacts and are not committed to Git. Record the dataset version, code commit, training configuration, and metrics when publishing a model.
