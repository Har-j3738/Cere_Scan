import numpy as np

from neuro_scan.dataset import CLASS_NAMES, LABELS, normalize_mri, to_rgb


def test_labels_are_four_class():
    assert CLASS_NAMES == ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    assert LABELS == {0: "Glioma", 1: "Meningioma", 2: "Pituitary", 3: "No Tumor"}


def test_normalize_mri_returns_uint8():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    normalized = normalize_mri(image)
    assert normalized.dtype == np.uint8


def test_to_rgb():
    rgb = to_rgb(np.ones((8, 8), dtype=np.float32))
    assert rgb.mode == "RGB"
    assert rgb.size == (8, 8)
