import numpy as np

from neuro_scan.dataset import LABELS, normalize_mri, to_rgb


def test_labels_are_three_class():
    assert LABELS == {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}


def test_normalize_mri_returns_uint8():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    normalized = normalize_mri(image)
    assert normalized.dtype == np.uint8


def test_to_rgb():
    rgb = to_rgb(np.ones((8, 8), dtype=np.float32))
    assert rgb.mode == "RGB"
    assert rgb.size == (8, 8)
