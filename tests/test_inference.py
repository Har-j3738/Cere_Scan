"""Tests for CereScan inference utilities."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from neuro_scan.inference import grad_cam, predict, preprocess
from neuro_scan.model import build_model


def test_preprocess_shape():
    image = Image.new("RGB", (640, 480))
    tensor = preprocess(image)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_predict_returns_four_class_probabilities():
    model = build_model(pretrained=False)
    image = Image.new("RGB", (224, 224))
    predicted, confidence, probabilities = predict(model, image, torch.device("cpu"))

    assert predicted in {"Glioma", "Meningioma", "Pituitary", "No Tumor"}
    assert 0.0 <= confidence <= 1.0
    assert probabilities.shape == (4,)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-6)


def test_grad_cam_returns_normalized_map():
    model = build_model(pretrained=False)
    image = Image.new("RGB", (224, 224))
    heatmap = grad_cam(model, image, torch.device("cpu"))

    assert heatmap.ndim == 2
    assert heatmap.size > 0
    assert np.isfinite(heatmap).all()
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
