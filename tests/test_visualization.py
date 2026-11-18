import numpy as np
from PIL import Image

from neuro_scan.visualization import overlay_heatmap


def test_overlay_heatmap_shape():
    image = Image.new("RGB", (32, 32))
    heatmap = np.zeros((32, 32), dtype=np.uint8)
    result = overlay_heatmap(image, heatmap)
    assert result.size == (32, 32)
