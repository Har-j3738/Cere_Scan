# Segmentation and longitudinal comparison

CereScan's classification dataset provides class labels, not pixel-level tumor masks. The segmentation loader therefore requires a separate, properly licensed dataset containing paired MRI images and binary masks. Supply a CSV with `image,mask` columns; paths may be absolute or relative to the process working directory. No masks are fabricated.

The binary U-Net returns logits. Train it only on actual paired labels; report Dice and IoU only on held-out labeled data. The helper metrics and pixel area calculation do not demonstrate model quality.

Longitudinal comparisons accept scans the operator has explicitly grouped as one case and dates them. The calculation describes changes in the provided binary-mask pixel counts only; it does not predict future growth. No image registration is performed, so scans must already be comparable in plane, scale, and acquisition. Pixel area and image percentage are not cm² or cm³. Physical volume requires suitable DICOM spatial metadata and 3-D segmentation, which this raster workflow does not provide.

This is an academic prototype, not a medical device or clinical measurement system.
