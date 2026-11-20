# CereScan Architecture

## Four-class MVP pipeline

```text
MRI image
   ↓
Input validation
   ↓
Resize to 224 × 224
   ↓
ImageNet normalization
   ↓
ResNet18 transfer learning
   ↓
Fully connected layer (4 outputs)
   ↓
Softmax
   ├── Glioma
   ├── Meningioma
   ├── Pituitary
   └── No Tumor
   ↓
Prediction + model probability
   ↓
Grad-CAM explanation
```

## Training pipeline

```text
BDNeuro-MRI
   ↓
Published train / validation / test split
   ↓
Class-balanced CrossEntropyLoss
   ↓
ResNet18 pretrained weights
   ↓
AdamW optimization
   ↓
Validation selection
   ↓
Best checkpoint
   ↓
Held-out test metrics
```

## Mathematical operations

The model uses standard deep-learning operations:

- image normalization: `x_norm = (x - μ) / σ`
- convolution: local weighted sums over image regions
- ReLU: `f(x) = max(0, x)`
- linear layer: `z = Wx + b`
- four-class softmax: `p_i = exp(z_i) / Σ_j exp(z_j)`
- cross-entropy loss: `L = -Σ_i y_i log(p_i)`
- backpropagation computes gradients of `L`
- AdamW updates the network parameters

The final softmax output has four values, one for each required class.

## What CereScan does not calculate

The current system is a **classification** system. It does not calculate:

- tumor volume;
- tumor diameter;
- tumor growth/regression;
- treatment response;
- a clinical segmentation boundary.

A future segmentation module would require pixel-level masks and a separate segmentation model/evaluation protocol.

## Explainability

Grad-CAM uses gradients from the final convolutional feature maps to produce a heatmap of image regions that influenced the selected prediction. It is an explanatory aid, not a validated tumor mask.
