# Neighborhood-based Preprocessing Methods for Brain Metastasis Classification

## Project Overview
This project investigates the impact of neighborhood-based image preprocessing techniques on brain metastasis classification from MRI images using deep learning (ResNet-50).

## Dataset Information
- **Source**: BCBM-RadioGenomics Dataset
- **Split Ratio**: 60% Training, 20% Validation, 20% Testing
- **Format**: 2D MRI slices with corresponding binary masks
- **Input Size**: 224x224 (ResNet standard size)
- **Structure**: 
  - Each patient folder contains:
    - `MRI/` subdirectory with grayscale image slices
    - `MASK/` subdirectory with binary annotation masks

---

## Preprocessing Methods Implemented

### 1. **Histogram Equalization (HE)**

#### Mathematical Formulation
```
Histogram Computation:
h(rk) = count of pixels with intensity rk

Probability Density Function (PDF):
pdf(rk) = h(rk) / (total pixels)

Cumulative Distribution Function (CDF):
cdf(rk) = Σ pdf(i) for i = 0 to k

Normalized CDF:
cdf_normalized(rk) = (cdf(rk) - cdf_min) / (1 - cdf_min)

Output Image Mapping:
s_k = round(cdf_normalized(rk) × (L - 1))
where L = 256 (number of gray levels)
```

#### Purpose
- Redistributes pixel intensities across the full range (0-255)
- Enhances local contrast and visibility of low-contrast features
- Helps improve neural network performance on MRI with varying illumination

#### Implementation Details
- Computed directly using histogram approach
- Applied to training dataset only
- Mask images copied unchanged (no processing required)

---

### 2. **Gaussian Filtering**

#### Mathematical Formulation
```
2D Gaussian Kernel Definition:
G(x, y) = (1 / (2πσ²)) × exp(-(x² + y²) / (2σ²))

where:
- σ = standard deviation (controls blur amount)
- x, y = coordinates relative to kernel center

Kernel Normalization:
kernel_normalized = kernel / Σ(kernel elements)

Convolution Operation:
I_filtered(i, j) = Σ Σ I(i+x, j+y) × G(x, y)
```

#### Parameters
- **Kernel Size**: 5×5
- **Standard Deviation (σ)**: 1.0

#### Purpose
- Reduces high-frequency noise in MRI scans
- Smooths image while preserving general structure
- Helps reduce overfitting by providing regularization through preprocessing

#### Implementation Details
- Convolution with padding (constant_values=0)
- Applied row-by-row and column-by-column
- Output clipped to [0, 255] range

---

### 3. **Laplacian Filtering (Edge Detection & Sharpening)**

#### Mathematical Formulation
```
8-Neighborhood Laplacian Kernel:
L = [  -1  -1  -1 ]
    [  -1   8  -1 ]
    [  -1  -1  -1 ]

Laplacian Response:
L_response(i, j) = Σ Σ I(i+x, j+y) × L(x, y)

Image Sharpening:
I_sharpened = I_original + L_response
```

#### Purpose
- Detects edge pixels through second derivative
- Enhances local discontinuities in image intensity
- Improves distinction between lesion regions and healthy tissue
- Critical for metastasis detection where boundaries are important

#### Implementation Details
- 8-neighborhood (includes diagonals)
- Direct convolution approach without external libraries
- Original image added to Laplacian response for sharpening effect
- Output clipped to [0, 255] range

---

## Model Architecture

### ResNet-50 Configuration
```
- **Base Model**: ResNet-50 (pretrained on ImageNet)
- **Weights**: ImageNet1K_V2
- **Input Shape**: (3, 224, 224) - 3-channel RGB
- **Output Layer**: Single fully connected layer (Binary classification)
- **Activation**: BCEWithLogitsLoss (Binary Cross-Entropy with Logits)
```

### Input Channel Adaptation
- MRI images are grayscale (1 channel)
- Converted to 3-channel by repeating across channels
- Allows use of pretrained ResNet weights without modification

---

## Training Configuration

### Hyperparameters
- **Batch Size**: 16
- **Epochs**: 50 (with early stopping via learning rate scheduler)
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Loss Function**: BCEWithLogitsLoss
- **Scheduler**: ReduceLROnPlateau
  - Mode: min (minimize validation loss)
  - Patience: 3 epochs
  - Factor: 0.5

### Data Augmentation (Training Only)
- Random horizontal flip: p=0.5
- Random rotation: 10 degrees
- Normalization: μ=0.5, σ=0.5 (per-channel)

### Validation & Test
- No augmentation (evaluation transforms only)
- Same normalization as training

---

## Label Definition

### Slice-Level Classification
```
Label = 1: If any non-zero pixel exists in corresponding MASK
Label = 0: If no mask exists OR mask is all zeros
```

### Rationale
- Binary classification at slice level
- Labels derived automatically from mask presence/content
- Enables slice-level predictions before patient-level aggregation

---

## Evaluation Metrics

### Slice-Level Performance Metrics
1. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
2. **Precision**: TP / (TP + FP)
3. **Recall**: TP / (TP + FN)
4. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
5. **Confusion Matrix**: 2×2 matrix showing TP, TN, FP, FN

### Additional Metrics (from PDF)
- **ROC-AUC Score**: Area under the Receiver Operating Characteristic curve
- **ROC Curve**: Trade-off between True Positive Rate and False Positive Rate

---

## Files Generated During Processing

### Output Directories
1. **Train_HE**: Histogram-equalized training images
2. **Train_Gaussian**: Gaussian-filtered training images
3. **Train_Laplacian**: Laplacian-filtered training images

### Saved Models
- **resnet50_best_slice.pth**: Best model checkpoint containing:
  - Model state dictionary
  - Optimizer state dictionary
  - Training epoch number
  - Validation loss value

### Predictions
- **test_slice_predictions.csv**: Test set predictions with columns:
  - image_path: Full path to test image
  - true: Ground truth label (0 or 1)
  - pred: Predicted label (0 or 1)
  - prob: Prediction probability (sigmoid output)

---

## Preprocessing Pipeline Workflow

```
Raw MRI Dataset
    ↓
Data Splitting (Train/Val/Test)
    ↓
┌─────────────────────────────────────────┐
│  Preprocessing Method Selection         │
├─────────────────────────────────────────┤
│ 1. Histogram Equalization (HE)          │
│ 2. Gaussian Smoothing                   │
│ 3. Laplacian Edge Enhancement           │
└─────────────────────────────────────────┘
    ↓
Preprocessed Training Data
    ↓
Resizing to 224×224 & Channel Repeat
    ↓
Data Augmentation (rotation, flip, norm)
    ↓
ResNet-50 Model Training
    ↓
Validation with Scheduler
    ↓
Best Model Checkpoint Selection
    ↓
Test Set Evaluation
    ↓
Performance Metrics & Confusion Matrix
```

---

## Technical Implementation Details

### Data Loading
- PyTorch Dataset & DataLoader classes
- Multi-worker loading (num_workers=4)
- Pin memory for GPU acceleration

### Image Processing
- PIL (Python Imaging Library) for image I/O
- NumPy for numerical operations
- Custom kernel implementations for filters

### Model Training
- PyTorch framework
- CUDA GPU acceleration (if available)
- Reproducibility: Fixed seed (SEED=42)

### Metrics Computation
- scikit-learn for metrics
- seaborn for visualization
- Confusion matrix with heatmap display

---

## Key Equations Summary

| Operation | Equation |
|-----------|----------|
| Histogram Equalization | $s_k = \text{round}\left(\frac{\text{cdf}(k) - \text{cdf}_{\min}}{1 - \text{cdf}_{\min}} \times (L-1)\right)$ |
| Gaussian Kernel | $G(x,y) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$ |
| Laplacian Response | $L = \sum_x \sum_y I(x,y) \times L_{\text{kernel}}(x,y)$ |
| Image Sharpening | $I_{\text{sharp}} = I_{\text{orig}} + L_{\text{response}}$ |
| Binary Classification Loss | $\text{BCEWithLogits}(y, \hat{y})$ |

---

## References & Related Concepts

### Image Processing Techniques
- Histogram Equalization (Adaptive vs Global)
- Gaussian Blur (Frequency domain perspective)
- Laplacian Filters (Second derivative edge detection)
- Convolution operations and kernels

### Deep Learning
- Transfer Learning with pre-trained models
- Fine-tuning strategies
- Data augmentation techniques
- Loss functions for binary classification

### Medical Image Analysis
- MRI preprocessing standards
- Slice-level vs patient-level classification
- Region of Interest (ROI) segmentation
- Contrast enhancement in medical imaging

---

## System Requirements

### Software Dependencies
- Python 3.x
- PyTorch (torch, torchvision)
- NumPy
- Pillow (PIL)
- scikit-learn
- seaborn
- matplotlib
- Google Colab (for notebook execution)

### Hardware Requirements
- GPU (NVIDIA CUDA-capable for acceleration)
- Minimum 8GB RAM
- Storage for dataset and checkpoints

---

## Notes for Reproduction

1. **Dataset Path**: Ensure dataset is accessible at the specified Google Drive paths
2. **Preprocessing Order**: Apply one preprocessing method at a time to compare effects
3. **Model Checkpoints**: Best models are saved automatically during training
4. **Reproducibility**: Seed values (42) ensure consistent results across runs
5. **Visualization**: Comparison plots show preprocessing effectiveness on sample slices

---

## Student Information

**Project**: Impact of Neighborhood-based Preprocessing on Brain Metastasis Classification from MRI Images
**Course**: CSE420 (Digital Image Processing)
**Institution**: Academic Institution
**Submission Date**: 2024

---

## Additional Documentation Files Required

See accompanying files for:
- `ARCHITECTURE.md` - Detailed model architecture diagrams and flow
- `EQUATIONS.md` - Complete mathematical formulations with derivations
- `RESULTS.md` - Performance metrics and visualizations
- `README.md` - Quick start guide and usage instructions
