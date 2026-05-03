# Project Architecture & Mathematical Foundations

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    BRAIN METASTASIS CLASSIFICATION              │
│              Using Neighborhood-Based Preprocessing             │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│               RAW MRI DATASET (2D Slices)                    │
│   Patient Folders → MRI Slices + Binary Mask Annotations    │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              DATA SPLITTING LAYER                            │
│  ┌────────────────┬────────────────┬────────────────┐       │
│  │    Train       │     Val        │     Test       │       │
│  │    (60%)       │     (20%)      │     (20%)      │       │
│  └────────────────┴────────────────┴────────────────┘       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│         PREPROCESSING METHODS (3 Parallel Streams)           │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Histogram  │  │  Gaussian    │  │  Laplacian       │  │
│  │ Equalization│  │  Smoothing   │  │  Sharpening      │  │
│  ├─────────────┤  ├──────────────┤  ├──────────────────┤  │
│  │ Input: HxW  │  │ Input: HxW   │  │ Input: HxW       │  │
│  │ Output: HxW │  │ Output: HxW  │  │ Output: HxW      │  │
│  │ (Enhanced)  │  │ (Smoothed)   │  │ (Edge-Enhanced)  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│         [HE Images]     [Gaussian]      [Laplacian]        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│         IMAGE PREPARATION FOR NEURAL NETWORK                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Resize to 224×224 (ResNet input size)          │   │
│  │  2. Convert Grayscale → RGB (repeat channels)      │   │
│  │  3. Apply Data Augmentation (training only):       │   │
│  │     - Random Flip (50% probability)                │   │
│  │     - Random Rotation (±10 degrees)                │   │
│  │  4. Normalize: μ=0.5, σ=0.5                        │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              DEEP LEARNING MODEL (ResNet-50)                │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Input: (Batch, 3, 224, 224)                       │    │
│  │  ↓                                                  │    │
│  │  Conv Layer 1 (64 filters)                         │    │
│  │  ↓                                                  │    │
│  │  Residual Blocks (Layer 1-4)                       │    │
│  │  ↓                                                  │    │
│  │  Global Average Pooling                            │    │
│  │  ↓                                                  │    │
│  │  Fully Connected: FC(2048 → 1) [MODIFIED]         │    │
│  │  ↓                                                  │    │
│  │  Output: Logit (single value)                      │    │
│  │  ↓                                                  │    │
│  │  Sigmoid: Probability ∈ [0, 1]                    │    │
│  └────────────────────────────────────────────────────┘    │
│         (Pre-trained on ImageNet1K_V2)                     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│          TRAINING LOOP (50 epochs max)                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │  For Each Batch:                                   │    │
│  │  1. Forward Pass: logits = model(images)          │    │
│  │  2. Compute Loss: BCEWithLogitsLoss(logits, y)    │    │
│  │  3. Backward Pass: optimizer.zero_grad(), loss.back() │    │
│  │  4. Update Weights: optimizer.step()              │    │
│  │                                                    │    │
│  │  After Each Epoch:                                 │    │
│  │  - Validate on Val Set                            │    │
│  │  - Save Best Model (lowest val_loss)              │    │
│  │  - Adjust LR via Scheduler (ReduceLROnPlateau)    │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│         EVALUATION ON TEST SET                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Load Best Model Checkpoint                        │    │
│  │  ↓                                                  │    │
│  │  Generate Predictions:                             │    │
│  │  - Probabilities (Sigmoid output)                  │    │
│  │  - Binary Predictions (threshold=0.5)             │    │
│  │  ↓                                                  │    │
│  │  Compute Metrics:                                  │    │
│  │  - Accuracy, Precision, Recall, F1                │    │
│  │  - Confusion Matrix                               │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│              RESULTS & VISUALIZATION                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Training Curves (Loss, Accuracy)              │    │
│  │  2. Confusion Matrix Heatmap                       │    │
│  │  3. Performance Metrics Table                      │    │
│  │  4. Prediction CSV (image, true, pred, prob)      │    │
│  │  5. Before/After Preprocessing Comparisons        │    │
│  └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## Detailed Mathematical Formulations

### 1. Histogram Equalization (HE)

#### Step-by-Step Derivation

**Problem**: Limited dynamic range in MRI images reduces contrast and makes features less distinguishable for neural networks.

**Solution**: Redistribute pixel intensities to use full range while preserving structure.

#### Algorithm

```
INPUT: Image I with dimensions H×W

STEP 1: Compute Histogram
─────────────────────────
h(rk) = count of pixels with intensity value rk
where k ∈ {0, 1, 2, ..., L-1} and L = 256

Total pixels: n = H × W

STEP 2: Compute Probability Density Function (PDF)
──────────────────────────────────────────────────
pdf(rk) = h(rk) / n

This represents the probability that a pixel has intensity rk.

STEP 3: Compute Cumulative Distribution Function (CDF)
─────────────────────────────────────────────────────
cdf(k) = Σ pdf(rj) for j = 0 to k

This accumulates probabilities from 0 to k.

STEP 4: Normalize CDF
────────────────────
Find first non-zero CDF value: cdf_min = min(cdf where cdf > 0)

Normalize:
cdf_norm(k) = (cdf(k) - cdf_min) / (1 - cdf_min)

This ensures output is in range [0, 1]

STEP 5: Map to Output Range [0, 255]
─────────────────────────────────────
s_k = round(cdf_norm(k) × (L - 1))
    = round(cdf_norm(k) × 255)

STEP 6: Create Lookup Table & Apply to Image
──────────────────────────────────────────────
mapping[k] = s_k  (for each intensity level k)

I_output(i, j) = mapping[I_input(i, j)]

OUTPUT: Histogram-equalized image with enhanced contrast
```

#### Mathematical Properties

```
Property 1 - Monotonicity:
  If i < j and cdf(i) < cdf(j), then mapping is monotonic

Property 2 - Conservation:
  ∫ pdf_output(s) ds = ∫ pdf_input(r) dr = 1
  
Property 3 - Full Range Utilization:
  Range of output = [0, 255] (vs potentially smaller input range)

Property 4 - Peak Enhancement:
  High-frequency regions (peaks in histogram) get more bins
  Low-frequency regions get fewer bins
```

#### Example Calculation

```
Consider a small 2×2 image:
I = [100  50]
    [100 150]

Intensity values: {50: 1, 100: 2, 150: 1}
h = {50: 1, 100: 2, 150: 1}
n = 4

PDF:
pdf = {50: 1/4=0.25, 100: 2/4=0.5, 150: 1/4=0.25}

CDF:
cdf = {50: 0.25, 100: 0.75, 150: 1.0}
cdf_min = 0.25

Normalized CDF:
cdf_norm = {50: 0, 100: 0.667, 150: 1.0}

Output mapping (×255):
s = {50: 0, 100: 170, 150: 255}

Output image:
I_out = [170   0]
        [170 255]
```

---

### 2. Gaussian Filtering

#### 2D Gaussian Function

**Definition**: The 2D Gaussian function is a symmetric probability distribution in 2D space.

```
Mathematical Definition:
───────────────────────

G(x, y; σ) = (1 / (2πσ²)) × exp(-(x² + y²) / (2σ²))

where:
  σ = standard deviation (controls spread)
  x = horizontal distance from center
  y = vertical distance from center
  (x, y) ∈ ℝ²
```

#### Kernel Creation

```
For a kernel of size K×K (K = 2k+1, where k=2 for 5×5):
──────────────────────────────────────────────────────

1. Define kernel matrix: kernel[i][j] for i,j ∈ [-k, k]

2. Compute Gaussian value at each position:
   kernel[i][j] = (1 / (2πσ²)) × exp(-(i² + j²) / (2σ²))

3. Normalization:
   kernel_normalized[i][j] = kernel[i][j] / (Σ Σ kernel[i][j])
   
   This ensures: Σ Σ kernel_normalized[i][j] = 1

Example for 3×3 kernel with σ=1:

Unnormalized:
k = [0.0625  0.1250  0.0625]
    [0.1250  0.2500  0.1250]
    [0.0625  0.1250  0.0625]
    (sum = 1.0, already normalized)

For 5×5 kernel with σ=1:
k = [0.0030  0.0133  0.0219  0.0133  0.0030]
    [0.0133  0.0596  0.0983  0.0596  0.0133]
    [0.0219  0.0983  0.1621  0.0983  0.0219]
    [0.0133  0.0596  0.0983  0.0596  0.0133]
    [0.0030  0.0133  0.0219  0.0133  0.0030]
```

#### Convolution Operation

```
Image Convolution:
─────────────────

I_filtered(i, j) = Σ Σ I(i+x, j+y) × G(x, y)
                   x  y

where (x, y) ranges over the kernel support region

More explicitly for 5×5 kernel:

I_filtered(i, j) = I(i-2,j-2)×k(-2,-2) + I(i-2,j-1)×k(-2,-1) + ...
                 + I(i+2,j+2)×k(+2,+2)

Boundary Handling:
──────────────────
Padding method: constant padding with 0

For image boundary at row 0, using 5×5 kernel:
  I_padded(−2,j) = 0
  I_padded(−1,j) = 0
  I_padded(0,j) = I(0,j)
  ...
```

#### Effect Analysis

```
σ = 0.5: Minimal smoothing (only nearest neighbors weighted heavily)

σ = 1.0: Moderate smoothing (bell-curve weight distribution)

σ = 2.0: Heavy smoothing (distant pixels contribute significantly)

Relationship: Larger σ → stronger smoothing → more frequency removal
```

---

### 3. Laplacian Filtering (Edge Detection)

#### Mathematical Foundation

**Laplacian Operator**: Second-order spatial derivative operator

```
Continuous Domain:
──────────────────

∇²f = ∂²f/∂x² + ∂²f/∂y²

The Laplacian measures the local curvature of intensity.

Discrete Approximation:
──────────────────────

∂²f/∂x² ≈ f(i+1,j) + f(i-1,j) - 2×f(i,j)

∂²f/∂y² ≈ f(i,j+1) + f(i,j-1) - 2×f(i,j)

∇²f ≈ f(i+1,j) + f(i-1,j) + f(i,j+1) + f(i,j-1) - 4×f(i,j)

KERNEL (4-Neighborhood):
L₄ = [  0  -1   0]
     [ -1   4  -1]
     [  0  -1   0]

KERNEL (8-Neighborhood):
L₈ = [ -1  -1  -1]
     [ -1   8  -1]
     [ -1  -1  -1]

Derivation of 8-neighborhood:
Including diagonals in second derivative:

∂²f/∂x∂y ≈ f(i+1,j+1) + f(i-1,j-1) - f(i+1,j-1) - f(i-1,j+1)

Combined with x and y second derivatives:
8-point Laplacian = sum of all weighted pixel differences
```

#### Convolution with Laplacian Kernel

```
Laplacian Response:
──────────────────

L_response(i, j) = Σ Σ I(i+x, j+y) × L(x, y)

Using 8-neighborhood Laplacian:

L_response(i,j) = -I(i-1,j-1) - I(i-1,j) - I(i-1,j+1)
                  - I(i,j-1) + 8×I(i,j) - I(i,j+1)
                  - I(i+1,j-1) - I(i+1,j) - I(i+1,j+1)

Property: L_response(i,j) ≈ 0 in homogeneous regions
          L_response(i,j) ≠ 0 at edges (intensity discontinuities)
```

#### Image Sharpening

```
Sharpening Formula:
──────────────────

I_sharpened(i, j) = I_original(i, j) + α × L_response(i, j)

Typically α = 1 for standard sharpening.

Effect of Laplacian addition:
- At edges: L_response is non-zero → intensifies transition
- In homogeneous regions: L_response ≈ 0 → little change
- Results in edge-enhanced image

Output Range Management:
I_sharpened_clipped = clip(I_sharpened, 0, 255)

This ensures valid grayscale range.

Why sharpening helps for metastasis detection:
─────────────────────────────────────────────
1. Metastases often have distinct borders
2. Laplacian enhances these boundaries
3. Neural network can better distinguish lesion regions
4. Improved feature extraction at edges
```

---

## Loss Function & Optimization

### Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

```
Mathematical Definition:
──────────────────────

For a batch of N samples:

L = -1/N × Σ [y_i × log(σ(z_i)) + (1 - y_i) × log(1 - σ(z_i))]

where:
  y_i ∈ {0, 1} = true label
  z_i = model logit output (unbounded)
  σ(z) = sigmoid(z) = 1 / (1 + exp(-z)) ∈ (0, 1)

Numerical Stability:
────────────────────
PyTorch combines sigmoid + BCE into single operation:
- Avoids numerical overflow in sigmoid for large z
- Provides better gradient flow
- Improves numerical stability

Expanded form (when stable computation preferred):
L = -1/N × Σ max(z_i, 0) - z_i × y_i + log(1 + exp(-|z_i|))
```

### Adam Optimizer

```
Update Rule (Adaptive Moment Estimation):
──────────────────────────────────────────

For parameter θ at iteration t:

m_t = β₁ × m_{t-1} + (1 - β₁) × g_t           (first moment estimate)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²         (second moment estimate)

m̂_t = m_t / (1 - β₁ᵗ)                        (bias-corrected first moment)
v̂_t = v_t / (1 - β₂ᵗ)                        (bias-corrected second moment)

θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)

where:
  g_t = gradient ∂L/∂θ
  α = learning rate (1e-4 in this project)
  β₁ = 0.9 (exponential decay rate for first moment, default)
  β₂ = 0.999 (exponential decay rate for second moment, default)
  ε = 1e-8 (small constant for numerical stability)

Properties:
───────────
- Adaptive per-parameter learning rates
- Combines momentum with RMSprop-like scaling
- Generally converges faster than SGD
- Good for complex, high-dimensional loss landscapes
```

### ReduceLROnPlateau Scheduler

```
Schedule:
─────────

If validation_loss does not improve for 'patience' epochs:
  new_learning_rate = old_learning_rate × factor

Configuration used:
- patience = 3 epochs
- factor = 0.5
- mode = 'min' (minimize validation loss)

Pseudocode:
──────────
if no_improvement_for_3_epochs:
    learning_rate = learning_rate × 0.5
    reset_no_improvement_counter

Effect:
───────
Initially: learning rate = 1e-4
After 3 epochs: 5e-5
After 6 more epochs: 2.5e-5
... and so on

Benefit: Fine-tunes learning rate based on training progress
         Helps escape local minima or plateaus
```

---

## Data Augmentation Strategy

```
Augmentation Applied (Training Only):
────────────────────────────────────

1. Random Horizontal Flip
   ─────────────────────
   probability = 0.5
   Effect: Doubles effective dataset size with mirror images
   MRI images are symmetric (brain structures), so valid
   
   Transformation: I_flip(i, j) = I(i, W-1-j)

2. Random Rotation
   ───────────────
   angle = uniform(-10°, +10°)
   Effect: Adds rotational invariance
   Lesions can appear at different orientations
   
   Transformation (2D rotation matrix):
   [x']   [cos(θ)  -sin(θ)] [x]
   [y'] = [sin(θ)   cos(θ)] [y]

3. Normalization
   ──────────────
   μ = 0.5 (per-channel mean)
   σ = 0.5 (per-channel std)
   
   I_norm = (I - μ) / σ
   
   Standardizes input to neural network
   Improves gradient flow and training stability

Rationale:
──────────
- Prevents overfitting to specific training samples
- Encourages learning of robust features
- Increases effective training set size
- Typical strategy for medical imaging
```

---

## Model Architecture Specifics

### ResNet-50 Structure

```
Input Layer:
─────────────
Input shape: (B, 3, 224, 224)
  B = batch size
  3 = RGB channels
  224, 224 = spatial dimensions


Initial Convolution:
─────────────────────
Conv(3, 64, kernel_size=7, stride=2, padding=3)
→ Output: (B, 64, 112, 112)

MaxPool(kernel_size=3, stride=2, padding=1)
→ Output: (B, 64, 56, 56)


ResNet-50 Backbone (Residual Blocks):
─────────────────────────────────────
Layer 1: 3 residual blocks,  64 channels  → (B,  64, 56, 56)
Layer 2: 4 residual blocks, 128 channels  → (B, 128, 28, 28)
Layer 3: 6 residual blocks, 256 channels  → (B, 256, 14, 14)
Layer 4: 3 residual blocks, 512 channels  → (B, 512,  7,  7)


Global Average Pooling:
───────────────────────
→ (B, 512, 1, 1)
→ (B, 512)


Fully Connected Layer (MODIFIED):
──────────────────────────────────
Original: FC(512 → 1000) for ImageNet classification
Modified: FC(512 → 1) for binary classification

Linear(in_features=512, out_features=1, bias=True)
→ Output: (B, 1) [single logit per sample]


Final Output:
─────────────
Sigmoid activation applied during loss computation
Final probability: p = σ(logit) ∈ (0, 1)
```

### Pre-training Strategy

```
Transfer Learning:
──────────────────

1. Load ResNet-50 weights trained on ImageNet1K
   - Weights learned from 1000 classification classes
   - Over 14 million images
   - Strong general-purpose feature extractors

2. Initialization Benefits:
   - Early layers: Low-level features (edges, textures)
   - Middle layers: Mid-level features (shapes, regions)
   - Late layers: High-level features (objects, concepts)
   
   These transfer well to medical images

3. Fine-tuning Strategy:
   - Keep backbone frozen initially (optional)
   - Only train new FC layer head
   - Or: Train entire network with low learning rate
   
   This project: Full network training with α = 1e-4

4. Why Transfer Learning Works:
   - Both ImageNet and MRI contain natural images
   - Feature hierarchies are generalizable
   - Reduces training data requirements
   - Improves convergence speed
```

---

## References & Additional Reading

### Image Processing Concepts
1. Histogram Equalization - Digital Image Processing (Gonzalez & Woods)
2. Gaussian Blur - Signal Processing Theory
3. Edge Detection - Canny, Laplacian operators

### Deep Learning
1. ResNet Architecture (He et al., 2015)
2. Transfer Learning Best Practices
3. Loss Functions for Classification

### Medical Imaging
1. MRI Image Preprocessing Standards
2. Computer-Aided Detection (CAD) Systems
3. Radiomics and Texture Analysis
