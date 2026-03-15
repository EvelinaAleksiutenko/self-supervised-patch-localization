# Self-Supervised Patch Localization - Report

## 1. Methodology

### Approach
The task simulates a simplified version of **image registration**, which is a core component in medical imaging workflows such as adaptive radiotherapy. 

We need to predict the (y, x) top-left coordinates of a 16×16 noisy patch within a 64×64 grayscale source image, **without using ground-truth coordinates in the loss function**. 

I did some research over common practises in solving image registration task and chose a **Siamese correlation filter**. The pipeline is:

1. A **shared-weight CNN encoder** maps both the source image and the patch into a learned feature space.
2. **Cross-correlation** slides the patch features over the source features, producing a 49×49 score map where each position (i, j) represents how well the patch matches at that location.
3. **Soft-argmax** converts the score map into continuous (y, x) coordinates via a temperature-scaled softmax followed by a weighted sum over coordinate grids.
4. **Extraction** (`grid_sample`) crops a 16×16 region from the source image at the predicted coordinates.
5. **NCC loss** (1 − Normalized Cross-Correlation) compares the extracted region to the input patch. No ground-truth coordinates are used.

### Processing alternatives:

- **Classical CV methods**:
  - *Intensity-based* methods (SSD, NCC, etc) slide or warp one image over the other and iteratively minimize a pixel-level dissimilarity score. 
  -  *Feature-based* methods (SIFT, SURF) detect sparse keypoints, compute descriptor vectors, match them across images, and estimate the transformation from matched pairs.  
  
  Both are more sensitive to noise without learned features.
- **Direct regression CNN** (predict (y, x) from concatenated source+patch): requires GT coordinates in the loss, violating the self-supervised constraint.


### Architecture

The encoder is a 3-layer CNN (`Conv2d → BatchNorm → ReLU` ×3) with **no striding or pooling**, preserving a 1:1 spatial correspondence between feature map and pixel coordinates. This ensures the cross-correlation output directly maps to pixel locations without rescaling. The model has **9,552 parameters**.

Batch Normalization stabilizes training by normalizing feature activations, allowing higher learning rates and faster convergence, while also providing mild regularization.

The ReLU activation function is used to introduce non-linearity, allowing the network to learn non-linear features and capture complex structures in images.
### Loss Function

**NCC loss** = 1 − NCC(extracted_region, patch). 

For each image in the batch and for each channel, the NCC is computed between the corresponding H×W patches of the extracted region and the target patch. The resulting NCC values are then averaged across the batch and channels to produce a single scalar loss.

NCC measures the linear correlation between the spatial intensity patterns of two patches. Because the inputs are mean-centered and normalized, NCC is invariant to global brightness and contrast changes. This makes it more robust to intensity shifts and moderate noise than pixel-wise losses such as L1 or MSE. NCC is also a standard similarity metric in image registration literature (e.g., VoxelMorph).

### Loss function alternatives:
- **Contrastive/triplet loss**: would require constructing positive/negative pairs using GT coordinates, causing same violating the self-supervised constraint.
- **L1/MSE as reconstruction loss**: viable alternatives to NCC, but sensitive to additive noise and intensity shifts. NCC is invariant to global intensity offset/scale and is the standard loss in image registration (VoxelMorph).

Pros of choosing NCC loss:
NCC requires mean computation, normalization, dot products
So it involves more operations then as an example L1 or MSE.
### Metrics

- **Training**: NCC loss (the self-supervised objective; should decrease monotonically).
- **Validation**: NCC loss + Mean Euclidean Distance (MED), Median ED, MAE_y, MAE_x, Acc@1px/2px/5px (computed using GT coords, for monitoring only).

    MED directly measures localization accuracy in pixels. 

    MAE_y and MAE_x shows axis-specific biases. 

    Acc@k thresholds show practical precision. 

    Additional metric worth considering: per-class accuracy (to detect hard image types).

## 2. Training & Validation Strategy

### Data Split

The provided `train_val_indices.pt` indices were split **70 / 15 / 15** (train / val / test). Since CIFAR-100 is a toy dataset and the task is straightforward, no exploratory data analysis was performed. An 80/10/10 split would give more training data but evaluate on fewer images - the 70/15/15 split was preferred for more reliable validation estimates.

### Hyperparameters

Key hyperparameters were tuned using **Optuna** (25 epochs per trial). 

Optuna-tuned values: encoder output channels = 16 (minimal capacity of channels), temperature = 1.87 (balances soft-argmax sharpness vs. gradient stability), batch size = 128 (largest possible value due to my VRAM contraint), lr = 7.3e-4, weight decay = 1.5e-5 (light regularization), warmup = 4 epochs, grad clip = 1.13 (prevents gradient explosions). Early stopping patience = 10 epochs.

**Optimizer**: AdamW — decoupled weight decay is more principled than L2 regularization in Adam, and the adaptive per-parameter learning rates help when different parts of the encoder converge at different speeds.

**Scheduler**: linear warmup (4 epochs) → cosine annealing to 1e-6. Warmup avoids large early updates when the soft-argmax outputs are far from the correct location. Cosine annealing provides smooth LR decay without needing to manually pick step milestones.


## 3. Results & Discussion

### Final Performance on validation set

Validation MED during training converged to **0.260 px** 
Final MED on the test set:  **0.230 px**
### Discussion

**Strengths:**
- Accuracy (MED < 0.3 px) with a tiny 9.5K-parameter model.
- Fully self-supervised - no GT coordinates in the training loop.
- Fast inference: ~0.5 ms/sample on GPU, ~8 ms on CPU.
- The architecture is interpretable: the correlation map directly visualizes where the model thinks the patch is.

**Weaknesses:**
- The no-stride encoder processes features at full 64×64 resolution, which would not scale to larger images without modification (e.g. feature pyramid).
- The model is trained on CIFAR-100 only; generalization to real medical images would require domain-specific data and potentially stronger augmentations.

**Potential improvements:**
- Multi-scale features (feature pyramid).
- Data augmentation (brightness, contrast, heavier noise) to improve generalization.

