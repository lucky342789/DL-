# Tomato Leaf Disease Classification — MLP vs. CNN

Two PyTorch image classification pipelines trained on the same dataset and split to compare a fully connected baseline against a convolutional architecture. **Note on comparison rigor**: the two experiments are close but not perfectly controlled — see [Known Asymmetries](#known-asymmetries-between-the-two-experiments) before treating the accuracy gap as attributable to architecture alone.

**Dataset**: PlantVillage Tomato Leaf Disease Dataset ([Kaggle: mohitsingh1804/plantvillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)), 10 classes, 14,529 training images / 3,631 validation images.

**Result**: CNN reaches **95.26%** validation accuracy vs. the MLP baseline's **86.20%** — a +9.06 point improvement.

---

## Repository Structure

```
.
├── assets/
│   ├── confusion_matrix_mlp.png
│   └── confusion_matrix_cnn.png
├── MLP.ipynb
├── CNN.ipynb
└── README.md
```

---

## Shared Setup

- Dataset loaded via `torchvision.datasets.ImageFolder`, pre-split into `train/` and `val/` directories (14,529 / 3,631 images, 10 classes).
- Images resized to **64×64**.
- Normalization statistics computed **from the training set itself** rather than using ImageNet defaults: `mean = [0.4496, 0.4651, 0.4003]`, `std = [0.1656, 0.1448, 0.1831]`.
- Loss function: `CrossEntropyLoss` for both models.

---

## Project 1 — MLP Baseline

### Architecture

```
Input (3×64×64) → Flatten (12,288)
   → Linear(12288 → 768) → ReLU → Dropout(0.310)
   → Linear(768 → 512)   → ReLU → Dropout(0.310)
   → Linear(512 → 10)
```

### Training Configuration

| | |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1.786e-05 |
| Weight decay | 9.077e-04 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | Patience = 5, monitored on validation loss; best checkpoint reloaded for final evaluation |
| Device | MPS |

### Training Log (excerpt)

```
Epoch 1/50  | Train Loss: 1.3169 | Val Loss: 0.9746 | Val Accuracy: 68.88%
Epoch 5/50  | Train Loss: 0.6027 | Val Loss: 0.5654 | Val Accuracy: 81.35%
Epoch 10/50 | Train Loss: 0.3929 | Val Loss: 0.4607 | Val Accuracy: 84.00%
Epoch 13/50 | Train Loss: 0.3084 | Val Loss: 0.4245 | Val Accuracy: 85.90%  <- best checkpoint region
Epoch 15/50 | Train Loss: 0.2717 | Val Loss: 0.4202 | Val Accuracy: 85.95%
Epoch 16/50 | Train Loss: 0.2495 | Val Loss: 0.4190 | Val Accuracy: 85.73%
```

Validation loss and accuracy improve steadily through training, with early stopping correctly halting once no further validation-loss improvement was found — a well-behaved run with no instability.

### Final Result

**Validation Accuracy: 86.20%**

**Confusion Matrix (MLP)**


**Classification Report** (from the notebook, `sklearn.metrics.classification_report`):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterial Spot | 0.83 | 0.88 | 0.86 | 425 |
| Early Blight | 0.69 | 0.49 | 0.58 | 200 |
| Late Blight | 0.83 | 0.82 | 0.82 | 382 |
| Leaf Mold | 0.88 | 0.79 | 0.83 | 191 |
| Septoria Leaf Spot | 0.83 | 0.84 | 0.83 | 354 |
| Spider Mites | 0.78 | 0.87 | 0.82 | 335 |
| Target Spot | 0.80 | 0.80 | 0.80 | 281 |
| Yellow Leaf Curl Virus | 0.95 | 0.95 | 0.95 | 1071 |
| Mosaic Virus | 0.84 | 0.77 | 0.80 | 74 |
| Healthy | 0.92 | 0.94 | 0.93 | 318 |
| **Accuracy** | | | **0.86** | 3631 |
| Macro avg | 0.83 | 0.82 | 0.82 | 3631 |
| Weighted avg | 0.86 | 0.86 | 0.86 | 3631 |

**Early Blight is the clear weak point** — recall of only 0.49 means the model misses roughly half of all actual Early Blight cases, the worst per-class result by a wide margin. Cross-referencing the confusion matrix confirms why: 32 Early Blight images are misclassified as Bacterial Spot and 28 as Late Blight — both diseases produce visually similar early-stage leaf spotting that a flattened pixel vector, with no spatial context, struggles to separate from genuine Early Blight lesions.

---

## Project 2 — CNN

### Architecture

A 5-convolution-layer network with batch normalization, LeakyReLU activations, and Kaiming (He) weight initialization — deeper and more deliberately regularized than a typical introductory CNN:

```
Input (3×64×64)
   → Conv2d(3->64,   k=7, padding=same) -> BatchNorm2d -> LeakyReLU(0.01) -> MaxPool2d(2)
   -> Conv2d(64->128, k=3, padding=same) -> BatchNorm2d -> LeakyReLU(0.01)
   -> Conv2d(128->128,k=3, padding=same) -> BatchNorm2d -> LeakyReLU(0.01) -> MaxPool2d(2)
   -> Conv2d(128->256,k=3, padding=same) -> BatchNorm2d -> LeakyReLU(0.01)
   -> Conv2d(256->256,k=3, padding=same) -> BatchNorm2d -> LeakyReLU(0.01) -> MaxPool2d(2)
   -> Flatten (256x8x8 = 16,384)
   -> Linear(16384 -> 128) -> LeakyReLU(0.01) -> Dropout(0.2)
   -> Linear(128 -> 64)    -> LeakyReLU(0.01) -> Dropout(0.3)
   -> Linear(64 -> 10)
```

**Total parameters: 3,224,010** (all trainable). Weights initialized with `kaiming_normal_` (fan_in, leaky_relu nonlinearity) rather than PyTorch's default initialization — a deliberate choice matched to the LeakyReLU activations used throughout.

### Training Configuration

| | |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Epochs | 20 (fixed — no early stopping) |
| Checkpointing | **None** — the model after epoch 20 is what's evaluated, not the best-validation-loss checkpoint |
| Device | MPS |

### Training Log (full)

```
Epoch [1/20]  Train Loss: 1.6701 | Train Acc: 52.03% | Val Loss: 1.1043 | Val Acc: 63.87%
Epoch [2/20]  Train Loss: 0.9094 | Train Acc: 70.57% | Val Loss: 0.6761 | Val Acc: 78.38%
Epoch [3/20]  Train Loss: 0.7883 | Train Acc: 74.81% | Val Loss: 0.5896 | Val Acc: 81.41%
Epoch [4/20]  Train Loss: 0.6211 | Train Acc: 80.66% | Val Loss: 0.5473 | Val Acc: 84.22%
Epoch [5/20]  Train Loss: 0.5276 | Train Acc: 83.30% | Val Loss: 0.3275 | Val Acc: 89.51%
Epoch [6/20]  Train Loss: 0.5134 | Train Acc: 83.54% | Val Loss: 0.4668 | Val Acc: 85.40%
Epoch [7/20]  Train Loss: 0.4371 | Train Acc: 85.82% | Val Loss: 0.5638 | Val Acc: 81.69%   <- val dip
Epoch [8/20]  Train Loss: 0.3820 | Train Acc: 87.78% | Val Loss: 0.3998 | Val Acc: 87.33%
Epoch [9/20]  Train Loss: 0.3773 | Train Acc: 89.01% | Val Loss: 0.2452 | Val Acc: 92.12%
Epoch [10/20] Train Loss: 0.3536 | Train Acc: 88.91% | Val Loss: 0.3499 | Val Acc: 88.32%
Epoch [11/20] Train Loss: 0.3192 | Train Acc: 90.74% | Val Loss: 0.2175 | Val Acc: 92.48%
Epoch [12/20] Train Loss: 0.3180 | Train Acc: 90.32% | Val Loss: 0.4607 | Val Acc: 86.53%
Epoch [13/20] Train Loss: 0.2578 | Train Acc: 91.64% | Val Loss: 0.6693 | Val Acc: 83.37%   <- val dip
Epoch [14/20] Train Loss: 0.2200 | Train Acc: 92.60% | Val Loss: 0.2522 | Val Acc: 92.23%
Epoch [15/20] Train Loss: 0.2213 | Train Acc: 93.21% | Val Loss: 0.1967 | Val Acc: 94.02%
Epoch [16/20] Train Loss: 0.2696 | Train Acc: 93.17% | Val Loss: 0.3510 | Val Acc: 88.82%   <- val dip
Epoch [17/20] Train Loss: 0.3654 | Train Acc: 88.93% | Val Loss: 0.2059 | Val Acc: 93.17%
Epoch [18/20] Train Loss: 0.2153 | Train Acc: 93.34% | ...
Epoch [20/20] Train Loss: 0.1727 | Train Acc: 94.54% | Val Loss: 0.1439 | Val Acc: 95.26%   (77.38s/epoch)
```

**Validation accuracy is visibly noisy epoch-to-epoch** — three separate dips (epochs 7, 13, 16) where validation accuracy drops 4-11 points below the previous epoch despite training accuracy climbing steadily. This is consistent with a relatively high learning rate (1e-3) for a BatchNorm-heavy network of this depth. The run happened to land on its best epoch at the very end (20/20), but this was not guaranteed — a run stopped at epoch 16, for instance, would have reported 88.82% instead of 95.26%.

### Final Result

**Training Accuracy: 94.54%** · **Validation Accuracy: 95.26%** · **Validation Loss: 0.1439**

**Confusion Matrix (CNN)**



**Classification Report** (derived from the confusion matrix above — the notebook itself only generates the confusion matrix, not a full report):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterial Spot | 0.98 | 0.96 | 0.97 | 425 |
| Early Blight | 0.93 | 0.63 | 0.75 | 200 |
| Late Blight | 0.86 | 0.97 | 0.91 | 382 |
| Leaf Mold | 0.97 | 0.93 | 0.95 | 191 |
| Septoria Leaf Spot | 0.98 | 0.96 | 0.97 | 354 |
| Spider Mites | 0.96 | 0.95 | 0.96 | 335 |
| Target Spot | 0.88 | 0.94 | 0.91 | 281 |
| Yellow Leaf Curl Virus | 0.98 | 1.00 | 0.99 | 1071 |
| Mosaic Virus | 0.96 | 0.95 | 0.95 | 74 |
| Healthy | 0.97 | 1.00 | 0.99 | 318 |
| **Accuracy** | | | **0.95** | 3631 |
| Macro avg | 0.95 | 0.93 | 0.93 | 3631 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 3631 |

**Early Blight remains the weakest class even for the CNN** — recall is still only 0.63, by far the lowest of any class, though a real improvement over the MLP's 0.49. 52 Early Blight images are still misclassified as Late Blight. This is the single clearest signal in both experiments: Early Blight and Late Blight share visual symptoms that neither architecture fully resolves, which is a better next research question than any further generic hyperparameter tuning — targeted data augmentation for this pair, higher input resolution, or literature on the specific visual features that distinguish these two diseases would be the productive next step.

---

## Baseline vs. CNN — Comparison

| | MLP | CNN |
|---|---|---|
| Validation Accuracy | 86.20% | **95.26%** |
| Weighted F1 | 0.86 | 0.95 |
| Worst class (recall) | Early Blight, 0.49 | Early Blight, 0.63 |
| Parameters | ~10.3M (mostly in first FC layer: 12288x768) | 3.2M |
| Epochs to result | 16 (early stopped) | 20 (fixed) |
| Training stability | Smooth, monotonic | Noisy, 3 validation dips |
| Batch size | 32 | 64 |

Despite having roughly **3x fewer parameters**, the CNN outperforms the MLP by 9 points — the improvement comes from architectural inductive bias (local receptive fields, weight sharing, hierarchical feature extraction), not from model capacity. The MLP's 12,288->768 first layer alone contains more parameters (~9.4M) than the CNN's entire network.

---

## Known Asymmetries Between the Two Experiments

Worth stating plainly rather than glossing over, since they affect how much of the accuracy gap is cleanly attributable to architecture alone:

1. **Different batch sizes** (32 for MLP vs. 64 for CNN) — not a controlled variable between the two runs.
2. **Different optimizers and learning rates** (AdamW, lr=1.786e-05 vs. Adam, lr=1e-3) — the MLP's hyperparameters look tuned (non-round values); no evidence in the notebook of how they were selected.
3. **No early stopping or checkpointing for the CNN** — the MLP evaluates its best validation-loss checkpoint; the CNN evaluates whatever state exists after a fixed 20 epochs. Given the CNN's validation accuracy was visibly unstable (see training log), this makes the 95.26% figure somewhat fortunate rather than a controlled, reproducible optimum. Re-running the CNN with the same checkpointing discipline as the MLP would make the comparison more rigorous and the reported number more trustworthy.
4. **BatchNorm and Kaiming initialization are unique to the CNN** — a reasonable and standard pairing with LeakyReLU, but it means the comparison isn't purely "MLP vs. CNN," it's "MLP baseline vs. CNN + BatchNorm + Kaiming init," several changes at once rather than one isolated variable.

None of this erases the result — a 9-point gap with 3x fewer parameters is a real and expected outcome for image data — but a rigorous version of this comparison would isolate architecture as the only changed variable, which this pair of notebooks doesn't quite do.

---

## Additional Analysis in the Notebooks (not reproduced here)

The CNN notebook includes cells that visualize **first-layer feature maps** (all 32 channels from `conv1` applied to a sample image) and the **first-layer convolution kernels themselves** (32 learned 3x3x3 filters, visualized as RGB patches). These are genuinely useful for understanding what the network is learning, but the rendered outputs weren't exported as image files, so they aren't reproduced in this README — re-run the relevant cells in `CNN.ipynb` to regenerate them, and consider saving them to `assets/` for a more complete write-up.

---

## Technologies Used

Python · PyTorch · TorchVision · NumPy · Matplotlib · scikit-learn

---

## Summary

Two experiments, close-to-controlled, show a CNN outperforming an MLP baseline by 9.06 points (86.20% -> 95.26%) on tomato leaf disease classification, using fewer total parameters. The comparison isn't perfectly isolated to architecture alone (see Known Asymmetries above), but the direction and rough magnitude of the result is consistent with what's expected from convolutional inductive bias on image data. Both models share the same weak point — Early Blight vs. Late Blight confusion — pointing to a specific, well-defined next step rather than generic further tuning.
