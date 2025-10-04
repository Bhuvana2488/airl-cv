# AIRL Internship Assignment — ViT & Text-Driven Segmentation

This repository contains my submissions for the AIRL internship coding assignment. It includes:

1. **Q1 — Vision Transformer (ViT) on CIFAR-10**  
   Implemented from scratch in PyTorch, trained on CIFAR-10 with best-effort optimizations.  

2. **Q2 — Text-Driven Image Segmentation using SAM 2**  
   End-to-end Colab notebook for segmenting objects based on text prompts.

---

## 🚀 How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/) and upload the relevant notebook (`q1.ipynb` or `q2.ipynb`).  
2. Enable GPU:  Runtime → Change runtime type → GPU
3. Run all cells.

**Q1**: Trains ViT on CIFAR-10, evaluates test accuracy, visualizes accuracy/loss curves.

**Q2**: Loads an image, accepts a text prompt, generates segmentation mask via SAM 2, and displays overlay.

---

## ⚙️ Q1 — Best ViT Model Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 32 |
| Patch Size | 4 |
| Num Classes | 10 |
| Embed Dim | 256 |
| Depth | 8 |
| Num Heads | 8 |
| MLP Ratio | 4.0 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.05 |
| Scheduler | Linear Warmup (5 epochs) → Cosine Decay |
| Batch Size | 128 |
| Epochs | 50 |

---

## 📈 Q1 — Results (CIFAR-10)

| Metric | Value |
|--------|-------|
| Best Test Accuracy |  83.33% |

---

## 🔍 Q1 — Short Analysis

- **Patch Size (4×4)**: Finer spatial details improved recognition of small CIFAR-10 objects; 8×8 patches reduced accuracy by ~2–3%.
- **Depth & Embedding (8 layers, 256-dim)**: Balanced performance and training time; deeper models overfit.
- **Augmentation Impact**: RandomCrop, HorizontalFlip, RandAugment improved generalization (~+3%).
- **Optimizer & LR Schedule**: AdamW with warmup + cosine decay stabilized training and converged faster.
- **Dropout (0.1)**: Slightly boosted validation accuracy (~+1%) and stabilized training.

---
## 📈 Q1 — Training Results

**Final Validation Accuracy: 83.33%** 


![Training Progress](https://github.com/Bhuvana2488/airl-cv/tree/main/images)

**Key Observations:**
- Good generalization with validation accuracy of 83.33%
- Stable training progression without significant overfitting
- Consistent improvement over 80 epochs of training

# Q2 — Text-Driven Image Segmentation with SAM 2

## 🎯 Goal
Perform text-prompted segmentation of objects in images using SAM 2 with text-to-mask conversion via GroundingDINO.

## 🚀 Pipeline Overview

### Step-by-Step Process:
1. **Image Loading** → Load input image from file or URL
2. **Text-to-Box Detection** → Use GroundingDINO to detect objects based on text prompt
3. **Mask Generation** → Feed bounding boxes to SAM 2 for precise segmentation
4. **Result Visualization** → Display original image, detected boxes, and final masks

### Key Components:
- **GroundingDINO**: Converts text prompts to bounding boxes
- **SAM 2 (Segment Anything Model 2)**: Generates pixel-perfect masks from boxes
- **Supervision**: Provides visualization utilities

## 📊 Performance & Results

### Image Processing:
- **Processing Time**: ~10-30 seconds per image
- **Detection Accuracy**: Good object localization with text prompts
- **Segmentation Quality**: High-precision masks with clear boundaries

### Video Processing (Bonus):
- **Processing Time**: ~1 hour for 10-30 second clips
- **Feature**: Frame-by-frame mask propagation using SAM 2
- **Output**: Consistent object tracking across video frames

## ⚙️ Technical Details

### Model Configuration:
- **SAM 2**: `sam2_hiera_s.yaml` with small checkpoint
- **GroundingDINO**: `SwinT_OGC` backbone
- **Detection Threshold**: 0.25 (configurable)

### Key Functions:
- `text_driven_segmentation()`: Main pipeline for single prompt
- `multi_object_segmentation()`: Handle multiple object types
- `analyze_segmentation_quality()`: Quantitative mask analysis
- `test_thresholds()`: Sensitivity analysis for detection

## 🎨 Example Usage

```python
# Single object segmentation
image_url = "https://example.com/image.jpg"
masks, boxes, labels = text_driven_segmentation(image_url, "car")

# Multiple objects
prompts = ["car", "person", "tree"]
multi_object_segmentation(image_url, prompts)

# Quality analysis
analyze_segmentation_quality(masks, boxes, image.shape)

## 📁 Repository Structure
AIRL-Assignment/
├── q1.ipynb # ViT implementation & training on CIFAR-10
├── q2.ipynb # Text-driven segmentation with SAM 2
├── README.md # This file

---

## 🛠️ Requirements

For Q1 (ViT):
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

For Q2 (SAM 2):
- segment-anything-2
- opencv-python
- matplotlib
- numpy

---

## 📝 Notes

- Both notebooks are designed to run on Google Colab with GPU acceleration
- Q1 training takes approximately 45 minutes on Colab GPU
- Q2 provides interactive text prompts for segmentation
- All code is self-contained and includes detailed comments

---

## 📄 License

This project is for educational purposes as part of the AIRL internship assignment.



