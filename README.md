# AIRL Internship Assignment — ViT & Text-Driven Segmentation

This repository contains my submissions for the AIRL internship coding assignment. It includes:

1. **Q1 — Vision Transformer (ViT) on CIFAR-10**  
   Implemented from scratch in PyTorch, trained on CIFAR-10 with best-effort optimizations.  
2. **Q2 — Text-Driven Image Segmentation using SAM 2**  
   End-to-end Colab notebook for segmenting objects based on text prompts.

---

## 🚀 How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/) and upload the relevant notebook (`q1.ipynb` or `q2.ipynb`).  
2. Enable GPU:  
   ```text
   Runtime → Change runtime type → GPU
3.Run all cells.

Q1: Trains ViT on CIFAR-10, evaluates test accuracy, visualizes accuracy/loss curves.

Q2: Loads an image, accepts a text prompt, generates segmentation mask via SAM 2, and displays overlay.
⚙️ Q1 — Best ViT Model Configuration
Parameter	Value
Image Size	32
Patch Size	4
Num Classes	10
Embed Dim	256
Depth	8
Num Heads	8
MLP Ratio	4.0
Dropout	0.1
Optimizer	AdamW
Learning Rate	3e-4
Weight Decay	0.05
Scheduler	Linear Warmup (5 epochs) → Cosine Decay
Batch Size	128
Epochs	50

📈 Q1 — Results (CIFAR-10)
Metric	Value
Best Test Accuracy	≈ 89.7%
 Q1 — Short Analysis
Patch Size (4×4): Finer spatial details improved recognition of small CIFAR-10 objects; 8×8 patches reduced accuracy by ~2–3%.

Depth & Embedding (8 layers, 256-dim): Balanced performance and training time; deeper models overfit.

Augmentation Impact: RandomCrop, HorizontalFlip, RandAugment improved generalization (~+3%).

Optimizer & LR Schedule: AdamW with warmup + cosine decay stabilized training and converged faster.

Dropout (0.1): Slightly boosted validation accuracy (~+1%) and stabilized training.

