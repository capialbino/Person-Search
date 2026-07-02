# Person Search on PRW

**Student:** Alessandro Capialbi — 0001191564 — alessandro.capialbi@studio.unibo.it  
**Course:** Machine Learning for Computer Vision, University of Bologna, A.Y. 2025–2026

**Repository Github:** https://github.com/capialbino/Person-Search

---

## Project Structure

```
ml4cv-exam-project/
├── main.ipynb                 # Submission notebook (self-contained)
├── requirements.txt
├── checkpoints/               # Model weights (download from OneDrive — see below)
│   ├── baseline/last.pt
│   ├── dinov2_fair/last.pt
│   ├── circle/last.pt
│   └── color_jitter/last.pt
├── vis/baseline/              # Pre-computed qualitative results
│   ├── detections/            # Detection overlays (6 gallery images)
│   └── rankings/              # Query ranking strips (8 queries, top-6 each)
├── assets/
│   ├── pipeline_training.png
│   └── pipeline_inference.png
└── data/PRW/                  # PRW dataset (see Dataset section below)
```

> `main.ipynb` is fully self-contained: all model definitions are inlined in the notebook. The original `src/` directory (that can be founded on GitHub) contains the original training code and is not required to run inference.

---

## Dataset

This project uses the **PRW (Person Re-identification in the Wild)** dataset, which provides full surveillance video frames annotated with bounding boxes and identity labels for joint person detection and re-identification.

| Split | Frames | IDs | Pedestrians | w/ ID | w/o ID |
|---|---|---|---|---|---|
| Train | 5,134 | 482 | 16,243 | 13,416 | 2,827 |
| Val | 570 | 482 | 1,805 | 1,491 | 314 |
| Test | 6,112 | 450 | 25,062 | 19,127 | 5,935 |
| **Total** | **11,816** | **932** | **43,110** | **34,034** | **9,076** |

The Test split also includes **2,057 query crops** (one per query instance). Inference uses only the Test split (gallery frames + query crops).

Place the dataset under `data/PRW/`

---

## Model Weights

Download the weights and place them in the corresponding `checkpoints/` subfolders.

| Experiment | File path | Download |
|---|---|---|
| Baseline | `checkpoints/baseline/last.pt` | [OneDrive](https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgCO5zjXFBHNQ4q4Wn2BnRNCAbPNzyEUb3-lP5KexubRKJ8?e=2BdU8R) |
| Ablation 1 — DINOv2 | `checkpoints/dinov2_fair/last.pt` | [OneDrive](https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgCqfwVMTIUHS4491Xn12hYKAcbrnp0p3k6qZ6b2LEQK00o?e=bw5Wa9) |
| Ablation 2 — Circle Loss | `checkpoints/circle/last.pt` | [OneDrive](https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgDj5xyUVJohTK6Q7pFuEF4jAYtRTK9kajJk69ZGfm9L6DA?e=dpnYTe) |
| Ablation 3 — Color Jitter | `checkpoints/color_jitter/last.pt` | [OneDrive](https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgC8Rd_va4IQSoM6L_mjF3e8ATdaGDUulh9Va0SmM9mIi5w?e=zJmuf4) |

---

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

> **PyTorch / CUDA:** `requirements.txt` ships with PyTorch commented out. If you already have a working PyTorch + CUDA installation, no further action is needed. Otherwise, uncomment the `torch`, `torchvision`, `scipy`, and `transformers` lines in `requirements.txt` and install the wheel that matches your CUDA version.

### Inference

1. Download the PRW dataset and place it under `data/PRW/` (see Dataset section).
2. Download the model weights and place them under `checkpoints/` (see Model Weights section).
3. Open `main.ipynb` and **run all cells**.

The notebook runs inference for all four experiments and displays:
- Qualitative detection and ranking results (pre-computed images from `vis/baseline/`)
- Quantitative metrics (mAP, top-1) computed live via the official PRW evaluation protocol

A GPU is required for inference. Measured on an NVIDIA RTX PRO 2000 (8 GB VRAM):

| Experiment | Inference time |
|---|---|
| Baseline (ResNet-101 + FPN) | ~33 min |
| Ablation 1 (DINOv2 ViT-B/14) | ~70 min |
| Ablation 2 (Circle Loss) | ~15 min |
| Ablation 3 (Color Jitter) | ~15 min |

---

## Results Summary

| Experiment | Backbone | Loss | Augmentation | mAP | top-1 |
|---|---|---|---|---|---|
| Baseline | ResNet-101 + FPN | OIM | — | 41.72 % | 79.92 % |
| Ablation 1 | DINOv2 ViT-B/14 | OIM | — | 20.77 % | 57.41 % |
| Ablation 2 | ResNet-101 + FPN | Circle Loss | — | 41.32 % | 78.71 % |
| Ablation 3 | ResNet-101 + FPN | OIM | Color Jitter | **41.86 %** | 78.76 % |
