# Person Search on PRW

**Student:** Alessandro Capialbi — 0001191564 — alessandro.capialbi@studio.unibo.it  
**Course:** Machine Learning for Computer Vision, University of Bologna, A.Y. 2025–2026

---

## Project Structure

```
ml4cv-exam-project/
├── main.ipynb                 # Submission notebook (run this)
├── train.py                   # Training entry point (CLI)
├── visualize.py               # Qualitative visualization script
├── eval_function.py           # Official PRW evaluation protocol (provided)
├── requirements.txt
├── src/
│   ├── datasets/
│   │   ├── prw.py             # PRWDataset (train / gallery / query)
│   │   └── transforms.py      # RandomHorizontalFlip, ColorJitter, RandomErasing
│   ├── models/
│   │   ├── person_search.py   # Faster R-CNN + re-ID head (baseline)
│   │   ├── dinov2_backbone.py # DINOv2 ViT-B/14 + SFP backbone (Ablation 1)
│   │   ├── reid_head.py       # GAP → FC → BN → L2-norm
│   │   ├── oim.py             # Online Instance Matching loss
│   │   └── circle_loss.py     # Circle Loss (Ablation 2)
│   ├── engine.py              # train_one_epoch, extract_gallery, evaluate
│   └── utils.py               # set_seed, save/load_checkpoint, AverageMeter
├── checkpoints/
│   ├── baseline/last.pt       # ResNet-101 + OIM  → mAP 41.72 %, top-1 79.78 %
│   ├── dinov2_fair/last.pt    # DINOv2 + OIM      → mAP 20.70 %, top-1 57.07 %
│   ├── circle/last.pt         # ResNet-101 + CircleLoss → mAP 41.30 %, top-1 78.80 %
│   └── color_jitter/last.pt   # ResNet-101 + OIM + CJ  → mAP 41.86 %, top-1 78.71 %
├── vis/baseline/
│   ├── detections/            # Detection overlays (6 gallery images)
│   └── rankings/              # Query ranking strips (8 queries, top-6 each)
├── assets/
│   ├── pipeline_training.png
│   └── pipeline_inference.png
└── data/PRW/                  # Dataset (not included — see below)
```

---

## How to Run

### Requirements

Run locally on a CUDA-capable GPU. Install dependencies:

```bash
pip install -r requirements.txt
```

> **PyTorch note:** The project was developed with `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` on an RTX PRO 2000 Blackwell (sm_120). For other CUDA versions install the matching wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

### Dataset

Download PRW from Kaggle:
```bash
kaggle datasets download edoardomerli/prw-person-re-identification-in-the-wild
unzip prw-person-re-identification-in-the-wild.zip -d data/PRW
```

### Model Weights

All checkpoint files are > 400 MB and therefore exceed Virtuale's 20 MB per-file limit. Download them from OneDrive and place them in the paths shown below:

| Experiment | File path | OneDrive link |
|---|---|---|
| Baseline | `checkpoints/baseline/last.pt` | *(add link before submission)* |
| Ablation 1 — DINOv2 | `checkpoints/dinov2_fair/last.pt` | *(add link before submission)* |
| Ablation 2 — Circle Loss | `checkpoints/circle/last.pt` | *(add link before submission)* |
| Ablation 3 — Color Jitter | `checkpoints/color_jitter/last.pt` | *(add link before submission)* |

### Running the Notebook

```bash
jupyter notebook main.ipynb
```

Open `main.ipynb` and **run all cells**. Training is disabled; the notebook loads pre-trained weights and displays pre-computed qualitative results from `vis/baseline/`. The optional evaluation cell (clearly commented out) re-runs the full gallery extraction (~25 min).

### Re-training from Scratch

```bash
# Baseline
python train.py --out-dir checkpoints/baseline --eval-at-end

# Ablation 1 — DINOv2 backbone (requires TRANSFORMERS_OFFLINE=1 if no internet)
python train.py --backbone dinov2 --out-dir checkpoints/dinov2_fair --eval-at-end

# Ablation 2 — Circle Loss
python train.py --reid-loss circle --out-dir checkpoints/circle --eval-at-end

# Ablation 3 — Color Jitter augmentation
python train.py --color-jitter --out-dir checkpoints/color_jitter --eval-at-end
```

Each run takes ~13 h on an RTX PRO 2000 8 GB.

---

## Results Summary

| Experiment | Backbone | Loss | Augmentation | mAP | top-1 |
|---|---|---|---|---|---|
| Baseline | ResNet-101 + FPN | OIM | — | 41.72 % | 79.78 % |
| Ablation 1 | DINOv2 ViT-B/14 | OIM | — | 20.70 % | 57.07 % |
| Ablation 2 | ResNet-101 + FPN | Circle Loss | — | 41.30 % | 78.80 % |
| Ablation 3 | ResNet-101 + FPN | OIM | Color Jitter | **41.86 %** | 78.71 % |
