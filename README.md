# Person Search on PRW

**Student:** Alessandro Capialbi — 0001191564 — alessandro.capialbi@studio.unibo.it  
**Course:** Machine Learning for Computer Vision, University of Bologna, A.Y. 2025–2026
**Repository Github:** https://github.com/capialbino/Person-Search

---

## Project Structure

```
ml4cv-exam-project/
├── main.ipynb                 # Submission notebook 
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
└── data/PRW/                  # Dataset 
```

---

## How to Run

### Requirements

Run locally. Install dependencies:

```bash
pip install -r requirements.txt
```
### Model Weights

| Experiment | File path | OneDrive link |
|---|---|---|
| Baseline | `checkpoints/baseline/last.pt` | *https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgCO5zjXFBHNQ4q4Wn2BnRNCAbPNzyEUb3-lP5KexubRKJ8?e=2BdU8R* |
| Ablation 1 — DINOv2 | `checkpoints/dinov2_fair/last.pt` | *https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgCqfwVMTIUHS4491Xn12hYKAcbrnp0p3k6qZ6b2LEQK00o?e=bw5Wa9* |
| Ablation 2 — Circle Loss | `checkpoints/circle/last.pt` | *https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgDj5xyUVJohTK6Q7pFuEF4jAYtRTK9kajJk69ZGfm9L6DA?e=dpnYTe* |
| Ablation 3 — Color Jitter | `checkpoints/color_jitter/last.pt` | *https://liveunibo-my.sharepoint.com/:f:/g/personal/alessandro_capialbi_studio_unibo_it/IgC8Rd_va4IQSoM6L_mjF3e8ATdaGDUulh9Va0SmM9mIi5w?e=zJmuf4* |

### Running the Notebook

```bash
jupyter notebook main.ipynb
```

Open `main.ipynb` and **run all cells**. Training is disabled; the notebook loads pre-trained weights and displays pre-computed qualitative results from `vis/baseline/`.

---

## Results Summary

| Experiment | Backbone | Loss | Augmentation | mAP | top-1 |
|---|---|---|---|---|---|
| Baseline | ResNet-101 + FPN | OIM | — | 41.72 % | 79.78 % |
| Ablation 1 | DINOv2 ViT-B/14 | OIM | — | 20.70 % | 57.07 % |
| Ablation 2 | ResNet-101 + FPN | Circle Loss | — | 41.30 % | 78.80 % |
| Ablation 3 | ResNet-101 + FPN | OIM | Color Jitter | **41.86 %** | 78.71 % |
