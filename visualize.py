"""Visualize a trained model's outputs: raw detections and person-search rankings.

Examples:
    python visualize.py --checkpoint checkpoints/baseline/last.pt --out-dir vis/baseline
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from eval_function import eval_search_prw
from src.datasets.prw import PRWDataset, collate_fn
from src.engine import evaluate, extract_gallery
from src.models.person_search import build_resnet_model
from src.utils import load_checkpoint


def plot_detections(dataset, dets, out_dir, n_images=6, score_thresh=0.5):
    """Draw raw detector output (boxes + scores) on a few gallery images."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(dataset), size=min(n_images, len(dataset)), replace=False)

    for idx in idxs:
        anno = dataset.annotations[idx]
        img_path = Path(dataset.img_prefix) / anno["img_name"]
        image = Image.open(img_path).convert("RGB")
        boxes = dets[idx]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        for x1, y1, x2, y2, score in boxes:
            if score < score_thresh:
                continue
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 5, 0), f"{score:.2f}", color="lime",
                     fontsize=9, fontweight="bold", backgroundcolor="black")
        ax.set_title(anno["img_name"])
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"det_{anno['img_name']}.png", dpi=120)
        plt.close(fig)
    print(f"saved {len(idxs)} detection visualizations to {out_dir}")


def plot_query_rankings(ret, img_root, out_dir, n_queries=6, topk=5, seed=0):
    """For a few queries, draw the query crop and its top-k gallery matches."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = ret["results"]
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(results), size=min(n_queries, len(results)), replace=False)

    for qi in idxs:
        entry = results[qi]
        fig, axes = plt.subplots(1, topk + 1, figsize=(3 * (topk + 1), 4))

        query_img = Image.open(Path(img_root) / entry["query_img"]).convert("RGB")
        x1, y1, x2, y2 = entry["query_roi"]
        axes[0].imshow(query_img.crop((x1, y1, x2, y2)))
        axes[0].set_title("QUERY", color="blue", fontweight="bold")
        axes[0].axis("off")

        for k in range(topk):
            g = entry["gallery"][k]
            gx1, gy1, gx2, gy2 = g["roi"][:4]
            gimg = Image.open(Path(img_root) / g["img"]).convert("RGB")
            crop = gimg.crop((gx1, gy1, gx2, gy2))
            axes[k + 1].imshow(crop)
            color = "green" if g["correct"] else "red"
            mark = "CORRECT" if g["correct"] else "wrong"
            axes[k + 1].set_title(f"#{k+1} {mark}\nscore={g['score']:.3f}", color=color, fontsize=9)
            axes[k + 1].axis("off")
            for spine_pos in ["top", "bottom", "left", "right"]:
                axes[k + 1].spines[spine_pos].set_visible(True)
                axes[k + 1].spines[spine_pos].set_color(color)
                axes[k + 1].spines[spine_pos].set_linewidth(3)

        fig.suptitle(f"Query #{qi}  ({entry['query_img']})")
        fig.tight_layout()
        fig.savefig(out_dir / f"query_{qi:04d}.png", dpi=120)
        plt.close(fig)
    print(f"saved {len(idxs)} query-ranking visualizations to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize person search model outputs")
    parser.add_argument("--data-root", default="data/PRW")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="vis")
    parser.add_argument("--n-det-images", type=int, default=6)
    parser.add_argument("--n-queries", type=int, default=6)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = build_resnet_model().to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    print(f"loaded checkpoint from {args.checkpoint}")

    gallery_set = PRWDataset(args.data_root, "gallery")
    query_set = PRWDataset(args.data_root, "query")
    gallery_loader = DataLoader(
        gallery_set, batch_size=2, num_workers=args.workers, collate_fn=collate_fn
    )
    query_loader = DataLoader(
        query_set, batch_size=2, num_workers=args.workers, collate_fn=collate_fn
    )

    print("running detection on the gallery...")
    gallery_dets, gallery_feats = extract_gallery(model, gallery_loader, device)
    plot_detections(gallery_set, gallery_dets, Path(args.out_dir) / "detections",
                     n_images=args.n_det_images, score_thresh=args.det_thresh)

    print("extracting query features and running full search ranking...")
    from src.engine import extract_queries
    query_feats = extract_queries(model, query_loader, device)
    gallery_feats_cws = [f * d[:, 4:5] for f, d in zip(gallery_feats, gallery_dets)]
    ret = eval_search_prw(
        gallery_set, query_set, gallery_dets, gallery_feats_cws, query_feats,
        det_thresh=args.det_thresh,
    )
    print(f"mAP={ret['mAP']:.2%}  top-1={ret['accs'][0]:.2%}")

    plot_query_rankings(ret, gallery_set.img_prefix, Path(args.out_dir) / "rankings",
                         n_queries=args.n_queries, topk=args.topk)

    (Path(args.out_dir) / "metrics.json").write_text(
        json.dumps({"mAP": float(ret["mAP"]), "top1": float(ret["accs"][0])}, indent=2)
    )


if __name__ == "__main__":
    main()
