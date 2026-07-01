"""Training and evaluation loops."""

import math
import sys

import numpy as np
import torch

from eval_function import eval_search_prw
from .utils import AverageMeter


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, *,
                    grad_clip=10.0, warmup_iters=1000, log_every=100):
    """Train for one epoch with mixed precision. Returns the avg of each loss term.

    A linear learning-rate warmup is applied during the first ``warmup_iters``
    iterations of epoch 0 to stabilize the early OIM updates.
    """
    model.train()
    meters: dict[str, AverageMeter] = {}

    warmup = None
    if epoch == 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, total_iters=min(warmup_iters, len(loader) - 1)
        )

    for it, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]

        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not math.isfinite(loss.item()):
            print(f"WARNING: NaN/inf loss at epoch {epoch} iter {it}, skipping batch", file=sys.stderr)
            print(loss_dict, file=sys.stderr)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            if warmup is not None:
                warmup.step()
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if warmup is not None:
            warmup.step()

        meters.setdefault("loss", AverageMeter()).update(loss.item())
        for name, value in loss_dict.items():
            meters.setdefault(name, AverageMeter()).update(value.item())

        if it % log_every == 0:
            stats = " ".join(f"{k}: {m.avg:.4f}" for k, m in meters.items())
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch} [{it:>5}/{len(loader)}] lr: {lr:.2e} {stats}")

    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def extract_gallery(model, loader, device):
    """Run detection on every gallery scene.

    Returns:
        dets: list (one per image) of [n_det, 5] arrays [x1, y1, x2, y2, score].
        feats: list (one per image) of [n_det, D] embedding arrays.
    """
    model.eval()
    dets, feats = [], []
    for images, _ in loader:
        outputs = model([img.to(device) for img in images])
        for out in outputs:
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            dets.append(np.hstack([boxes, scores[:, None]]).astype(np.float32))
            feats.append(out["embeddings"].cpu().numpy().astype(np.float32))
    return dets, feats


@torch.no_grad()
def extract_queries(model, loader, device):
    """Embed the ground-truth query boxes. Returns a list of D-dim arrays."""
    model.eval()
    feats = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        boxes = [t["boxes"].to(device) for t in targets]
        emb = model.extract_query_features(images, boxes)
        feats.extend(emb.cpu().numpy().astype(np.float32))
    return feats


def evaluate(model, gallery_loader, query_loader, device, *,
             det_thresh=0.5, use_cws=True):
    """Full person search evaluation with the official PRW protocol.

    Args:
        use_cws: Confidence Weighted Similarity (as in SeqNet) — scale each
            gallery embedding by its detection score, so that low-confidence
            detections rank lower at equal cosine similarity.

    Returns:
        The dict produced by ``eval_search_prw`` (contains mAP, top-1 and the
        per-query top-10 ranking, useful for qualitative visualization).
    """
    gallery_dets, gallery_feats = extract_gallery(model, gallery_loader, device)
    query_feats = extract_queries(model, query_loader, device)

    if use_cws:
        gallery_feats = [f * d[:, 4:5] for f, d in zip(gallery_feats, gallery_dets)]

    return eval_search_prw(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_feats,
        det_thresh=det_thresh,
    )
