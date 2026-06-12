"""Training entry point.

Examples:
    python train.py --data-root data/PRW --out-dir checkpoints/baseline
    python train.py --backbone dinov2 --out-dir checkpoints/dinov2
    python train.py --reid-loss circle --out-dir checkpoints/circle
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.prw import PRWDataset, collate_fn
from src.datasets.transforms import build_transforms
from src.engine import evaluate, train_one_epoch
from src.utils import save_checkpoint, set_seed


def build_model(args, num_pids: int):
    if args.reid_loss == "circle":
        from src.models.circle_loss import CircleLoss

        reid_loss = CircleLoss(num_features=256, num_pids=num_pids)
    else:
        reid_loss = None  # default OIM is built inside the model factory

    if args.backbone == "dinov2":
        from src.models.dinov2_backbone import build_dinov2_model

        return build_dinov2_model(num_pids=num_pids, reid_loss=reid_loss)
    from src.models.person_search import build_resnet_model

    return build_resnet_model(num_pids=num_pids, reid_loss=reid_loss)


def main():
    parser = argparse.ArgumentParser(description="Train a Person Search model on PRW")
    parser.add_argument("--data-root", default="data/PRW", help="PRW dataset root")
    parser.add_argument("--out-dir", default="checkpoints/baseline")
    parser.add_argument("--backbone", choices=["resnet101", "dinov2"], default="resnet101")
    parser.add_argument("--reid-loss", choices=["oim", "circle"], default="oim")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[10, 14])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-at-end", action="store_true", help="run test eval after training")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    train_set = PRWDataset(args.data_root, "train", transforms=build_transforms(is_train=True))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    print(f"train: {len(train_set)} images, {train_set.num_pids} identities")

    model = build_model(args, num_pids=train_set.num_pids).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=0.1
    )
    scaler = torch.amp.GradScaler("cuda")

    history = []
    for epoch in range(args.epochs):
        start = time.time()
        losses = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        lr_scheduler.step()
        history.append({"epoch": epoch, **losses, "minutes": (time.time() - start) / 60})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        save_checkpoint(out_dir / "last.pt", model, optimizer, lr_scheduler, epoch)
        print(f"epoch {epoch} done in {history[-1]['minutes']:.1f} min")

    if args.eval_at_end:
        gallery_set = PRWDataset(args.data_root, "gallery")
        query_set = PRWDataset(args.data_root, "query")
        gallery_loader = DataLoader(
            gallery_set, batch_size=2, num_workers=args.workers, collate_fn=collate_fn
        )
        query_loader = DataLoader(
            query_set, batch_size=2, num_workers=args.workers, collate_fn=collate_fn
        )
        results = evaluate(model, gallery_loader, query_loader, device)
        (out_dir / "metrics.json").write_text(
            json.dumps({"mAP": float(results["mAP"]), "top1": float(results["accs"][0])}, indent=2)
        )


if __name__ == "__main__":
    main()
