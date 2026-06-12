"""General-purpose utilities: reproducibility, checkpointing, metric logging."""

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed every RNG used in the project for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str | Path, model, optimizer=None, lr_scheduler=None, epoch: int = 0) -> None:
    """Save model (and optionally training state) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, model, optimizer=None, lr_scheduler=None, device="cpu") -> int:
    """Load a checkpoint into ``model`` (and optionally training state).

    Returns:
        The epoch stored in the checkpoint (0 if absent).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    return ckpt.get("epoch", 0)


class AverageMeter:
    """Keeps a running average of a scalar quantity (e.g. a loss term)."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)
