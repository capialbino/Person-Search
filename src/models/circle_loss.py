"""Circle Loss for person re-ID (Sun et al., CVPR 2020).

Circle Loss unifies pair-based and class-level metric learning with adaptive
per-pair margins. Unlike OIM, which maintains a momentum LUT and uses
cross-entropy over all identities, Circle Loss optimises a weighted softplus
over per-sample positive/negative similarity gaps using a learnable class
weight matrix. The adaptive weights alpha_p and alpha_n focus training on
pairs that are close to the decision boundary (the "difficult" region).

Same forward interface as OIMLoss: forward(embeddings, pids) → scalar.
Unlabeled persons (pids == -1) are silently ignored.
"""

import torch
import torch.nn.functional as F
from torch import nn


class CircleLoss(nn.Module):
    """Circle Loss with class-level supervision.

    Each of the ``num_pids`` identities has a learnable unit-norm class centre
    (weight vector). For each labeled embedding we compute cosine similarities
    to all centres, then apply the Circle Loss weighting.

    Args:
        num_features: embedding dimensionality (must match the re-ID head).
        num_pids: number of labeled training identities (483 for PRW).
        m: margin — controls the size of the "ambiguous" zone (default 0.25).
        gamma: scale factor applied to logits (default 256).
    """

    def __init__(
        self,
        num_features: int = 256,
        num_pids: int = 483,
        m: float = 0.25,
        gamma: float = 256,
    ):
        super().__init__()
        self.m = m
        self.gamma = gamma
        # Learnable class centres — normalised to unit sphere before use.
        self.weight = nn.Parameter(torch.empty(num_pids, num_features))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, embeddings: torch.Tensor, pids: torch.Tensor) -> torch.Tensor:
        """Compute Circle Loss over labeled person embeddings.

        Args:
            embeddings: [N, D] L2-normalised features from the re-ID head.
            pids: [N] identity labels in [0, num_pids) or -1 for unlabeled.
        """
        labeled = pids >= 0
        if not labeled.any():
            return embeddings.sum() * 0

        feats = embeddings[labeled]          # (M, D)
        labels = pids[labeled]               # (M,)
        M = len(labels)

        # Cosine similarity to all class centres.
        weight = F.normalize(self.weight, dim=1)  # (num_pids, D)
        sim = feats @ weight.t()                  # (M, num_pids)

        # ---- positive term ------------------------------------------------
        # sp: similarity of each sample to its own class centre.
        idx = torch.arange(M, device=labels.device)
        sp = sim[idx, labels]                      # (M,)
        alpha_p = (1 + self.m - sp).detach().clamp(min=0)   # (M,)
        logit_p = -alpha_p * (sp - (1 - self.m)) * self.gamma  # (M,)

        # ---- negative terms -----------------------------------------------
        # sn: similarities to every class centre *except* the true one.
        is_pos = torch.zeros(M, self.weight.size(0), dtype=torch.bool, device=sim.device)
        is_pos[idx, labels] = True
        sn = sim[~is_pos].reshape(M, -1)          # (M, num_pids - 1)
        alpha_n = (sn + self.m).detach().clamp(min=0)        # (M, num_pids - 1)
        logit_n = alpha_n * (sn - self.m) * self.gamma       # (M, num_pids - 1)

        # softplus( logsumexp(negatives) + logit_p ) averaged over the batch.
        loss = F.softplus(torch.logsumexp(logit_n, dim=1) + logit_p).mean()
        return loss
