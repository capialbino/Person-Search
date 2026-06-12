"""Re-ID embedding head: pooled ROI features -> L2-normalized identity embedding."""

import torch.nn.functional as F
from torch import nn


class ReIDHead(nn.Module):
    """GAP -> FC -> BN -> L2 normalization.

    Global average pooling collapses the 7x7 spatial grid produced by ROI Align,
    a fully-connected layer projects to the embedding space and a BatchNorm layer
    ("BNNeck", common in re-ID literature) smooths the embedding distribution.
    The output is L2-normalized so that the cosine similarity used at inference
    time reduces to a dot product.
    """

    def __init__(self, in_channels: int = 256, embedding_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_channels, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, roi_features):
        """Args: roi_features [N, C, 7, 7] from ROI Align. Returns [N, D] embeddings."""
        x = roi_features.mean(dim=(2, 3))  # global average pooling
        x = self.bn(self.fc(x))
        return F.normalize(x, dim=1)
