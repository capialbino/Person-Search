"""DINOv2 ViT backbone for the Person Search model (backbone ablation).

DINOv2 (Oquab et al., 2023) is a self-supervised Vision Transformer. Unlike a
CNN, it produces a *single*-scale grid of patch tokens (stride = patch_size),
not a natural multi-scale pyramid. To plug it into the existing Faster R-CNN
heads (which expect 5 feature levels, see ``_pedestrian_anchor_generator`` in
``person_search.py``), we build a ViTDet-style simple feature pyramid (Li et
al., 2022): the single-scale ViT feature map is projected to ``out_channels``
and then both upsampled (×4, ×2) and downsampled (÷2, ÷4) to produce 5 levels
at approximate strides [4, 8, 16, 32, 64] — matching the ResNet-101 baseline.

To keep the ablation fair, only the first ``num_blocks - unfreeze_blocks``
encoder blocks are frozen; the last ``unfreeze_blocks`` blocks and the final
layer-norm are trained with a reduced learning rate (see train.py). This
mirrors how the ResNet-101 baseline is fine-tuned end-to-end.

Weights are downloaded from HuggingFace (``facebook/dinov2-base``) via the
``transformers`` library, since the official ``torch.hub`` source
(facebookresearch/dinov2 on GitHub) was unreachable in this environment.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection import FasterRCNN

from .oim import OIMLoss
from .person_search import PersonSearchModel, PersonSearchRoIHeads, _pedestrian_anchor_generator
from .reid_head import ReIDHead


class Dinov2Backbone(nn.Module):
    """Partially fine-tuned DINOv2 ViT + ViTDet-style simple feature pyramid.

    The first (num_blocks - unfreeze_blocks) encoder blocks are frozen; the
    last unfreeze_blocks blocks and the layer-norm are trained (at a reduced LR
    configured in train.py). The pyramid produces 5 feature levels at
    approximate strides [4, 8, 14, 28, 56], matching the ResNet-101 baseline's
    [4, 8, 16, 32, 64] closely enough for the shared anchor generator and RoI
    align to work correctly.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        out_channels: int = 256,
        unfreeze_blocks: int = 6,
    ):
        super().__init__()
        from transformers import Dinov2Model

        self.vit = Dinov2Model.from_pretrained(model_name)
        self.patch_size = self.vit.config.patch_size
        embed_dim = self.vit.config.hidden_size
        self.out_channels = out_channels
        self.unfreeze_blocks = unfreeze_blocks

        # Freeze everything, then selectively unfreeze the tail of the encoder.
        for p in self.vit.parameters():
            p.requires_grad = False
        num_blocks = len(self.vit.encoder.layer)
        for block in self.vit.encoder.layer[num_blocks - unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.vit.layernorm.parameters():
            p.requires_grad = True

        # Project ViT embed dim → FPN channels (trained from scratch).
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        # ViTDet simple feature pyramid: upsample for fine levels, downsample
        # for coarse levels (each uses bilinear resize + 3×3 conv to avoid
        # checkerboard artifacts from ConvTranspose2d).
        def _upsample_block(scale):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )

        self.up4 = _upsample_block(4)   # stride 14 → ~3.5  (≈ 4)
        self.up2 = _upsample_block(2)   # stride 14 → ~7    (≈ 8)
        self.down2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen encoder blocks in eval mode so their BatchNorm / dropout
        # statistics are not disturbed during training.
        num_blocks = len(self.vit.encoder.layer)
        self.vit.embeddings.eval()
        for block in self.vit.encoder.layer[: num_blocks - self.unfreeze_blocks]:
            block.eval()
        return self

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        b, _, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size

        tokens = self.vit(x, interpolate_pos_encoding=True).last_hidden_state
        patch_tokens = tokens[:, 1:]   # drop CLS token
        feat_map = patch_tokens.transpose(1, 2).reshape(b, -1, grid_h, grid_w)

        p = self.proj(feat_map)        # stride 14  → shared base
        p0 = self.up4(p)               # stride ~4  → level "0"
        p1 = self.up2(p)               # stride ~8  → level "1"
        p2 = p                         # stride 14  → level "2"
        p3 = self.down2(p)             # stride 28  → level "3"
        pool = self.pool(p3)           # stride 56  → "pool"
        return OrderedDict([("0", p0), ("1", p1), ("2", p2), ("3", p3), ("pool", pool)])


def build_dinov2_model(
    num_pids: int = 483,
    embedding_dim: int = 256,
    reid_loss: torch.nn.Module | None = None,
    min_size: int = 900,
    max_size: int = 1500,
    model_name: str = "facebook/dinov2-base",
    unfreeze_blocks: int = 6,
) -> PersonSearchModel:
    """Build the DINOv2-backbone ablation: ViT-B/14 (partially fine-tuned) + ViTDet FPN, OIM loss."""
    backbone = Dinov2Backbone(model_name=model_name, out_channels=256, unfreeze_blocks=unfreeze_blocks)
    model = PersonSearchModel(
        backbone,
        num_classes=2,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=_pedestrian_anchor_generator(),
        box_nms_thresh=0.4,
        box_detections_per_img=100,
    )

    base = model.roi_heads
    model.roi_heads = PersonSearchRoIHeads(
        reid_head=ReIDHead(in_channels=backbone.out_channels, embedding_dim=embedding_dim),
        reid_loss=reid_loss or OIMLoss(num_features=embedding_dim, num_pids=num_pids),
        box_roi_pool=base.box_roi_pool,
        box_head=base.box_head,
        box_predictor=base.box_predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=base.score_thresh,
        nms_thresh=base.nms_thresh,
        detections_per_img=base.detections_per_img,
    )
    return model
