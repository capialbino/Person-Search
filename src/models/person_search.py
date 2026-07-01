"""End-to-end Person Search model: Faster R-CNN + re-ID embedding branch.

The model is a torchvision Faster R-CNN (ResNet + FPN backbone) whose ROI heads
are extended with a re-ID branch:

  * training: the standard detection losses (RPN + classification + box
    regression) are computed as usual; in parallel, the ROI-aligned features of
    *person* proposals are projected to L2-normalized embeddings and fed to the
    re-ID loss (OIM by default), giving a single joint loss.
  * inference: detections are produced as usual, then their embeddings are
    computed by re-pooling the backbone features on the final refined boxes —
    the exact same path used to extract query features, so query and gallery
    embeddings are directly comparable.

Structure inspired by the OIM baseline (Xiao et al., CVPR 2017) and SeqNet
(https://github.com/serend1p1ty/SeqNet); implementation built on torchvision.
"""

import torch
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss

from .oim import OIMLoss
from .reid_head import ReIDHead


class PersonSearchRoIHeads(RoIHeads):
    """ROI heads with a parallel re-ID embedding branch."""

    def __init__(self, reid_head, reid_loss, **kwargs):
        super().__init__(**kwargs)
        self.reid_head = reid_head
        self.reid_loss = reid_loss

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            return self._forward_train(features, proposals, image_shapes, targets)
        return self._forward_eval(features, proposals, image_shapes)

    def _forward_train(self, features, proposals, image_shapes, targets):
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
            proposals, targets
        )
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        class_logits, box_regression = self.box_predictor(self.box_head(box_features))
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets
        )

        # Re-ID branch: embed only the proposals matched to a person (fg). Their
        # identity is the pid of the matched GT box (-1 if the person is unlabeled).
        pids = []
        for target, idxs in zip(targets, matched_idxs):
            if target["pids"].numel() > 0:
                pids.append(target["pids"][idxs])
            else:  # image without annotated persons: every proposal is background
                pids.append(torch.full_like(idxs, -1))
        pids = torch.cat(pids)
        fg_mask = torch.cat(labels) == 1

        losses = {
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
        }
        # BatchNorm in the re-ID head needs at least 2 samples in training mode.
        if fg_mask.sum() >= 2:
            # Force fp32: BN + FC can overflow in fp16, producing NaN embeddings.
            with torch.amp.autocast("cuda", enabled=False):
                embeddings = self.reid_head(box_features[fg_mask].float())
            losses["loss_reid"] = self.reid_loss(embeddings, pids[fg_mask])
        else:
            losses["loss_reid"] = class_logits.sum() * 0.0
        return [], losses

    def _forward_eval(self, features, proposals, image_shapes):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        class_logits, box_regression = self.box_predictor(self.box_head(box_features))
        boxes, scores, labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )

        # Embeddings are computed on the final refined boxes (after NMS), the same
        # path used for query boxes, so the two feature spaces match exactly.
        embeddings = self.embed_boxes(features, boxes, image_shapes)

        results = [
            {"boxes": b, "labels": l, "scores": s, "embeddings": e}
            for b, l, s, e in zip(boxes, labels, scores, embeddings)
        ]
        return results, {}

    def embed_boxes(self, features, boxes, image_shapes):
        """Compute re-ID embeddings for arbitrary boxes (per-image list)."""
        roi_features = self.box_roi_pool(features, list(boxes), image_shapes)
        embeddings = self.reid_head(roi_features)
        return embeddings.split([len(b) for b in boxes])


class PersonSearchModel(FasterRCNN):
    """Faster R-CNN with re-ID ROI heads and a query feature extraction path."""

    @torch.no_grad()
    def extract_query_features(self, images, boxes):
        """Embed the given ground-truth query boxes.

        Args:
            images: list of CHW float tensors (original resolution).
            boxes: list of [1, 4] xyxy tensors, one query box per image.

        Returns:
            [N, D] tensor of L2-normalized query embeddings.
        """
        targets = [{"boxes": b} for b in boxes]
        images, targets = self.transform(images, targets)  # resizes boxes too
        features = self.backbone(images.tensors)
        resized_boxes = [t["boxes"] for t in targets]
        embeddings = self.roi_heads.embed_boxes(features, resized_boxes, images.image_sizes)
        return torch.cat(embeddings)


def build_resnet_model(
    num_pids: int = 483,
    embedding_dim: int = 256,
    reid_loss: torch.nn.Module | None = None,
    min_size: int = 900,
    max_size: int = 1500,
) -> PersonSearchModel:
    """Build the baseline model: ResNet-101 + FPN backbone, OIM re-ID loss.

    A custom ``reid_loss`` (e.g. CircleLoss) can be injected for ablations.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.IMAGENET1K_V2,
        trainable_layers=3,  # freeze conv1 + layer1, BN frozen everywhere
    )
    model = PersonSearchModel(
        backbone,
        num_classes=2,  # background + person
        min_size=min_size,
        max_size=max_size,
        # pedestrians are tall: bias anchor aspect ratios (h/w) accordingly
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


def _pedestrian_anchor_generator():
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    sizes = ((32,), (64,), (128,), (256,), (512,))  # one size per FPN level
    aspect_ratios = ((1.0, 2.0, 3.0),) * len(sizes)
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
