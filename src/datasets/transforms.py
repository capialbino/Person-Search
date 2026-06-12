"""Training-time data augmentation.

Resizing and normalization are NOT done here: they are handled inside the model
by torchvision's ``GeneralizedRCNNTransform``, which also rescales the boxes.
"""

import torch


class RandomHorizontalFlip:
    """Flip the image and its bounding boxes horizontally with probability ``p``."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            image = image.flip(-1)
            width = image.shape[-1]
            boxes = target["boxes"]
            target["boxes"] = torch.stack(
                [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1
            )
        return image, target


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def build_transforms(is_train: bool):
    return Compose([RandomHorizontalFlip(0.5)]) if is_train else None
