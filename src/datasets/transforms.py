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


class ColorJitter:
    """Random brightness, contrast and saturation jitter (tensor-compatible).

    Helps re-ID robustness to illumination differences between query and
    gallery cameras without altering image structure — so RPN quality is
    unaffected, unlike Random Erasing on full images.

    Args:
        brightness: max deviation from 1.0 for brightness factor.
        contrast: max deviation from 1.0 for contrast factor.
        saturation: max deviation from 1.0 for saturation factor.
    """

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image, target):
        from torchvision.transforms.functional import (
            adjust_brightness, adjust_contrast, adjust_saturation,
        )
        if torch.rand(1) < 0.5:
            f = torch.empty(1).uniform_(1 - self.brightness, 1 + self.brightness).item()
            image = adjust_brightness(image, f)
        if torch.rand(1) < 0.5:
            f = torch.empty(1).uniform_(1 - self.contrast, 1 + self.contrast).item()
            image = adjust_contrast(image, f)
        if torch.rand(1) < 0.5:
            f = torch.empty(1).uniform_(1 - self.saturation, 1 + self.saturation).item()
            image = adjust_saturation(image, f)
        return image, target


class RandomErasing:
    """Randomly erase a rectangular region of the image (Zhong et al., AAAI 2020).

    Designed for re-ID: forces the model not to rely on any single body region.
    Applied to the full image; erased pixels are filled with the channel mean.

    Args:
        p: probability of applying the augmentation.
        sl, sh: min/max fraction of image area to erase.
        r1, r2: min/max aspect ratio of the erased rectangle.
    """

    def __init__(self, p: float = 0.5, sl: float = 0.02, sh: float = 0.4,
                 r1: float = 0.3, r2: float = 3.33):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, image, target):
        if torch.rand(1).item() > self.p:
            return image, target
        _, h, w = image.shape
        area = h * w
        for _ in range(100):
            erase_area = torch.empty(1).uniform_(self.sl, self.sh).item() * area
            aspect = torch.empty(1).uniform_(self.r1, self.r2).item()
            eh = int(round((erase_area * aspect) ** 0.5))
            ew = int(round((erase_area / aspect) ** 0.5))
            if eh < h and ew < w:
                x = torch.randint(0, w - ew, (1,)).item()
                y = torch.randint(0, h - eh, (1,)).item()
                image = image.clone()
                image[:, y:y + eh, x:x + ew] = image.mean()
                return image, target
        return image, target


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def build_transforms(is_train: bool, random_erasing: bool = False, color_jitter: bool = False):
    if not is_train:
        return None
    transforms = [RandomHorizontalFlip(0.5)]
    if color_jitter:
        transforms.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    if random_erasing:
        transforms.append(RandomErasing(p=0.5))
    return Compose(transforms)
