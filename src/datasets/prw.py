"""PRW (Person Re-identification in the Wild) dataset.

The dataset ships as:
    PRW root/
    ├── frames/           11816 full-scene .jpg images (e.g. c1s1_000151.jpg)
    ├── annotations/      one .mat file per frame with [pid, x, y, w, h] rows
    ├── query_box/        cropped query images (for visualization only)
    ├── query_info.txt    one query per line: "pid x y w h frame_name"
    ├── frame_train.mat   list of training frame names
    └── frame_test.mat    list of test (gallery) frame names

Identity convention: pids in [1, 932] are labeled identities, pid == -2 marks an
"ambiguous person" (a real pedestrian without identity label).

The annotation parsing logic is adapted from the SeqNet repository
(https://github.com/serend1p1ty/SeqNet/blob/master/datasets/prw.py).
"""

import re
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.v2 import functional as TF

UNLABELED_PID = -2  # raw dataset marker for "person without identity"


def _camera_id(img_name: str) -> int:
    """Extract the camera id from a PRW frame name (e.g. 'c1s1_000151.jpg' -> 1)."""
    return int(re.search(r"c(\d+)", img_name).group(1))


def _load_frame_names(root: Path, split: str) -> list[str]:
    """Read the official train/test frame list from the .mat split files."""
    if split == "train":
        mat = loadmat(root / "frame_train.mat")["img_index_train"]
    else:
        mat = loadmat(root / "frame_test.mat")["img_index_test"]
    return [str(entry[0][0]) + ".jpg" for entry in mat]


def _load_frame_boxes(root: Path, img_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (boxes_xyxy, pids) for one frame from its annotation .mat file.

    The annotation key is inconsistent across files ('box_new', 'anno_file' or
    'anno_previous'), a known quirk of the official release.
    """
    anno = loadmat(root / "annotations" / f"{img_name}.mat")
    for key in ("box_new", "anno_file", "anno_previous"):
        if key in anno:
            rows = anno[key]
            break
    else:
        raise KeyError(f"No annotation key found in {img_name}.mat")
    pids = rows[:, 0].astype(np.int64)
    boxes = rows[:, 1:5].astype(np.float32)  # [x, y, w, h]
    boxes[:, 2:] += boxes[:, :2]  # -> [x1, y1, x2, y2]
    boxes = np.clip(boxes, 0, None)  # a few coordinates are slightly negative
    return boxes, pids


class PRWDataset(Dataset):
    """PRW dataset with three operating modes.

    split="train":   returns (image, target) pairs for detector + re-ID training.
                     ``target["pids"]`` holds labels remapped to [0, num_pids) for
                     the OIM look-up table, with -1 for unlabeled persons.
    split="gallery": returns full test scenes; ``self.annotations`` follows the
                     structure expected by ``eval_search_prw``.
    split="query":   returns test query crops defined in query_info.txt; the model
                     extracts one feature vector per query box.
    """

    def __init__(self, root: str | Path, split: str, transforms=None):
        assert split in ("train", "gallery", "query")
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.img_prefix = str(self.root / "frames")

        if split == "query":
            self.annotations = self._load_query_annotations()
        else:
            frame_split = "train" if split == "train" else "test"
            self.annotations = self._load_scene_annotations(frame_split)

        if split == "train":
            # Map the raw labeled pids (non-contiguous in [1, 932]) to a dense
            # range [0, num_pids) used as class indices by the OIM look-up
            # table. PRW train has 483 labeled identities (matches ID_train.mat).
            labeled = sorted(
                {int(p) for a in self.annotations for p in a["pids"] if p != UNLABELED_PID}
            )
            self.pid_to_label = {pid: i for i, pid in enumerate(labeled)}
            self.num_pids = len(labeled)

    def _load_scene_annotations(self, frame_split: str) -> list[dict]:
        annotations = []
        for img_name in _load_frame_names(self.root, frame_split):
            boxes, pids = _load_frame_boxes(self.root, img_name)
            annotations.append(
                {
                    "img_name": img_name,
                    "boxes": boxes,
                    "pids": pids,
                    "cam_id": _camera_id(img_name),
                }
            )
        return annotations

    def _load_query_annotations(self) -> list[dict]:
        annotations = []
        for line in (self.root / "query_info.txt").read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            pid = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            img_name = parts[5] + ".jpg"
            box = np.clip(np.array([[x, y, x + w, y + h]], dtype=np.float32), 0, None)
            annotations.append(
                {
                    "img_name": img_name,
                    "boxes": box,
                    "pids": np.array([pid]),
                    "cam_id": _camera_id(img_name),
                }
            )
        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        anno = self.annotations[idx]
        image = decode_image(str(Path(self.img_prefix) / anno["img_name"]))
        image = TF.to_dtype(image, torch.float32, scale=True)

        target = {
            "img_name": anno["img_name"],
            "boxes": torch.from_numpy(anno["boxes"].copy()),
            # single foreground class: every pedestrian is a "person" for detection
            "labels": torch.ones(len(anno["boxes"]), dtype=torch.int64),
        }
        if self.split == "train":
            target["pids"] = torch.tensor(
                [self.pid_to_label.get(int(p), -1) for p in anno["pids"]],
                dtype=torch.int64,
            )

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    """Detection-style collate: keep images/targets as lists (variable sizes)."""
    return tuple(zip(*batch))
