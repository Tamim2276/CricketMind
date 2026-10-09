"""The one object between .npy files on disk and tensors on the GPU.

Shape contract, stated out loud because everything downstream assumes it:

    on disk   (T, 224, 224, 3) uint8      what build_dataset.py wrote
    __getitem__  (T, 3, 224, 224) float32 normalised
    DataLoader   (B, T, 3, 224, 224)

Normalised with ImageNet mean/std, not 0-1. The pretrained encoder was fitted
on inputs scaled that way, so any other scaling shifts every feature it has
ever seen.

**The split CSVs name clips that do not exist.** 211 of 10,091 failed
preprocessing -- too short after the camera cut, or the batter detected in too
little of the shot. Missing clips are dropped here, loudly, rather than
crashing on epoch 3.
"""
import csv
import json
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["CricShotDataset", "IMAGENET_MEAN", "IMAGENET_STD", "PROCESSED"]

PROCESSED = os.path.join("data", "processed")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CricShotDataset(Dataset):
    """One cropped clip and its label.

    `variant` picks which preprocessing to read: "box" for the plain crop,
    "seg" for the background-removed one. Same clips, same labels, so the two
    are directly comparable.
    """

    def __init__(self, split_csv: str, variant: str = "box",
                 processed: str = PROCESSED, classes: Optional[Sequence[str]] = None,
                 transform=None, strict: bool = False):
        self.variant = variant
        self.processed = processed
        self.transform = transform
        if classes:
            self.classes = list(classes)
        else:
            self.classes = _classes_for(split_csv)
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        self.items: List[tuple] = []
        self.missing: List[str] = []
        with open(split_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                path = self._path(row["clip"])
                if os.path.exists(path):
                    self.items.append((path, int(row["label"]), row["clip"]))
                else:
                    self.missing.append(row["clip"])

        if self.missing and strict:
            raise FileNotFoundError(
                f"{len(self.missing)} clips in {split_csv} have no {variant} "
                f"output, first: {self.missing[0]}")
        if not self.items:
            raise RuntimeError(f"no usable clips in {split_csv} for '{variant}'")

    def _path(self, clip: str) -> str:
        return os.path.join(self.processed, self.variant,
                            clip.replace("/", os.sep)[:-4] + ".npy")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label, _ = self.items[i]
        frames = np.load(path)                       # (T, 224, 224, 3) uint8
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
        x = (x - self.mean) / self.std               # broadcasts over T
        if self.transform is not None:
            x = self.transform(x)
        return x, label

    def label_counts(self) -> dict:
        """Clips per class, after dropping the missing ones."""
        out = {}
        for cls in self.classes:
            out[cls] = 0
        for _, label, _ in self.items:
            out[self.classes[label]] += 1
        return out

    def __repr__(self) -> str:
        miss = ""
        if self.missing:
            miss = f", {len(self.missing)} missing"
        return (f"CricShotDataset({len(self.items)} clips, "
                f"variant={self.variant!r}, {len(self.classes)} classes{miss})")


def _classes_for(split_csv: str) -> List[str]:
    """classes.json sits beside the split it belongs to."""
    path = os.path.join(os.path.dirname(split_csv), "classes.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no classes.json beside {split_csv}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
