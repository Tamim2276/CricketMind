"""The Dataset that reads cached features instead of video.

The mirror image of `dataset.py`. Same split CSVs, same classes, same labels,
same missing-clip handling -- only the source changed:

    CricShotDataset   data/processed/box/<class>/<clip>.npy      15 pictures
    FeatureDataset    data/features/effnetv2s/grid/<clip>.pt     15 descriptions

So the model that reads from this one is only the temporal head: the frozen
backbone already ran, back in `extract.py`.

`view` picks the shape handed to the head. One view, one directory, one
reshape, and **no fallbacks**:

    "flat"   grid/  (T, C, H, W) -> (T, C*H*W)   62720; reproduces the authors
    "grid"   grid/  (T, C, H, W) -> unchanged    for heads that use position
    "avg"    avg/   (T, C)       -> unchanged    1280; Families A and D

A fallback -- "read avg/ if it exists, else reduce grid/" -- is the kind of
convenience that lets two experiments silently read different numbers, and
nothing downstream could tell you which. `avg/` is written by
`extract.py --derive avg` (step 8.8); until then that view raises.

**The class order comes from `classes.json`, via the same `_classes_for` that
`dataset.py` uses.** Not re-implemented. If this file decided which shot was
label 0 by its own rule, every comparison between a cached run and the
end-to-end baseline would be quietly meaningless.
"""
import csv
import os
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from src.data.dataset import _classes_for

__all__ = ["FeatureDataset", "FEATURES", "VIEWS", "DIR_FOR_VIEW"]

FEATURES = os.path.join("data", "features", "effnetv2s")
VIEWS = ("flat", "grid", "avg")
DIR_FOR_VIEW = {"flat": "grid", "grid": "grid", "avg": "avg"}


class FeatureDataset(Dataset):
    """One cached clip and its label. `(T, D)` for flat/avg, `(T, C, H, W)` for grid.

    Exposes `.items`, `.classes` and `.missing` with the same meanings as
    `CricShotDataset`, because `train.py` reads all three -- `limit_clips`
    indexes `.items[i][1]` for the label, so the tuple layout matters.
    """

    def __init__(self, split_csv: str, features: str = FEATURES,
                 view: str = "flat", classes: Optional[Sequence[str]] = None,
                 strict: bool = False):
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        self.view = view
        self.features = features
        self.root = os.path.join(features, DIR_FOR_VIEW[view])
        if classes:
            self.classes = list(classes)
        else:
            self.classes = _classes_for(split_csv)

        if not os.path.isdir(self.root):
            extra = ""
            if view == "avg":
                extra = "  Run: python -m src.extract --derive avg  (step 8.8)"
            raise FileNotFoundError(
                f"no feature cache at {self.root} for view={view!r}.{extra}")

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
                f"{len(self.missing)} clips in {split_csv} have no {view} "
                f"features, first: {self.missing[0]}")
        if not self.items:
            raise RuntimeError(f"no usable clips in {split_csv} for '{view}'")

        self.item_shape = tuple(self[0][0].shape)
        self.feature_dim = 1
        for size in self.item_shape[1:]:
            self.feature_dim *= size

    def _path(self, clip: str) -> str:
        return os.path.join(self.root, clip.replace("/", os.sep)[:-4] + ".pt")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label, _ = self.items[i]
        # the cache is fp16 to halve the disk; the model wants fp32 and lets
        # autocast decide the rest. That conversion belongs here, so the
        # training loop never has to know what precision the cache uses.
        x = torch.load(path, map_location="cpu").float()
        if self.view == "flat":
            x = x.flatten(1)            # (T, C, H, W) -> (T, C*H*W), C-major
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
        return (f"FeatureDataset({len(self.items)} clips, view={self.view!r}, "
                f"{self.feature_dim} per frame, "
                f"{len(self.classes)} classes{miss})")
