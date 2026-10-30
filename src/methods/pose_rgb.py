"""Pose and RGB together: pose frames attend to RGB frames (E04, 29.2).

Each clip is the pose cache's (T, 99) beside EfficientNetV2-S's avg cache,
(T, 1280), as one (T, 1379) tensor, so the training loop moves one tensor
as always; `CrossAttnHead` splits it again.

**One flip coin per clip for both halves.** A mirrored pose beside an
unmirrored picture would teach the model that left and right disagree.
"""
import os

import torch
from torch.utils.data import Dataset

from src.data.features import FeatureDataset
from src.models.head import CrossAttnHead

NAME = "pose_rgb"

READS = ("features", "rgb_features", "flip", "attn_heads", "dense",
         "dropout")


class PoseRGBDataset(Dataset):
    """The clips both caches hold, pose then RGB per frame."""

    def __init__(self, split_csv, pose, rgb, mirror=0.0, classes=None):
        self.pose = FeatureDataset(split_csv, features=pose, view="pose",
                                   mirror=mirror, classes=classes)
        self.rgb = FeatureDataset(split_csv, features=rgb, view="avg",
                                  mirror=mirror, classes=self.pose.classes)
        self.classes = self.pose.classes
        self.mirror = mirror
        in_rgb = set()
        for item in self.rgb.items:
            in_rgb.add(item[2])
        self.items = []
        self.missing = list(self.pose.missing)
        for item in self.pose.items:
            if item[2] in in_rgb:
                self.items.append(item)
            else:
                self.missing.append(item[2])
        for clip in self.rgb.missing:
            if clip not in self.missing:
                self.missing.append(clip)
        if not self.items:
            raise RuntimeError(f"no clip in {split_csv} has both caches")
        self.pose_dim = self.pose.feature_dim
        self.rgb_dim = self.rgb.feature_dim
        self.feature_dim = self.pose_dim + self.rgb_dim

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, label, clip = self.items[i]
        flipped = False
        if self.mirror and float(torch.rand(())) < self.mirror:
            flipped = True
        pose = torch.load(self.pose._path(clip, flipped),
                          map_location="cpu").float().flatten(1)
        rgb = torch.load(self.rgb._path(clip, flipped),
                         map_location="cpu").float()
        if pose.shape[0] != rgb.shape[0]:
            raise ValueError(f"{clip}: {pose.shape[0]} pose frames but "
                             f"{rgb.shape[0]} RGB frames")
        return torch.cat([pose, rgb], dim=1), label

    def label_counts(self):
        out = {}
        for cls in self.classes:
            out[cls] = 0
        for _, label, _ in self.items:
            out[self.classes[label]] += 1
        return out


def check(cfg):
    """Why this config cannot start yet, or None."""
    if not cfg["rgb_features"]:
        return "pose_rgb needs rgb_features: an RGB cache with an avg view"
    if not os.path.isdir(os.path.join(cfg["rgb_features"], "avg")):
        return f"needs {os.path.join(cfg['rgb_features'], 'avg')}"
    return None


def datasets(cfg):
    train_set = PoseRGBDataset(os.path.join(cfg["split"], "train.csv"),
                               cfg["features"], cfg["rgb_features"],
                               mirror=cfg["flip"])
    val_set = eval_set(cfg, os.path.join(cfg["split"], "val.csv"))
    return train_set, val_set


def eval_set(cfg, split_csv, classes=None):
    return PoseRGBDataset(split_csv, cfg["features"], cfg["rgb_features"],
                          classes=classes)


def model(cfg, train_set, num_classes):
    data = getattr(train_set, "dataset", train_set)
    return CrossAttnHead(data.pose_dim, data.rgb_dim, num_classes,
                         attn_heads=cfg["attn_heads"], dense=cfg["dense"],
                         dropout=cfg["dropout"])
