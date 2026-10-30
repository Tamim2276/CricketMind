"""Motion into a 3D CNN: optical flow (W01) or frame differences (W05).

Same protocol as Day 23, same r3d_18, two inputs:

    motion: flow   the RAFT cache, mirrored on the fly (flip, negate u)
    motion: diff   the box clips, as video_gru reads them; the differences
                   are taken inside the model, so flipping the clip flips
                   them exactly

W05 is the cheapest motion signal there is, and W01's control: is flow
worth its cost over subtracting frames?
"""
import os

import torch

from src.data.features import FeatureDataset
from src.methods import video_gru
from src.models.motion import KINDS, MotionClassifier
from src.preprocess.flow import mirror_flow

NAME = "motion"

READS = ("motion", "features", "variant", "processed", "flip", "backbone",
         "pretrained", "same_clips_as")

# torchvision's video ResNets: their first layer is stem[0], a Conv3d
BACKBONES = ("r3d_18", "mc3_18", "r2plus1d_18")


class FlowDataset(FeatureDataset):
    """The flow cache, (14, 2, 112, 112) a clip, mirrored on the fly."""

    def __init__(self, split_csv, features, mirror=0.0, classes=None):
        self.flip = mirror        # read by __getitem__ during the parent's init
        super().__init__(split_csv, features=features, view="flow",
                         classes=classes)

    def __getitem__(self, i):
        x, label = super().__getitem__(i)
        if self.flip and float(torch.rand(())) < self.flip:
            x = mirror_flow(x)
        return x, label


def check(cfg):
    """Why this config cannot start yet, or None."""
    if cfg["motion"] not in KINDS:
        return f"motion must be one of {KINDS}, got {cfg['motion']!r}"
    if cfg["backbone"] not in BACKBONES:
        return f"motion needs a video ResNet: one of {BACKBONES}"
    if cfg["motion"] == "flow":
        if not cfg["features"] or not os.path.isdir(
                os.path.join(cfg["features"], "flow")):
            return "needs the flow cache: python -m src.preprocess.flow"
        return None
    return video_gru.data_problem(cfg)


def datasets(cfg):
    if cfg["motion"] == "diff":
        return video_gru.datasets(cfg)
    train_set = FlowDataset(os.path.join(cfg["split"], "train.csv"),
                            cfg["features"], mirror=cfg["flip"])
    val_set = eval_set(cfg, os.path.join(cfg["split"], "val.csv"))
    return train_set, val_set


def eval_set(cfg, split_csv, classes=None):
    if cfg["motion"] == "diff":
        return video_gru.eval_set(cfg, split_csv, classes)
    return FlowDataset(split_csv, cfg["features"], classes=classes)


def model(cfg, train_set, num_classes):
    return MotionClassifier(cfg["backbone"], num_classes, cfg["motion"],
                            pretrained=cfg["pretrained"])
