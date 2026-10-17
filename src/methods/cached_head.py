"""A swappable head on cached features: the probe now, Family A on Day 10.

    linear   one Linear on the clip's mean feature (P01-P05)
    mlp      Linear -> ReLU -> Linear, same input (Family V)
    gru      the authors' GRU-128, as cached_gru (A03)
    mean     the GRU swapped for an average over frames (A01)
    max      the GRU swapped for the largest value over frames (A02)

Per-frame caches are averaged over time for linear and mlp. Video caches
are already one vector per clip, so there is nothing for a GRU to walk.
"""
import os

from src.data.features import FeatureDataset
from src.models.head import GRUHead, LinearHead, MLPHead, PoolHead

NAME = "cached_head"

READS = ("features", "view", "flip", "head", "hidden", "dense", "dropout",
         "bidirectional", "layers", "attn_heads", "feature_norm")

HEADS = ("linear", "mlp", "gru", "mean", "max")   # the rest come in 10.2-10.4

# heads that read the 15 frames one by one, so they need a time axis
NEEDS_FRAMES = ("gru", "mean", "max")


def check(cfg):
    """Why this config cannot start yet, or None."""
    if cfg["head"] not in HEADS:
        return f"no {cfg['head']} head yet (Day 10)"
    if cfg["bidirectional"]:
        return "bidirectional not built yet (Day 10)"
    if cfg["layers"] != 1:
        return "more than 1 layer not built yet (Day 10)"
    if cfg["feature_norm"] != "none":
        return f"feature_norm {cfg['feature_norm']} not built yet (Day 10)"
    if cfg["head"] in NEEDS_FRAMES and cfg["view"] == "video":
        return (f"{cfg['head']} needs frames; the video view is one vector "
                f"per clip")
    return None


def datasets(cfg):
    train_set = FeatureDataset(os.path.join(cfg["split"], "train.csv"),
                               features=cfg["features"], view=cfg["view"],
                               mirror=cfg["flip"])
    val_set = eval_set(cfg, os.path.join(cfg["split"], "val.csv"))
    return train_set, val_set


def eval_set(cfg, split_csv, classes=None):
    return FeatureDataset(split_csv, features=cfg["features"],
                          view=cfg["view"], classes=classes)


def model(cfg, train_set, num_classes):
    reason = check(cfg)
    if reason:
        raise ValueError(reason)
    data = getattr(train_set, "dataset", train_set)
    dim = data.feature_dim
    if cfg["head"] == "linear":
        return LinearHead(dim, num_classes)
    if cfg["head"] == "mlp":
        return MLPHead(dim, num_classes, cfg["dense"], cfg["dropout"])
    if cfg["head"] in ("mean", "max"):
        return PoolHead(dim, num_classes, how=cfg["head"],
                        dense=cfg["dense"], dropout=cfg["dropout"])
    return GRUHead(dim, num_classes, cfg["hidden"], cfg["dense"],
                   cfg["dropout"])
