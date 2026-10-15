"""GRU-128 on cached features. The backbone is not in this model at all.

It already ran, once, in `extract.py`. So this trains the head alone, which is
why it is minutes rather than hours -- and why it scores far lower:

    cached_gru_noflip      51.20% val   flip 0.0
    cached_gru             53.66% val   flip 0.5

Against the end-to-end 75.95% / 79.23%. **Freezing costs 24.75 points**,
because ImageNet features do not hold the discriminative cricket signal and
fine-tuning is what teaches the backbone to see it. These numbers rank heads
against each other; they are not comparable to the baselines.

Flip is not an image transform here -- it is a choice of file, `grid/` or
`grid_mirror/`. Exact rather than approximate, because `ClipFlip` is binary and
both halves are cached.
"""
import os

from src.data.features import FeatureDataset
from src.models.head import GRUHead

NAME = "cached_gru"


def datasets(cfg):
    train_set = FeatureDataset(os.path.join(cfg["split"], "train.csv"),
                               features=cfg["features"], view=cfg["view"],
                               mirror=cfg["flip"])
    val_set = FeatureDataset(os.path.join(cfg["split"], "val.csv"),
                             features=cfg["features"], view=cfg["view"])
    return train_set, val_set


def model(cfg, train_set, num_classes):
    # sized from the cache, never hardcoded: 62720 for view="flat", 1280 for
    # "avg". `train_set` may be a Subset when limit_clips is set.
    data = getattr(train_set, "dataset", train_set)
    return GRUHead(data.feature_dim, num_classes, cfg["hidden"], cfg["dense"],
                   cfg["dropout"])
