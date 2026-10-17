"""A swappable head on cached features: the probe, Family V and Family A.

    linear       one Linear on the clip's mean feature (P01-P05)
    mlp          Linear -> ReLU -> Linear, same input (Family V)
    gru          the authors' GRU-128, as cached_gru (A03); bidirectional (A05)
    lstm         the same, with an LSTM (A06, A07)
    mean, max    the GRU swapped for a pool over frames (A01, A02)
    transformer  attention across frames, `layers` x `attn_heads` (A08, A09)
    tcn          dilated convolutions over time (A10)
    attnpool     a learned weight per frame (A11)
    netvlad      frames described by 8 learned centres (A12)

`feature_norm: layer` puts a LayerNorm on every frame first (A13).

Per-frame caches are averaged over time for linear and mlp. Video caches
are already one vector per clip, so the frame-reading heads refuse them.
"""
import os

from src.data.features import FeatureDataset
from src.models.head import (WIDTH, AttnPoolHead, FrameNorm, GRUHead,
                             LinearHead, LSTMHead, MLPHead, NetVLADHead,
                             PoolHead, TCNHead, TransformerHead)

NAME = "cached_head"

READS = ("features", "view", "flip", "head", "hidden", "dense", "dropout",
         "bidirectional", "layers", "attn_heads", "feature_norm")

HEADS = ("linear", "mlp", "gru", "lstm", "mean", "max", "transformer", "tcn",
         "attnpool", "netvlad")

# heads that read the 15 frames one by one, so they need a time axis
NEEDS_FRAMES = ("gru", "lstm", "mean", "max", "transformer", "tcn",
                "attnpool", "netvlad")

RECURRENT = ("gru", "lstm")


def check(cfg):
    """Why this config cannot start, or None.

    A setting the chosen head would ignore is refused, not dropped: it
    would still be written into the ledger row as if it had mattered.
    """
    from src.train import DEFAULTS

    head = cfg["head"]
    if head not in HEADS:
        return f"unknown head {head!r}"
    if head in NEEDS_FRAMES and cfg["view"] == "video":
        return f"{head} needs frames; the video view is one vector per clip"
    if cfg["bidirectional"] and head not in RECURRENT:
        return f"bidirectional only applies to gru and lstm, not {head}"
    if cfg["hidden"] != DEFAULTS["hidden"] and head not in RECURRENT:
        return f"hidden only applies to gru and lstm, not {head}"
    if cfg["layers"] != 1 and head != "transformer":
        return f"layers only applies to the transformer, not {head}"
    if head == "transformer":
        if WIDTH % cfg["attn_heads"]:
            return f"attn_heads must divide {WIDTH}"
    elif cfg["attn_heads"] != DEFAULTS["attn_heads"]:
        return f"attn_heads only applies to the transformer, not {head}"
    if cfg["feature_norm"] not in ("none", "layer"):
        return f"unknown feature_norm {cfg['feature_norm']!r}"
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


def _head(cfg, dim, num_classes):
    head = cfg["head"]
    dense = cfg["dense"]
    dropout = cfg["dropout"]
    if head == "linear":
        return LinearHead(dim, num_classes)
    if head == "mlp":
        return MLPHead(dim, num_classes, dense, dropout)
    if head in ("mean", "max"):
        return PoolHead(dim, num_classes, how=head, dense=dense,
                        dropout=dropout)
    if head == "lstm":
        return LSTMHead(dim, num_classes, hidden=cfg["hidden"], dense=dense,
                        dropout=dropout, bidirectional=cfg["bidirectional"])
    if head == "transformer":
        return TransformerHead(dim, num_classes, layers=cfg["layers"],
                               attn_heads=cfg["attn_heads"], dense=dense,
                               dropout=dropout)
    if head == "tcn":
        return TCNHead(dim, num_classes, dense=dense, dropout=dropout)
    if head == "attnpool":
        return AttnPoolHead(dim, num_classes, dense=dense, dropout=dropout)
    if head == "netvlad":
        return NetVLADHead(dim, num_classes, dense=dense, dropout=dropout)
    return GRUHead(dim, num_classes, cfg["hidden"], dense, dropout,
                   bidirectional=cfg["bidirectional"])


def model(cfg, train_set, num_classes):
    reason = check(cfg)
    if reason:
        raise ValueError(reason)
    data = getattr(train_set, "dataset", train_set)
    dim = data.feature_dim
    head = _head(cfg, dim, num_classes)
    if cfg["feature_norm"] == "layer":
        return FrameNorm(dim, head)
    return head
