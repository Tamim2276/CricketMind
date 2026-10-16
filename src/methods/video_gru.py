"""The authors' method: EfficientNetV2-S over 15 frames, then GRU-128.

End to end -- the backbone trains. This is what produced every baseline:

    mimic_author           79.23% val   78.14% test   flip 0.5
    mimic_author_noflip    75.95% val                 flip 0.0
    mimic_author_seg       76.07% val   76.97% test   variant: seg

Reads `data/processed/<variant>/*.npy`. Flip is an image transform here, done
once per clip by `ClipFlip`.
"""
from src.data.augment import build_transform
from src.data.dataset import CricShotDataset
from src.models.model import build_model

NAME = "video_gru"

# every config key this file acts on, beyond the loop's own
READS = ("variant", "processed", "flip", "backbone", "pretrained", "freeze",
         "hidden", "dense", "dropout", "pool")


def check(cfg):
    """Why this config cannot start yet, or None."""
    from torchvision.models import list_models
    if cfg["backbone"] not in list_models():
        return f"torchvision has no {cfg['backbone']}"
    return None


def datasets(cfg):
    import os

    train_set = CricShotDataset(os.path.join(cfg["split"], "train.csv"),
                                variant=cfg["variant"],
                                processed=cfg["processed"],
                                transform=build_transform(cfg["flip"]))
    val_set = eval_set(cfg, os.path.join(cfg["split"], "val.csv"))
    return train_set, val_set


def eval_set(cfg, split_csv, classes=None):
    return CricShotDataset(split_csv, variant=cfg["variant"],
                           processed=cfg["processed"], classes=classes)


def model(cfg, train_set, num_classes):
    return build_model(num_classes=num_classes, backbone=cfg["backbone"],
                       pretrained=cfg["pretrained"], freeze=cfg["freeze"],
                       hidden=cfg["hidden"], dense=cfg["dense"],
                       dropout=cfg["dropout"], pool=cfg["pool"])
