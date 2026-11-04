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
         "hidden", "dense", "dropout", "pool", "keras_dense",
         "grad_checkpoint", "drop_path", "same_clips_as", "scales",
         "extra_size", "head", "layers", "attn_heads", "colour_jitter",
         "random_crop", "time_jitter")

# the heads an end-to-end run can end in (Family F adds the last two)
HEADS = ("gru", "transformer", "gru_attn")


def check(cfg):
    """Why this config cannot start yet, or None."""
    from src.models.encoder import SCALES, TAPS, backbone_problem
    problem = backbone_problem(cfg["backbone"], cfg["pretrained"])
    if problem:
        return problem
    if cfg["head"] not in HEADS:
        return f"video_gru ends in one of {HEADS}, not {cfg['head']!r}"
    if cfg["scales"] not in SCALES:
        return f"scales must be one of {SCALES}"
    if cfg["scales"] == "multi" and cfg["backbone"] not in TAPS:
        return f"multi-scale knows the stages of {tuple(TAPS)} only"
    return data_problem(cfg)


def data_problem(cfg):
    """Why the clips this config reads are not all there yet, or None.
    video3d asks the same question."""
    import os

    folder = os.path.join(cfg["processed"], cfg["variant"])
    if not os.path.isdir(folder):
        if cfg["variant"] == "replica":
            return "needs the replica clips: python -m tools.replica --build --all"
        return f"needs {folder}"
    # a folder is not a dataset: the 300-clip replica sample made R03 look
    # ready, and the dataset quietly skips clips that are missing
    have, want = _present(cfg)
    if have < want * 0.98:
        if cfg["variant"] == "replica":
            return (f"needs the replica clips ({have} of {want} train clips): "
                    f"python -m tools.replica --build --all")
        return f"needs {folder} ({have} of {want} train clips)"
    return None


def _present(cfg):
    """(train clips this variant has, train clips the box variant has).

    Box is the yardstick, not the split: 146 of the author split's 6,466
    train clips never came out of preprocessing (Day 4), and every run has
    always trained on the other 6,320."""
    import csv
    import os

    have = 0
    want = 0
    with open(os.path.join(cfg["split"], "train.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rel = row["clip"].replace("/", os.sep)[:-4] + ".npy"
            if not os.path.exists(os.path.join(cfg["processed"], "box", rel)):
                continue
            want += 1
            if os.path.exists(os.path.join(cfg["processed"], cfg["variant"],
                                           rel)):
                have += 1
    return have, want


def datasets(cfg):
    import os

    train_set = CricShotDataset(os.path.join(cfg["split"], "train.csv"),
                                variant=cfg["variant"],
                                processed=cfg["processed"],
                                transform=build_transform(
                                    cfg["flip"], colour=cfg["colour_jitter"],
                                    crop=cfg["random_crop"],
                                    time=cfg["time_jitter"]),
                                same_clips_as=cfg["same_clips_as"])
    val_set = eval_set(cfg, os.path.join(cfg["split"], "val.csv"))
    return train_set, val_set


def eval_set(cfg, split_csv, classes=None):
    return CricShotDataset(split_csv, variant=cfg["variant"],
                           processed=cfg["processed"], classes=classes,
                           same_clips_as=cfg["same_clips_as"])


def model(cfg, train_set, num_classes):
    return build_model(num_classes=num_classes, backbone=cfg["backbone"],
                       pretrained=cfg["pretrained"], freeze=cfg["freeze"],
                       hidden=cfg["hidden"], dense=cfg["dense"],
                       dropout=cfg["dropout"], pool=cfg["pool"],
                       keras_dense=cfg["keras_dense"],
                       grad_checkpoint=cfg["grad_checkpoint"],
                       drop_path=cfg["drop_path"], scales=cfg["scales"],
                       extra_size=cfg["extra_size"], head=cfg["head"],
                       layers=cfg["layers"], attn_heads=cfg["attn_heads"])
