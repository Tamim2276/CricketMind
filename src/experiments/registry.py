"""Every experiment as data: a base config plus the few keys that change.

An entry names its base and a delta. The delta is the experiment, so "one
variable at a time" is something you can read off the entry, and test.

The run name and the split are never written here. The harness adds both,
so no experiment can be run on one split only by accident.
"""
import os
import re

from src import methods
from src.train import DEFAULTS, LOOP_READS, load_config

__all__ = ["ADOPT", "BASES", "ENTRIES", "FAMILIES", "SPLITS", "entries",
           "get", "family_of", "run_name", "resolve", "unread_keys",
           "status", "config_differences"]

# forward slashes, as in the yamls and the ledger; both OSes accept them
SPLITS = {
    "author": "data/splits_author",
    "grouped": "data/splits_grouped",
}

# family -> (what it asks, the plan day it runs)
FAMILIES = {
    "P": ("linear probe on each cache", "9"),
    "A": ("temporal heads on frozen EfficientNetV2-S", "10-11"),
    "D": ("classical models on frozen EfficientNetV2-S", "12"),
    "V": ("video- and image-pretrained features, frozen", "14-15"),
    "B": ("CNN backbones, end to end", "17-22"),
    "C": ("3D and video models, end to end", "23-27"),
    "R": ("the authors' own method, from their notebooks, on their "
          "preprocessing", "7.7"),
    "E": ("the batter's pose instead of pixels", "28-29"),
    "W": ("bat, ball and motion: small physical signals", "30-32"),
    "F": ("multi-scale, hybrid and two-stream models", "32-36"),
}

MAX_DELTA = 3

# runs that already happened under an older name, with the same config.
# The harness checks the configs match before it reuses one.
ADOPT = {"B01_effnetv2s__author": "mimic_author"}

BASES = {
    # the cache read by cached_gru.yaml, under a head that can be swapped
    "cached_head": {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "cached_head", "head": "gru"},
    },
    # the same cache, averaged over time, fitted once; no epochs to flip over
    "cached_classical": {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "classical", "view": "avg", "head": "linear",
                "flip": 0.0},
    },
    # any cache, one small MLP; most video caches have no time axis left
    "frozen_mlp": {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "cached_head", "head": "mlp"},
    },
    # any cache, one Linear: the probe
    "frozen_linear": {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "cached_head", "head": "linear"},
    },
    # the authors' end-to-end model, exactly as it reproduced 78.14%.
    # Families A-B ran with deterministic kernels; from C on they are off
    # (1.5-1.7x faster, 2026-10-27). Each row records its own mode
    "mimic_author": {
        "config": os.path.join("configs", "mimic_author.yaml"),
        "set": {},
    },
    # the same training protocol around a 3D model
    "video3d": {
        "config": os.path.join("configs", "mimic_author.yaml"),
        "set": {"method": "video3d"},
    },
    # VideoMAE, a plain ViT: B01's lr 1e-4 wiped its pretraining within one
    # epoch (val 14%, where its frozen features gave 61.7%). The VideoMAE
    # paper's fine-tuning rate, scaled to batch 4, is ~1e-5. Batch 4 fills
    # the card, so 2 x accum 2, which is exact with no BatchNorm (23.3)
    "video3d_vit": {
        "config": os.path.join("configs", "mimic_author.yaml"),
        "set": {"method": "video3d", "lr": 1e-5, "batch_size": 2,
                "accum": 2},
    },
    # Day 23's protocol around r3d_18, on motion instead of colour (32.2)
    "motion3d": {
        "config": os.path.join("configs", "mimic_author.yaml"),
        "set": {"method": "motion", "backbone": "r3d_18"},
    },
    # finished runs' probabilities, averaged; nothing trains (29.1)
    "fusion": {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "fusion", "flip": 0.0},
    },
    # the pose cache beside the RGB avg cache, one head over both (29.2)
    "pose_rgb": {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "pose_rgb", "features": "data/features/pose",
                "rgb_features": "data/features/effnetv2s"},
    },
    # their training notebook, read line by line (7.7)
    "authors_port": {
        "config": os.path.join("configs", "authors_port.yaml"),
        "set": {},
    },
}


def _cache(name):
    return "data/features/" + name


ENTRIES = [
    # ---- P: one Linear on each of the five caches built in 8.10 ----
    {"id": "P01_effnetv2s", "base": "frozen_linear",
     "cfg": {"features": _cache("effnetv2s"), "view": "avg"},
     "why": "how much shot information do ImageNet EfficientNetV2-S features "
            "hold? Also D01's probe fitted by gradient descent; the two "
            "should agree within a point"},
    {"id": "P02_convnext_t", "base": "frozen_linear",
     "cfg": {"features": _cache("convnext_tiny"), "view": "avg"},
     "why": "does a modern ImageNet CNN carry more than EfficientNetV2-S?"},
    {"id": "P03_r3d_18", "base": "frozen_linear",
     "cfg": {"features": _cache("r3d_18"), "view": "video"},
     "why": "does a small Kinetics video model beat both ImageNet caches?"},
    {"id": "P04_videomae", "base": "frozen_linear",
     "cfg": {"features": _cache("videomae_base"), "view": "video"},
     "why": "does a large Kinetics video transformer beat the small one?"},
    {"id": "P05_clip_b16", "base": "frozen_linear",
     "cfg": {"features": _cache("clip_vit_b16"), "view": "avg"},
     "why": "do features learned from 400M captioned photos rival the "
            "Kinetics ones?"},

    # ---- A: the temporal head, with everything else fixed ----
    {"id": "A01_mean", "base": "cached_head", "cfg": {"head": "mean"},
     "why": "control: if mean pooling ties GRU-128, frame order does not "
            "matter, and that is a finding"},
    {"id": "A02_max", "base": "cached_head", "cfg": {"head": "max"},
     "why": "does one decisive frame carry the class?"},
    {"id": "A03_gru128", "base": "cached_head", "cfg": {},
     "why": "the authors' head, and the reference for this family"},
    {"id": "A04_gru64", "base": "cached_head", "cfg": {"hidden": 64},
     "why": "does half the GRU lose anything? The authors report 85.52%"},
    {"id": "A05_bigru128", "base": "cached_head",
     "cfg": {"bidirectional": True},
     "why": "does reading the follow-through first help?"},
    {"id": "A06_lstm64", "base": "cached_head",
     "cfg": {"head": "lstm", "hidden": 64},
     "why": "is an LSTM better than a GRU here? The authors report 85.42%"},
    {"id": "A07_bilstm64", "base": "cached_head",
     "cfg": {"head": "lstm", "hidden": 64, "bidirectional": True},
     "why": "the authors' second best at 87.65%; does it hold up?"},
    {"id": "A08_tf2l4h", "base": "cached_head",
     "cfg": {"head": "transformer", "layers": 2, "attn_heads": 4},
     "why": "does letting frame 1 attend straight to frame 15 help?"},
    {"id": "A09_tf4l8h", "base": "cached_head",
     "cfg": {"head": "transformer", "layers": 4, "attn_heads": 8},
     "why": "does more attention capacity help on only 15 frames?"},
    {"id": "A10_tcn", "base": "cached_head", "cfg": {"head": "tcn"},
     "why": "is a dilated 1-D convolution as good as attention, and cheaper?"},
    {"id": "A11_attnpool", "base": "cached_head", "cfg": {"head": "attnpool"},
     "why": "which frames matter? The weights make a figure for free"},
    {"id": "A12_netvlad", "base": "cached_head", "cfg": {"head": "netvlad"},
     "why": "does learned aggregation, strong in video retrieval, help?"},
    {"id": "A13_ln_gru128", "base": "cached_head",
     "cfg": {"feature_norm": "layer"},
     "why": "39% of the GRU is dead at epoch 1 on flat features; does "
            "centring the input bring it back?"},
    {"id": "A14_gru128_avg", "base": "cached_head", "cfg": {"view": "avg"},
     "why": "1,280 averaged features or 62,720 with the grid kept: which "
            "wins, the information or the unsaturated GRU?"},

    # ---- D: classical models on the same cache ----
    {"id": "D01_linear", "base": "cached_classical", "cfg": {},
     "why": "a lower bound on how separable the features already are"},
    {"id": "D02_svm", "base": "cached_classical", "cfg": {"head": "svm"},
     "why": "does an RBF SVM, strong on small high-dimensional data, win?"},
    {"id": "D03_rf", "base": "cached_classical", "cfg": {"head": "rf"},
     "why": "does a random forest do well, and which features does it use?"},
    {"id": "D04_xgb", "base": "cached_classical", "cfg": {"head": "xgb"},
     "why": "is gradient boosting the best classical option here too?"},
    {"id": "D05_knn", "base": "cached_classical", "cfg": {"head": "knn"},
     "why": "are clips of the same shot neighbours in feature space?"},
    {"id": "D06_mlp", "base": "cached_classical", "cfg": {"head": "mlp"},
     "why": "does a non-linear head without recurrence beat the linear one?"},

    # ---- V: frozen features from other pretraining, one MLP head ----
    {"id": "V01_r3d_18", "base": "frozen_mlp",
     "cfg": {"features": _cache("r3d_18"), "view": "video"},
     "why": "does motion pretraining (Kinetics) beat ImageNet photos?"},
    {"id": "V02_mc3_18", "base": "frozen_mlp",
     "cfg": {"features": _cache("mc3_18"), "view": "video"},
     "why": "mixed 2D/3D convolution against full 3D"},
    {"id": "V03_r2plus1d_18", "base": "frozen_mlp",
     "cfg": {"features": _cache("r2plus1d_18"), "view": "video"},
     "why": "does splitting space and time in each layer help?"},
    {"id": "V04_s3d", "base": "frozen_mlp",
     "cfg": {"features": _cache("s3d"), "view": "video"},
     "why": "a separable 3D Inception: do cheaper convolutions keep the "
            "signal?"},
    {"id": "V05_swin3d_t", "base": "frozen_mlp",
     "cfg": {"features": _cache("swin3d_t"), "view": "video"},
     "why": "does a video transformer beat the 3D CNNs, frozen?"},
    {"id": "V06_mvitv2_s", "base": "frozen_mlp",
     "cfg": {"features": _cache("mvit_v2_s"), "view": "video"},
     "why": "multiscale attention against windowed attention"},
    {"id": "V07_videomae", "base": "frozen_mlp",
     "cfg": {"features": _cache("videomae_base"), "view": "video"},
     "why": "masked-video pretraining: the strongest frozen candidate?"},
    {"id": "V08_timesformer", "base": "frozen_mlp",
     "cfg": {"features": _cache("timesformer_base"), "view": "video"},
     "why": "divided space-time attention against VideoMAE's joint one"},
    {"id": "V09_clip_b16", "base": "frozen_mlp",
     "cfg": {"features": _cache("clip_vit_b16"), "view": "avg"},
     "why": "do caption-trained image features need motion pretraining at "
            "all?"},
    {"id": "V10_dinov2_b", "base": "frozen_mlp",
     "cfg": {"features": _cache("dinov2_base"), "view": "avg"},
     "why": "self-supervised image features with no labels: how far do "
            "they go?"},

    # ---- B: the spatial encoder, GRU-128 head fixed, end to end ----
    {"id": "B01_effnetv2s", "base": "mimic_author", "cfg": {},
     "why": "the authors' backbone; the author row is the 78.14% "
            "reproduction, the grouped row has never been measured"},
    {"id": "B02_effnetv2b3", "base": "mimic_author",
     "cfg": {"backbone": "tf_efficientnetv2_b3.in1k"},
     "why": "does two-thirds the size cost anything? The authors report "
            "88.39%"},
    {"id": "B03_xception", "base": "mimic_author",
     "cfg": {"backbone": "legacy_xception.tf_in1k"},
     "why": "the authors report 88.64%; does it reproduce?"},
    {"id": "B04_convnext_t", "base": "mimic_author",
     "cfg": {"backbone": "convnext_tiny"},
     "why": "does a modern conv design help once it is fine-tuned?"},
    {"id": "B05_mobilenetv3_l", "base": "mimic_author",
     "cfg": {"backbone": "mobilenet_v3_large"},
     "why": "how much accuracy does a 5.4M edge model give up?"},
    {"id": "B06_resnet50", "base": "mimic_author",
     "cfg": {"backbone": "resnet50"},
     "why": "where does the baseline everyone knows land?"},
    # batch 4 spilled out of the card's memory (17.2), and batch 2 broke the
    # head: BatchNorm over two clips turns every feature into +-1. So batch
    # 4 like the rest, with the backbone's stages recomputed in backward
    {"id": "B07_regnety_16gf", "base": "mimic_author",
     "cfg": {"backbone": "regnet_y_16gf", "grad_checkpoint": True},
     "why": "does raw capacity (83M) help on 10k clips?"},
    {"id": "B08_swin_t", "base": "mimic_author",
     "cfg": {"backbone": "swin_t"},
     "why": "a vision transformer as the per-frame encoder"},

    # ---- C: joint space-time models, fine-tuned ----
    {"id": "C01_r2plus1d_18", "base": "video3d",
     "cfg": {"backbone": "r2plus1d_18"},
     "why": "factorised 3D convolution: strong and cheap"},
    {"id": "C02_r3d_18", "base": "video3d", "cfg": {"backbone": "r3d_18"},
     "why": "the classic 3D CNN; against V01 it prices fine-tuning"},
    {"id": "C03_i3d", "base": "video3d", "cfg": {"backbone": "i3d_r50"},
     "why": "the standard video baseline"},
    {"id": "C04_slowfast_r50", "base": "video3d",
     "cfg": {"backbone": "slowfast_r50"},
     "why": "do two pathways at two frame rates help?"},
    {"id": "C05_x3d_m", "base": "video3d", "cfg": {"backbone": "x3d_m"},
     "why": "the most accuracy per FLOP among 3D models?"},
    {"id": "C06_mvitv2_s", "base": "video3d",
     "cfg": {"backbone": "mvit_v2_s"},
     "why": "a multiscale vision transformer, fine-tuned"},
    {"id": "C07_swin3d_t", "base": "video3d",
     "cfg": {"backbone": "swin3d_t"},
     "why": "Video Swin, the authors' own suggestion for future work"},
    # lr 1e-5 and batch 2 x 2 come from its base; see video3d_vit
    {"id": "C08_videomae", "base": "video3d_vit",
     "cfg": {"backbone": "videomae_base"},
     "why": "self-supervised video pretraining: the strongest candidate "
            "to pass 89%"},

    # ---- R: the authors' method as their notebooks run it (7.7) ----
    # a 2 x 2 with B01: protocol (B01's or theirs) x frames (box or theirs).
    # The first runs without deterministic kernels (2026-10-25)
    {"id": "R01_port_replica", "base": "authors_port",
     "cfg": {"variant": "replica", "same_clips_as": "box"},
     "why": "their notebook's method on frames made their way: the closest "
            "anyone can get to their 89%"},
    {"id": "R02_port_box", "base": "authors_port", "cfg": {},
     "why": "their notebook's method on our crops: what the protocol alone "
            "changes against B01"},
    {"id": "R03_mimic_replica", "base": "mimic_author",
     "cfg": {"variant": "replica", "same_clips_as": "box"},
     "why": "B01's protocol on frames made their way: what the "
            "preprocessing alone changes"},

    # ---- E: 33 body points a frame instead of pixels (Day 28) ----
    {"id": "E01_pose_gru", "base": "cached_head",
     "cfg": {"features": _cache("pose"), "view": "pose"},
     "why": "pose against RGB with nothing else changed: E01 against A03"},
    {"id": "E02_stgcn", "base": "cached_head",
     "cfg": {"features": _cache("pose"), "view": "pose", "head": "stgcn"},
     "why": "joints as graph nodes, edges along the skeleton: the GNN"},
    # the better of E01 and E02 is picked on val, per split
    {"id": "E03_fuse_rgb_pose", "base": "fusion",
     "cfg": {"fuse": [["A09_tf4l8h"], ["E01_pose_gru", "E02_stgcn"]]},
     "why": "the best RGB head and the best pose head, answers averaged: "
            "does each stream know something the other does not?"},
    {"id": "E04_xattn_rgb_pose", "base": "pose_rgb", "cfg": {},
     "why": "pose frames attend to RGB frames: deeper than averaging two "
            "answers"},

    # ---- W: the bat and ball tracks, from the authors' detectors (Day 30) ----
    {"id": "W02_bat", "base": "cached_head",
     "cfg": {"features": _cache("tracks"), "view": "bat"},
     "why": "the bat's path, angle and speed, 10 numbers a frame: does the "
            "bat alone tell a pull from a hook?"},
    {"id": "W03_ball", "base": "cached_head",
     "cfg": {"features": _cache("tracks"), "view": "ball"},
     "why": "the ball's path, 5 numbers a frame; capped by finding the ball "
            "in ~40% of frames"},
    {"id": "W04_bat_ball_pose", "base": "cached_head",
     "cfg": {"features": _cache("tracks"), "view": "bat_ball_pose"},
     "why": "bat, ball and body together, 114 numbers a frame: a physical "
            "description of the stroke"},
    # motion into r3d_18, Day 23's protocol (Day 32)
    {"id": "W01_flow_r3d", "base": "motion3d",
     "cfg": {"motion": "flow", "features": _cache("flow")},
     "why": "the classic motion stream: RAFT optical flow into r3d_18"},
    {"id": "W05_framediff_r3d", "base": "motion3d", "cfg": {"motion": "diff"},
     "why": "the cheapest motion signal, frame differences: W01's control"},

    # ---- F: multi-scale, hybrid and two-stream (Days 32-36) ----
    {"id": "F03_two_stream", "base": "fusion",
     "cfg": {"fuse": [["C02_r3d_18"], ["W01_flow_r3d"]]},
     "why": "the classic two-stream recipe: RGB r3d_18 and flow r3d_18, "
            "answers averaged. Does motion add to an RGB 3D CNN?"},
]

_ID = re.compile(r"^([A-Z])\d\d_[a-z0-9_]+$")


def family_of(exp_id):
    match = _ID.match(exp_id)
    if not match:
        raise ValueError(f"bad experiment id {exp_id!r}")
    return match.group(1)


def entries(family=None):
    if family is None:
        return list(ENTRIES)
    chosen = []
    for entry in ENTRIES:
        if family_of(entry["id"]) == family:
            chosen.append(entry)
    return chosen


def get(exp_id):
    for entry in ENTRIES:
        if entry["id"] == exp_id:
            return entry
    raise KeyError(f"no experiment {exp_id!r}; see --list")


def run_name(exp_id, split):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; have {sorted(SPLITS)}")
    return f"{exp_id}__{split}"


def resolve(entry, split, overrides=None):
    """The full config one run trains with: defaults, base, delta, then
    the name and split the harness owns. `overrides` is for tests."""
    base = BASES[entry["base"]]
    cfg = dict(DEFAULTS)
    cfg.update(load_config(base["config"]))
    cfg.update(base["set"])
    cfg.update(entry["cfg"])
    cfg["name"] = run_name(entry["id"], split)
    cfg["split"] = SPLITS[split]
    if overrides:
        cfg.update(overrides)
    return cfg


def config_differences(old, new):
    """Keys where an older run's config does not match `new`.

    A key the old run did not have yet must sit at its default, which is
    the behaviour the old code had.

    `deterministic` is skipped: it picks which kernels run, not what the
    experiment is, and each row records how its own run went."""
    differ = []
    for key in sorted(new):
        if key in ("name", "figures", "deterministic"):
            continue
        if key in old:
            if old[key] != new[key]:
                differ.append(key)
        elif new[key] != DEFAULTS.get(key):
            differ.append(key)
    return differ


def _method(cfg):
    try:
        return methods.load(cfg["method"])
    except ValueError:
        return None


def unread_keys(entry):
    """Delta keys the entry's method would ignore. None if the method is
    not built yet, so there is nothing to check against."""
    cfg = resolve(entry, "author")
    method = _method(cfg)
    if method is None:
        return None
    reads = set(LOOP_READS)
    for key in getattr(method, "READS", ()):
        reads.add(key)
    unread = []
    for key in entry["cfg"]:
        if key not in reads:
            unread.append(key)
    return unread


def status(entry, overrides=None):
    """'ready', or the first thing standing in the way."""
    cfg = resolve(entry, "author", overrides)
    method = _method(cfg)
    if method is None:
        return f"needs method {cfg['method']}"
    unread = unread_keys(entry)
    if unread:
        return f"{cfg['method']} ignores {', '.join(unread)}"
    if cfg["features"] and not os.path.isdir(cfg["features"]):
        return f"needs cache {os.path.basename(cfg['features'])}"
    check = getattr(method, "check", None)
    if check is not None:
        reason = check(cfg)
        if reason:
            return reason
    return "ready"
