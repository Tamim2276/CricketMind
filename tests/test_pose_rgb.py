"""29.2: pose frames attend to RGB frames (E04)."""
import json
import os

import numpy as np
import pytest
import torch

from src import train as T
from src.methods import pose_rgb as M
from src.models.head import CrossAttnHead
from src.preprocess import pose as P

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def caches(tmp_path):
    """A tiny pose cache and a tiny RGB avg cache over the same clips, and
    one clip only the pose cache has."""
    split = tmp_path / "splits"
    split.mkdir()
    pose_dir = tmp_path / "pose"
    rgb_dir = tmp_path / "rgb"
    rng = np.random.default_rng(0)
    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(4):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            stem = clip.replace("/", os.sep)[:-4] + ".pt"
            pose = rng.normal(size=(5, 33, 3)).astype(np.float32) + label
            for leaf, value in (("pose", pose), ("pose_mirror", P.mirror(pose))):
                path = pose_dir / leaf / stem
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.from_numpy(value), str(path))
            if clip == "Pull/vid23_0.avi":
                continue                        # the RGB cache lacks this one
            # mirror half marked by its sign, so a test can see which was read
            rgb = torch.full((5, 8), 1.0 + label)
            for leaf, value in (("avg", rgb), ("avg_mirror", -rgb)):
                path = rgb_dir / leaf / stem
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(value.half(), str(path))
    for name in ("train", "val"):
        text = "clip,class,label,match\n"
        for clip, cls, label in rows:
            text += f"{clip},{cls},{label},vid{label}\n"
        (split / f"{name}.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return str(split), str(pose_dir), str(rgb_dir)


def test_each_frame_is_the_pose_then_the_rgb(caches):
    split, pose, rgb = caches
    data = M.PoseRGBDataset(os.path.join(split, "train.csv"), pose, rgb)
    x, _ = data[0]
    assert x.shape == (5, 99 + 8)
    assert data.pose_dim == 99 and data.rgb_dim == 8
    assert data.feature_dim == 107


def test_a_clip_missing_from_either_cache_is_left_out(caches):
    split, pose, rgb = caches
    data = M.PoseRGBDataset(os.path.join(split, "train.csv"), pose, rgb)
    assert len(data) == 11
    assert "Pull/vid23_0.avi" in data.missing


def test_one_coin_flips_both_halves(caches):
    split, pose, rgb = caches
    flipped = M.PoseRGBDataset(os.path.join(split, "train.csv"), pose, rgb,
                               mirror=1.0)
    plain = M.PoseRGBDataset(os.path.join(split, "train.csv"), pose, rgb)
    x, _ = flipped[0]
    y, _ = plain[0]
    # the RGB half came from avg_mirror (negative) ...
    assert torch.all(x[:, 99:] < 0)
    # ... and the pose half from pose_mirror, at the same time
    assert torch.allclose(x[:, :99].view(5, 33, 3),
                          P.mirror(y[:, :99].view(5, 33, 3)))


def test_cross_attention_gives_15_scores_and_trains():
    torch.manual_seed(0)
    head = CrossAttnHead(99, 1280, 15)
    x = torch.randn(4, 15, 99 + 1280)
    out = head(x)
    assert out.shape == (4, 15)
    w = head.last_weights
    assert w.shape == (4, 15, 15)
    assert torch.allclose(w.sum(dim=2), torch.ones(4, 15), atol=1e-5)
    before = head.rgb_proj.weight.detach().clone()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    torch.nn.functional.cross_entropy(out, torch.zeros(4).long()).backward()
    opt.step()
    # the RGB side learns: the pose frames really read from it
    assert not torch.equal(before, head.rgb_proj.weight)


def test_without_an_rgb_cache_it_says_so(tmp_path):
    cfg = dict(T.DEFAULTS)
    cfg["rgb_features"] = None
    assert "needs rgb_features" in M.check(cfg)
    cfg["rgb_features"] = str(tmp_path / "nothing")
    assert M.check(cfg).startswith("needs ")


def test_e04_trains_through_the_loop(caches, tmp_path):
    split, pose, rgb = caches
    cfg = {"name": "x", "method": "pose_rgb", "features": pose,
           "rgb_features": rgb, "split": split, "out": str(tmp_path / "exp"),
           "batch_size": 3, "epochs": 1, "patience": 99, "lr": 1e-3,
           "workers": 0, "flip": 0.5}
    row = T.train(cfg, quiet=True)
    assert row["epochs_run"] == 1
    assert row["method"] == "pose_rgb"
