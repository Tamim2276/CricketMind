"""Day 42's Grad-CAM: the maps must come from the right place and land on the
right frames. Real MViT and F02-shaped models, random weights, on the CPU.
"""
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.model import build_model
from src.models.video_classifier import VideoClassifier
from tools import gradcam as G

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def mvit():
    torch.manual_seed(0)
    return VideoClassifier("mvit_v2_s", 3, pretrained=False)


@pytest.fixture(scope="module")
def f02():
    torch.manual_seed(0)
    return build_model(num_classes=3, backbone="efficientnet_v2_s",
                       pretrained=False, scales="multi", head="transformer",
                       layers=2, pool="flatten", dense=64)


def _clip(frames=15, size=224):
    torch.manual_seed(1)
    return torch.randn(frames, 3, size, size)


# ---- the maths --------------------------------------------------------------

def test_frame_maps_weigh_each_frame_by_its_own_gradients():
    maps = torch.zeros(2, 2, 2, 2)
    maps[0, 0, 0, 0] = 1.0     # frame 1: channel 0, top left
    maps[1, 1, 1, 1] = 1.0     # frame 2: channel 1, bottom right
    grads = torch.zeros(2, 2, 2, 2)
    grads[0, 0] = 1.0          # frame 1 cares about channel 0
    grads[1, 1] = -1.0         # frame 2's channel 1 pushes the answer down
    cam = G.cam_of(maps, grads, "frames", frames=2, size=2)
    assert cam.shape == (2, 2, 2)
    assert cam[0, 0, 0] > 0
    assert cam[1].max() == 0   # only the part that pushes up is kept


def test_mvit_maps_drop_the_class_token_and_find_their_place():
    # 1 class token, then (t=2, h=2, w=2) tokens, 3 channels
    maps = torch.zeros(1, 9, 3)
    maps[0, 0] = 100.0         # the class token must not show anywhere
    maps[0, 1 + 6] = 1.0       # t=1, h=1, w=0
    grads = torch.ones(1, 9, 3)
    cam = G.cam_of(maps, grads, "mvit", frames=2, size=2, thw=(2, 2, 2))
    assert cam.shape == (2, 2, 2)
    assert cam.argmax() == np.ravel_multi_index((1, 1, 0), (2, 2, 2))


def test_an_unknown_model_fails_loudly():
    with pytest.raises(SystemExit, match="knows MViT"):
        G.target_layer(nn.Linear(2, 2))


# ---- real models ------------------------------------------------------------

def _check(model, label):
    probs, cams = G.grad_cam(model, _clip(), CPU, label)
    guess = int(np.argmax(probs))
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert guess in cams
    for target in cams:
        assert cams[target].shape == (15, 224, 224)
        assert np.isfinite(cams[target]).all()
        assert cams[target].min() >= 0
    peak = 0.0
    for target in cams:
        peak = max(peak, float(cams[target].max()))
    assert peak == pytest.approx(1.0)
    for weight in model.parameters():
        assert weight.grad is None
    return guess, cams


def test_mvit_gives_a_map_per_stored_frame(mvit):
    _check(mvit, None)


def test_a_mistake_gets_a_map_for_its_true_shot_too(mvit):
    probs, _ = G.grad_cam(mvit, _clip(), CPU)
    wrong = (int(np.argmax(probs)) + 1) % 3
    guess, cams = _check(mvit, wrong)
    assert sorted(cams) == sorted([guess, wrong])


def test_f02_gives_a_map_per_frame(f02):
    _check(f02, None)


def test_no_hook_is_left_behind(mvit):
    layer, _ = G.target_layer(mvit)
    G.grad_cam(mvit, _clip(), CPU)
    assert len(layer._forward_hooks) == 0
    assert len(mvit.backbone.blocks[-1]._forward_pre_hooks) == 0


# ---- picking and drawing ----------------------------------------------------

def test_pick_takes_the_most_confident_once_per_true_class():
    probs = np.array([[0.9, 0.1, 0.0],    # right, class 0
                      [0.95, 0.05, 0.0],  # right, class 0, surer
                      [0.1, 0.8, 0.1],    # right, class 1
                      [0.7, 0.3, 0.0],    # wrong, a 1
                      [0.0, 0.99, 0.01]]) # wrong, a 2
    part = {"probs": probs, "labels": [0, 0, 1, 1, 2]}
    right, wrong = G.pick(part, count=5)
    assert right == [1, 2]
    assert wrong == [4, 3]


def test_shown_frames_spread_from_first_to_last():
    assert G._shown(15) == [0, 2, 4, 6, 8, 10, 12, 14]
    assert G._shown(4) == [0, 1, 2, 3]


def test_draw_writes_one_picture(tmp_path):
    frames = np.zeros((15, 16, 16, 3), dtype=np.uint8)
    cam = np.zeros((15, 16, 16))
    cam[:, 4, 4] = 1.0
    path = tmp_path / "01.png"
    G.draw(frames, [("the crop", None), ("for Pull", cam)], "a title", str(path))
    assert path.stat().st_size > 0


# ---- the command ------------------------------------------------------------

class _Data:
    classes = ["Pull", "Hook", "Sweep"]

    def __init__(self, tmp_path):
        self.items = []
        for i in range(2):
            path = tmp_path / f"clip{i}.npy"
            np.save(path, np.zeros((15, 16, 16, 3), dtype=np.uint8))
            self.items.append((str(path), 0, f"Pull/vid{i}_0.avi"))

    def __getitem__(self, i):
        return torch.zeros(15, 3, 16, 16), 0


def _command(tmp_path, monkeypatch):
    run = "C06_mvitv2_s__grouped"
    out = tmp_path / "experiments"
    (out / run).mkdir(parents=True)
    (out / run / "best.pt").write_bytes(b"")
    data = _Data(tmp_path)

    def restore(ckpt, split_csv, device=None):
        assert split_csv.endswith("val.csv")
        return {}, data, None, {}

    def fake_cam(model, clip, device, label=None):
        cams = {1: np.ones((15, 16, 16)), 0: np.zeros((15, 16, 16))}
        return np.array([0.2, 0.7, 0.1]), cams

    monkeypatch.setattr(G, "_restore", restore)
    monkeypatch.setattr(G, "grad_cam", fake_cam)
    return run, str(out), str(tmp_path / "figures")


def test_chosen_clips_become_pictures_and_an_index(tmp_path, monkeypatch):
    run, out, figures = _command(tmp_path, monkeypatch)
    folder = os.path.join(figures, run)
    os.makedirs(folder)
    with open(os.path.join(folder, "09.png"), "wb") as handle:
        handle.write(b"left by an earlier run")
    code = G.main(["--clips", "Pull/vid0_0.avi", "Pull/vid1_0.avi",
                   "--out", out, "--figures", figures, "--device", "cpu"])
    assert code == 0
    assert sorted(os.listdir(folder)) == ["01.png", "02.png", "index.md"]
    with open(os.path.join(folder, "index.md"), encoding="utf-8") as handle:
        index = handle.read()
    assert "Pull/vid1_0.avi" in index
    assert "the model says Hook 70%" in index


def test_a_clip_not_in_val_fails_loudly(tmp_path, monkeypatch):
    run, out, figures = _command(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="not a grouped val clip"):
        G.main(["--clips", "Pull/vid9_0.avi", "--out", out,
                "--figures", figures, "--device", "cpu"])


def test_picking_on_the_cpu_with_nothing_scored_stops(tmp_path, monkeypatch,
                                                      capsys):
    run, out, figures = _command(tmp_path, monkeypatch)
    code = G.main(["--out", out, "--figures", figures, "--device", "cpu"])
    assert code == 1
    assert "--clips" in capsys.readouterr().out


def test_a_missing_run_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        G.main(["--run", "C99_none__grouped", "--out", str(tmp_path)])
