"""Day 32: optical flow (W01), frame differences (W05), two-stream (F03)."""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import train as T
from src.methods import motion as M
from src.models.motion import MotionClassifier, adapt_stem
from src.preprocess import flow as W

CLASSES = ["Cover Drive", "Sweep", "Pull"]


class FakeRaft:
    """Every pixel moves `step` to the right; records the input range."""

    def __init__(self, step=4.0):
        self.step = step
        self.seen = []

    def __call__(self, first, second):
        self.seen.append((float(first.min()), float(first.max())))
        b, _, h, w = first.shape
        out = torch.zeros(b, 2, h, w)
        out[:, 0] = self.step
        return [out * 0.5, out]          # RAFT's guesses; the last one counts


# ---- the flow itself -------------------------------------------------------------

def test_shrinking_a_flow_field_shrinks_its_moves():
    flow = torch.zeros(1, 2, 224, 224)
    flow[:, 0] = 4.0
    small = W.shrink(flow)
    assert small.shape == (1, 2, 112, 112)
    assert torch.allclose(small[:, 0], torch.full((1, 112, 112), 2.0))


def test_the_mirror_flips_the_field_and_turns_right_into_left():
    flow = torch.zeros(3, 2, 8, 8)
    flow[:, 0, :, :4] = 2.0              # the left half moves right
    flow[:, 1] = 1.0
    out = W.mirror_flow(flow)
    assert torch.all(out[:, 0, :, 4:] == -2.0)   # now the right half, leftward
    assert torch.all(out[:, 0, :, :4] == 0)
    assert torch.all(out[:, 1] == 1.0)           # up and down untouched
    assert torch.equal(W.mirror_flow(out), flow)


def test_a_clip_gives_one_flow_per_pair_at_112(tmp_path):
    frames = np.random.default_rng(0).integers(0, 256, (6, 224, 224, 3),
                                               dtype=np.uint8)
    raft = FakeRaft()
    flow = W.clip_flow(raft, frames, "cpu")
    assert flow.shape == (5, 2, 112, 112)
    assert torch.allclose(flow[:, 0], torch.full((5, 112, 112), 2.0))
    low, high = raft.seen[0]
    assert -1.0 <= low and high <= 1.0           # RAFT's input range


@pytest.fixture
def clips(tmp_path):
    proc = tmp_path / "processed"
    split = tmp_path / "splits"
    split.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(3):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            path = proc / "box" / clip.replace("/", os.sep)[:-4]
            path.parent.mkdir(parents=True, exist_ok=True)
            frames = np.clip(40 + 70 * label + rng.integers(0, 20, (4, 32, 32, 3)),
                             0, 255).astype(np.uint8)
            np.save(str(path) + ".npy", frames)
    for name in ("train", "val"):
        text = "clip,class,label,match\n"
        for clip, cls, label in rows:
            text += f"{clip},{cls},{label},vid{label}\n"
        (split / f"{name}.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return str(proc), str(split), str(tmp_path / "flow")


def test_the_cache_is_fp16_resumable_and_finite(clips):
    proc, _, out = clips
    assert W.build(proc, out, limit=4, device="cpu", model=FakeRaft(),
                   quiet=True) == 0
    second = FakeRaft()
    W.build(proc, out, device="cpu", model=second, quiet=True)
    assert len(second.seen) == 5                 # only the 5 clips not done
    flow = torch.load(os.path.join(out, "flow", "Pull", "vid20_0.pt"))
    assert flow.dtype == torch.float16
    assert flow.shape == (3, 2, 112, 112)


def test_a_flow_that_is_not_finite_is_a_failure(clips):
    proc, _, out = clips
    assert W.build(proc, out, limit=2, device="cpu",
                   model=FakeRaft(step=float("nan")), quiet=True) == 2


def test_verify_finds_a_clip_that_changed(clips):
    proc, _, out = clips
    W.build(proc, out, device="cpu", model=FakeRaft(), quiet=True)
    assert W.verify(proc, out, count=5, device="cpu", model=FakeRaft()) == {}
    # the same clips, recomputed by a model that now says something else
    bad = W.verify(proc, out, count=5, device="cpu", model=FakeRaft(step=6.0))
    assert len(bad) >= 5


# ---- the motion network ------------------------------------------------------------

def test_the_first_layer_starts_from_the_mean_of_the_rgb_filters():
    torch.manual_seed(0)
    from torchvision.models import get_model
    model = get_model("r3d_18", weights=None)
    old = model.stem[0].weight.detach().clone()
    adapt_stem(model, 2)
    new = model.stem[0].weight
    assert model.stem[0].in_channels == 2
    mean = old.mean(dim=1, keepdim=True)
    assert torch.allclose(new[:, 0:1], mean) and torch.allclose(new[:, 1:2], mean)


@pytest.mark.parametrize("kind, shape", [
    ("flow", (2, 14, 2, 112, 112)),
    ("diff", (2, 15, 3, 224, 224)),
])
def test_motion_gives_15_scores_and_trains_whole(kind, shape):
    torch.manual_seed(0)
    model = MotionClassifier("r3d_18", 15, kind, pretrained=False).train()
    before = model.backbone.stem[0].weight.detach().clone()
    out = model(torch.randn(*shape))
    assert out.shape == (2, 15)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    nn.functional.cross_entropy(out, torch.zeros(2).long()).backward()
    opt.step()
    assert not torch.equal(before, model.backbone.stem[0].weight)


def test_frame_differences_flip_with_the_clip():
    """Flipping the clip before the differences is the same as flipping
    after, so ClipFlip on the box clip is an exact mirror for W05."""
    model = MotionClassifier("r3d_18", 15, "diff", pretrained=False)
    x = torch.randn(1, 4, 3, 224, 224)
    flipped_first = model.prepare(torch.flip(x, dims=[-1]))
    flipped_after = torch.flip(model.prepare(x), dims=[-1])
    assert flipped_first.shape == (1, 3, 3, 112, 112)
    assert torch.allclose(flipped_first, flipped_after, atol=1e-5)


def test_the_flow_dataset_mirrors_on_the_fly(clips):
    proc, split, out = clips
    W.build(proc, out, device="cpu", model=FakeRaft(), quiet=True)
    plain = M.FlowDataset(os.path.join(split, "train.csv"), out)
    flipped = M.FlowDataset(os.path.join(split, "train.csv"), out, mirror=1.0)
    x, _ = plain[0]
    y, _ = flipped[0]
    assert torch.allclose(y, W.mirror_flow(x))


def test_the_registry_knows_what_each_waits_for(tmp_path):
    from src.experiments import registry as R
    over = {"features": str(tmp_path / "flow")}
    assert R.status(R.get("W01_flow_r3d"), over) == "needs cache flow"
    assert R.status(R.get("W05_framediff_r3d")) == "ready"
    assert R.resolve(R.get("F03_two_stream"), "author")["fuse"] == \
        [["C02_r3d_18"], ["W01_flow_r3d"]]


def test_w05_trains_through_the_loop(clips, tmp_path):
    proc, split, _ = clips
    cfg = {"name": "w", "method": "motion", "motion": "diff",
           "backbone": "r3d_18", "pretrained": False, "split": split,
           "out": str(tmp_path / "exp"), "processed": proc, "batch_size": 3,
           "epochs": 1, "patience": 99, "lr": 1e-3, "workers": 0}
    row = T.train(cfg, quiet=True)
    assert row["epochs_run"] == 1
    assert row["method"] == "motion"
