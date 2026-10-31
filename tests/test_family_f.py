"""Day 33: Family F, each B01 with one change (F01, F04, F05) or F01 with
one change (F02)."""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import train as T
from src.models.encoder import FrameEncoder
from src.models.head import GRUAttnHead, GRUHead
from src.models.model import build_model

CLASSES = ["Cover Drive", "Sweep", "Pull"]
B01_GRID = 7 * 7 * 1280            # 62,720: what B01's GRU reads


def _pair(**change):
    """B01's encoder and a changed one with the same weights, in eval."""
    torch.manual_seed(0)
    plain = FrameEncoder("efficientnet_v2_s", pretrained=False,
                         pool="flatten").eval()
    changed = FrameEncoder("efficientnet_v2_s", pretrained=False,
                           pool="flatten", **change).eval()
    changed.load_state_dict(plain.state_dict(), strict=False)
    return plain, changed


# ---- F01: the fine and mid stages stacked onto B01's grid -----------------------

def test_multi_scale_adds_the_fine_and_mid_maps_on_the_same_grid():
    _, multi = _pair(scales="multi")
    # 1,280 + 64 + 160 = 1,504 channels on the 7x7 grid
    assert multi.out_dim == 7 * 7 * (1280 + 64 + 160)


def test_multi_scale_keeps_b01s_numbers_first_and_adds_no_weights():
    plain, multi = _pair(scales="multi")
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        a = plain(x)
        b = multi(x)
    assert torch.allclose(b[..., :B01_GRID], a, atol=1e-5)
    mine = set()
    for name, _ in multi.named_parameters():
        mine.add(name)
    theirs = set()
    for name, _ in plain.named_parameters():
        theirs.add(name)
    assert mine == theirs                     # only running stats are new


def test_the_added_maps_are_normalised_to_b01s_scale():
    """Found 2026-11-01: raw, the inner stages were ~5x wider than B01's
    grid and pinned half the GRU from epoch 1."""
    _, multi = _pair(scales="multi")
    multi.train()
    x = torch.randn(2, 3, 3, 224, 224)
    with torch.no_grad():
        out = multi(x).flatten(0, 1)
    added = out[:, B01_GRID:].reshape(6, 64 + 160, 49)
    # each channel near mean 0, spread 1 over the batch (then averaged
    # over 4x4 or 2x2 cells, which can only shrink it)
    assert added.mean().abs() < 0.1
    assert 0.2 < added.std() < 1.5


def test_multi_scale_is_refused_where_it_cannot_work():
    with pytest.raises(ValueError, match="stages of"):
        FrameEncoder("resnet50", pretrained=False, pool="flatten",
                     scales="multi")
    with pytest.raises(ValueError, match="pool='flatten'"):
        FrameEncoder("efficientnet_v2_s", pretrained=False, pool="avg",
                     scales="multi")


# ---- F04: the same frame again at 112 ------------------------------------------

def test_a_second_size_joins_the_first_and_adds_no_weights():
    plain, both = _pair(extra_size=112)
    small = FrameEncoder("efficientnet_v2_s", pretrained=False,
                         pool="flatten", size=112)
    assert both.out_dim == B01_GRID + small.out_dim
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        a = plain(x)
        b = both(x)
    assert torch.allclose(b[..., :B01_GRID], a, atol=1e-5)
    assert set(both.state_dict()) == set(plain.state_dict())


# ---- F05: attention over every GRU step -------------------------------------------

def test_gru_attention_keeps_b01s_gru_and_tail():
    torch.manual_seed(0)
    plain = GRUHead(16, 15, hidden=32, dense=64)
    torch.manual_seed(0)
    attn = GRUAttnHead(16, 15, hidden=32, dense=64)
    # the same GRU and tail, drawn the same way; only the scorer is new
    for name, value in plain.state_dict().items():
        assert torch.equal(value, attn.state_dict()[name]), name
    out = attn.train()(torch.randn(4, 15, 16))
    assert out.shape == (4, 15)
    w = attn.last_weights
    assert w.shape == (4, 15)
    assert torch.allclose(w.sum(dim=1), torch.ones(4), atol=1e-5)


def test_each_head_builds_and_an_unknown_one_is_refused():
    kinds = {"gru": "GRUHead", "transformer": "TransformerHead",
             "gru_attn": "GRUAttnHead"}
    for head, cls in kinds.items():
        model = build_model(num_classes=15, backbone="resnet18",
                            pretrained=False, head=head, layers=2)
        assert type(model.head).__name__ == cls
    with pytest.raises(ValueError, match="unknown head"):
        build_model(num_classes=15, backbone="resnet18", pretrained=False,
                    head="lstm")


# ---- the registry -----------------------------------------------------------------

def test_each_f_run_is_b01_with_one_change():
    from src.experiments import registry as R
    b01 = R.resolve(R.get("B01_effnetv2s"), "author")
    expect = {"F01_multiscale": {"scales"},
              "F04_multires": {"extra_size"},
              "F05_gru_attn": {"head"}}
    for exp_id, keys in expect.items():
        cfg = R.resolve(R.get(exp_id), "author")
        differ = set()
        for key in b01:
            if key != "name" and cfg[key] != b01[key]:
                differ.add(key)
        assert differ == keys, exp_id
    f01 = R.resolve(R.get("F01_multiscale"), "author")
    f02 = R.resolve(R.get("F02_multiscale_tf"), "author")
    differ = set()
    for key in f01:
        if key != "name" and f01[key] != f02[key]:
            differ.add(key)
    assert differ == {"head", "layers"}          # F02 against F01: the head


def test_an_end_to_end_run_names_a_bad_head():
    from src.methods import video_gru
    cfg = dict(T.DEFAULTS)
    cfg.update({"head": "netvlad"})
    assert "ends in one of" in video_gru.check(cfg)


# ---- through the loop ---------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
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
            frames = np.clip(40 + 70 * label + rng.integers(0, 20, (3, 224, 224, 3)),
                             0, 255).astype(np.uint8)
            np.save(str(path) + ".npy", frames)
    for name in ("train", "val"):
        text = "clip,class,label,match\n"
        for clip, cls, label in rows:
            text += f"{clip},{cls},{label},vid{label}\n"
        (split / f"{name}.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return str(split), str(tmp_path / "exp"), str(proc)


@pytest.mark.parametrize("change", [
    {"scales": "multi"},
    {"extra_size": 112},
    {"head": "gru_attn"},
])
def test_an_f_model_trains_through_the_loop(workspace, change):
    split, out, proc = workspace
    cfg = {"name": "f", "split": split, "out": out, "processed": proc,
           "backbone": "efficientnet_v2_s", "pool": "flatten",
           "pretrained": False, "batch_size": 3, "epochs": 1, "patience": 99,
           "lr": 1e-3, "workers": 0}
    cfg.update(change)
    row = T.train(cfg, quiet=True)
    assert row["epochs_run"] == 1
