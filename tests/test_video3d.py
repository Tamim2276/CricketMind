"""Day 23: video models fine-tuned whole, with a new last layer (Family C)."""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision.models import get_model

from src import train as T
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.methods import video3d as M
from src.models.video_classifier import VideoClassifier

CLASSES = ["Cover Drive", "Sweep", "Pull"]


def _one_step(model, x, labels):
    """One Adam step: the output, and whether the first layer moved."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for p in model.parameters():
        first = p
        break
    before_first = first.detach().clone()
    out = model(x)
    nn.functional.cross_entropy(out, labels).backward()
    opt.step()
    return out, not torch.equal(before_first, first.detach())


# ---- every model: 15 scores out, and all of it trains --------------------

@pytest.mark.parametrize("name, batch, size", [
    ("r3d_18", 2, 112),
    ("r2plus1d_18", 2, 112),
    ("mvit_v2_s", 1, 224),
    ("swin3d_t", 1, 224),
])
def test_each_model_gives_15_scores_and_trains_whole(name, batch, size):
    torch.manual_seed(0)
    model = VideoClassifier(name, 15, pretrained=False).train()
    assert model.size == size
    x = torch.randn(batch, 15, 3, 224, 224)
    out, first_layer_moved = _one_step(model, x, torch.zeros(batch).long())
    assert out.shape == (batch, 15)
    # the first layer moved too: end to end, not a frozen backbone
    assert first_layer_moved


def test_only_the_last_layer_is_new():
    """Same seed, same weights everywhere but the classifier: the rest is
    the model as published."""
    torch.manual_seed(0)
    plain = get_model("r3d_18", weights=None)
    torch.manual_seed(0)
    ours = VideoClassifier("r3d_18", 15, pretrained=False)
    mine = ours.backbone.state_dict()
    for key, value in plain.state_dict().items():
        if key.startswith("fc."):
            continue
        assert torch.equal(value, mine[key]), key
    assert mine["fc.weight"].shape == (15, 512)


def test_mvit_keeps_its_dropout_and_only_its_linear_changes():
    model = VideoClassifier("mvit_v2_s", 15, pretrained=False)
    assert isinstance(model.backbone.head[0], nn.Dropout)
    assert model.backbone.head[1].out_features == 15


# ---- the clip is converted the way Family V's extraction did it ----------

def test_frames_size_and_colours_come_from_the_weights():
    r3d = VideoClassifier("r3d_18", 15, pretrained=False)
    assert (r3d.frames, r3d.size) == (None, 112)        # any length, 112px
    assert r3d.mean != IMAGENET_MEAN                     # Kinetics colours
    mvit = VideoClassifier("mvit_v2_s", 15, pretrained=False)
    assert (mvit.frames, mvit.size) == (16, 224)         # exactly 16


def test_the_clip_is_converted_exactly_as_family_v_did(tmp_path):
    """V01 read r3d_18 features made by extract.py's _Clips; C02 must feed
    r3d_18 the same numbers, or C02 against V01 is not only fine-tuning."""
    from src.extract import _Clips

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (15, 224, 224, 3)).astype(np.uint8)
    np.save(str(tmp_path / "clip.npy"), frames)
    model = VideoClassifier("r3d_18", 15, pretrained=False)

    theirs = _Clips(str(tmp_path), ["clip"], mean=model.mean, std=model.std,
                    size=model.size)[0][0]
    # what dataset.py hands over: ImageNet colours at 224
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    ours = model.prepare(((x - mean) / std).unsqueeze(0))[0]
    assert ours.shape == (15, 3, 112, 112)
    assert torch.allclose(ours, theirs, atol=1e-4)


def test_slowfast_gets_a_fast_and_a_slow_copy_of_the_clip():
    """No pytorchvideo needed: a stand-in records what it is handed."""
    seen = []

    class Record(nn.Module):
        def forward(self, paths):
            for path in paths:
                seen.append(tuple(path.shape))
            return torch.zeros(1, 15)

    model = VideoClassifier("r3d_18", 15, pretrained=False)
    model.name = "slowfast_r50"
    model.frames = 32
    model.size = 224
    model.backbone = Record()
    model(torch.randn(1, 15, 3, 224, 224))
    assert seen == [(1, 3, 8, 224, 224), (1, 3, 32, 224, 224)]


# ---- VideoMAE --------------------------------------------------------------

def test_videomae_gets_16_frames_eager_attention_and_a_new_classifier():
    pytest.importorskip("transformers")
    model = VideoClassifier("videomae_base", 15, pretrained=False).eval()
    assert model.frames == 16
    assert model.backbone.config._attn_implementation == "eager"
    assert model.backbone.classifier.out_features == 15
    for p in model.parameters():
        assert p.requires_grad
    with torch.no_grad():
        assert model(torch.randn(1, 15, 3, 224, 224)).shape == (1, 15)


def test_c08_trains_two_at_a_time_and_has_no_batchnorm():
    """23.3: batch 4 needs the whole card. With LayerNorm only, nothing
    looks across the clips in a batch, so 2 x accum 2 is exactly batch 4.
    B07's collapse came from BatchNorm over two clips; there is none here."""
    pytest.importorskip("transformers")
    from src.experiments import registry as R

    cfg = R.resolve(R.get("C08_videomae"), "author")
    assert (cfg["batch_size"], cfg["accum"]) == (2, 2)
    model = VideoClassifier("videomae_base", 15, pretrained=False)
    for layer in model.modules():
        assert not isinstance(layer, nn.modules.batchnorm._BatchNorm)
    for entry in R.entries("C"):
        if entry["id"] == "C08_videomae":
            continue
        cfg = R.resolve(entry, "author")
        assert (cfg["batch_size"], cfg["accum"]) == (4, 1), entry["id"]


# ---- pytorchvideo, when it installs ---------------------------------------

@pytest.mark.parametrize("name", ["i3d_r50", "slowfast_r50", "x3d_m"])
def test_the_pytorchvideo_models_give_15_scores(name):
    pytest.importorskip("pytorchvideo")
    model = VideoClassifier(name, 15, pretrained=False).eval()
    assert getattr(model.backbone.blocks[-1], "activation", None) is None
    with torch.no_grad():
        assert model(torch.randn(1, 15, 3, 224, 224)).shape == (1, 15)


# ---- the method ------------------------------------------------------------

def _cfg(**over):
    cfg = dict(T.DEFAULTS)
    cfg.update({"method": "video3d", "backbone": "r3d_18"})
    cfg.update(over)
    return cfg


def test_every_key_it_reads_is_a_config_key():
    for key in M.READS:
        assert key in T.DEFAULTS, key


def _missing():
    return "needs pytorchvideo: pip install pytorchvideo"


def _not_cached(repo, filename):
    return False


def test_check_names_what_is_missing(monkeypatch):
    assert "not a video model" in M.check(_cfg(backbone="convnext_tiny"))
    monkeypatch.setattr(M, "_pytorchvideo_problem", _missing)
    assert M.check(_cfg(backbone="i3d_r50")).startswith("needs pytorchvideo")
    import src.models.encoder as encoder
    monkeypatch.setattr(encoder, "_hf_cached", _not_cached)
    told = M.check(_cfg(backbone="videomae_base"))
    assert "HF_HUB_OFFLINE=0 hf download MCG-NJU/videomae" in told


def test_a_c_run_sees_exactly_the_clips_b01_sees():
    from src.experiments import registry as R
    from src.methods import video_gru

    for split in ("author", "grouped"):
        c = M.datasets(R.resolve(R.get("C02_r3d_18"), split))
        b = video_gru.datasets(R.resolve(R.get("B01_effnetv2s"), split))
        for mine, theirs in zip(c, b):
            assert mine.items == theirs.items


@pytest.fixture
def workspace(tmp_path):
    """Tiny real clips on disk, as test_train.py makes them."""
    proc = tmp_path / "processed"
    split = tmp_path / "splits"
    split.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(4):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            p = proc / "box" / clip.replace("/", os.sep)[:-4]
            p.parent.mkdir(parents=True, exist_ok=True)
            base = 40 + 70 * label
            frames = np.clip(base + rng.integers(0, 20, (4, 16, 16, 3)),
                             0, 255).astype(np.uint8)
            np.save(str(p) + ".npy", frames)
    for name in ("train", "val"):
        text = "clip,class,label,match\n"
        for clip, cls, label in rows:
            text += f"{clip},{cls},{label},vid{label}\n"
        (split / f"{name}.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return str(split), str(tmp_path / "exp"), str(proc)


def test_a_c_model_trains_through_the_loop(workspace):
    """The whole of train.py: epochs, best.pt, the row. Nothing in the
    loop knows it is a video model."""
    split, out, proc = workspace
    cfg = {"name": "c", "method": "video3d", "backbone": "r3d_18",
           "pretrained": False, "split": split, "out": out,
           "processed": proc, "batch_size": 3, "epochs": 1, "patience": 99,
           "lr": 1e-3, "workers": 0}
    row = T.train(cfg, quiet=True)
    assert row["method"] == "video3d"
    assert row["epochs_run"] == 1
    assert os.path.exists(os.path.join(out, "c", "best.pt"))
