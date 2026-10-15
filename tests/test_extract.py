"""The cache is correct, and the two bugs that made it wrong cannot return.

A wrong feature cache is the worst defect available in this project: it trains
to *some* accuracy, so nothing raises, and every experiment built on it is
wrong together. These tests check tensors and bytes rather than accuracy.

Two of them are regressions for real defects, both found on 2026-10-15:

    the view bug      torch.save writes a view's whole storage, so feats[j]
                      saved all batch_size clips into every file -- 4x the
                      bytes, correct shape, correct values, no error
    the SiLU floor    the first version of the check asserted min == 0.0 on
                      the grounds that the backbone ends in a ReLU-family
                      activation. It ends in SiLU, whose minimum is -0.2785.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.dataset import CricShotDataset
from src.data.features import FEATURES as CACHE
from src.data.features import FeatureDataset
from src.extract import (SILU_FLOOR, _check_size, _Clips, clip_list,
                         encode_clip, extract, grid_shape, out_path)
from src.models.encoder import FrameEncoder
from src.utils.device import get_amp_settings, get_device

GRID = os.path.join(CACHE, "grid")
SPLIT = os.path.join("data", "splits_author", "train.csv")
PROCESSED_BOX = os.path.join("data", "processed", "box")

needs_cache = pytest.mark.skipif(
    not os.path.isdir(GRID), reason="no feature cache; run src.extract first")
needs_clips = pytest.mark.skipif(
    not os.path.isdir(PROCESSED_BOX), reason="no processed clips")


# --- a fake encoder, so the plumbing can be tested without a CNN -----------

class _Fake(nn.Module):
    """Returns a known ramp, so a reshape error is visible in the values."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.zeros(1))     # so .parameters() works

    def forward(self, x):
        b, t = x.shape[:2]
        ramp = torch.arange(self.out_dim, dtype=torch.float32)
        return ramp.expand(b, t, self.out_dim).clone()


def _amp_off():
    class Off:
        dtype = None
        device_type = "cpu"
    return Off()


# --- encode_clip ------------------------------------------------------------

def test_encode_clip_folds_a_batch_to_the_grid():
    enc = _Fake(2 * 3 * 3)
    x = torch.zeros(4, 15, 3, 8, 8)
    out = encode_clip(x, enc, torch.device("cpu"), _amp_off(), (2, 3, 3))
    assert out.shape == (4, 15, 2, 3, 3)
    assert out.dtype == torch.float16


def test_encode_clip_takes_one_clip_without_a_batch_axis():
    enc = _Fake(2 * 3 * 3)
    out = encode_clip(torch.zeros(15, 3, 8, 8), enc, torch.device("cpu"),
                      _amp_off(), (2, 3, 3))
    assert out.shape == (15, 2, 3, 3)


def test_encode_clip_is_channel_major():
    """The ramp 0,1,2,... must fill the last axis first, not the first axis.

    This is the (C, H, W) vs (H, W, C) convention. Both shapes are valid and
    only one matches what torchvision's flatten produced.
    """
    enc = _Fake(2 * 3 * 3)
    out = encode_clip(torch.zeros(15, 3, 8, 8), enc, torch.device("cpu"),
                      _amp_off(), (2, 3, 3)).float()
    # channel 0 holds 0..8, channel 1 holds 9..17
    assert out[0, 0].flatten().tolist() == list(range(9))
    assert out[0, 1].flatten().tolist() == list(range(9, 18))


# --- the view bug ----------------------------------------------------------

def test_a_saved_slice_without_clone_writes_the_whole_batch(tmp_path):
    """The defect itself, pinned. If torch ever stops doing this, this test
    fails and the .clone() in extract.py can be reconsidered."""
    # big enough that pickle's fixed ~1.5 KB header cannot hide the 4x
    batch = torch.zeros(4, 15, 128, 7, 7, dtype=torch.float16)
    one = tmp_path / "view.pt"
    torch.save(batch[0], one)
    cloned = tmp_path / "clone.pt"
    torch.save(batch[0].clone(), cloned)
    assert one.stat().st_size > 3 * cloned.stat().st_size


def test_check_size_accepts_a_clone_and_rejects_a_view(tmp_path):
    batch = torch.zeros(4, 15, 128, 7, 7, dtype=torch.float16)

    good = str(tmp_path / "good.pt")
    torch.save(batch[0].clone(), good)
    _check_size(good, batch[0])                 # must not raise

    bad = str(tmp_path / "bad.pt")
    torch.save(batch[0], bad)
    with pytest.raises(RuntimeError, match="view instead of a copy"):
        _check_size(bad, batch[0])


# --- grid_shape ------------------------------------------------------------

def test_grid_shape_checks_its_own_multiplication():
    """A backbone whose map is not square must fail, not round to something
    plausible. Faked by lying about out_dim."""
    enc = FrameEncoder("efficientnet_v2_s", pretrained=False, freeze=True,
                       pool="flatten")
    enc.out_dim = 62721                          # one more than 1280*7*7
    with pytest.raises(ValueError, match="not 1280 x n x n"):
        grid_shape(enc)


def test_grid_shape_finds_the_real_grid():
    enc = FrameEncoder("efficientnet_v2_s", pretrained=False, freeze=True,
                       pool="flatten")
    assert grid_shape(enc) == (1280, 7, 7)
    assert 1280 * 7 * 7 == enc.out_dim


# --- paths and listing -----------------------------------------------------

def test_out_path_puts_the_view_in_the_path_and_not_the_split():
    got = out_path(os.path.join("data", "features", "x"), "grid", "Pull/a")
    assert got.endswith(os.path.join("x", "grid", "Pull", "a.pt"))
    assert "train" not in got


@needs_clips
def test_clip_list_is_sorted_and_strips_the_extension():
    rels = clip_list(PROCESSED_BOX)
    assert rels == sorted(rels)
    assert rels[0] == "Cover Drive/vid100_41"
    for rel in rels[:20]:
        assert not rel.endswith(".npy")


# --- the real cache --------------------------------------------------------

@pytest.fixture(scope="module")
def encoder():
    device = get_device()
    enc = FrameEncoder("efficientnet_v2_s", pretrained=True, freeze=True,
                       pool="flatten").to(device).eval()
    return enc, device, get_amp_settings(device), grid_shape(enc)


@needs_cache
@needs_clips
def test_the_cache_is_what_encode_clip_produces(encoder):
    """Check 1. The strong one: compares the files to the function that wrote
    them, so it tests serialisation and layout rather than re-deriving the
    arithmetic and testing that instead.

    **Bit-exact**, which it only is because `encode_clip` calls
    `.contiguous()`. Without that the answer depended on `--workers` -- see
    test_the_cache_does_not_depend_on_the_worker_count.
    """
    enc, device, amp, grid = encoder
    rels = clip_list(PROCESSED_BOX)[:4]
    data = _Clips(PROCESSED_BOX, rels)

    chunk = []
    for i in range(len(rels)):
        chunk.append(data[i][0])
    fresh = encode_clip(torch.stack(chunk), enc, device, amp, grid)

    for j, rel in enumerate(rels):
        saved = torch.load(out_path(CACHE, "grid", rel), map_location="cpu")
        assert saved.shape == fresh[j].shape
        assert saved.dtype == torch.float16
        assert torch.equal(saved, fresh[j]), rel      # bit-exact, see below


@needs_cache
@needs_clips
def test_averaging_the_grid_equals_the_avg_pool_encoder(encoder):
    """Check 2. Two independent routes to the same numbers, in fp32 so the
    answer is 1e-7 rather than bf16 noise. A (7, 7, 1280) fold would come out
    around 1e-1 here."""
    enc, device, amp, grid = encoder
    pooled = FrameEncoder("efficientnet_v2_s", pretrained=True, freeze=True,
                          pool="avg").to(device).eval()
    data = _Clips(PROCESSED_BOX, clip_list(PROCESSED_BOX)[:2])

    with torch.no_grad():
        for i in range(2):
            x = data[i][0][None].to(device)
            derived = enc(x).float().unflatten(-1, grid).mean(dim=(-1, -2))
            assert float((derived - pooled(x).float()).abs().max()) < 1e-4


@needs_cache
def test_every_sampled_tensor_sits_on_the_silu_floor():
    """Check 3. Not `>= 0`: EfficientNetV2-S ends in SiLU, so a hard 0.0 would
    mean a ReLU backbone and far below -0.2785 would mean the wrong layer."""
    rels = []
    for folder, _, files in os.walk(GRID):
        for name in files:
            if name.endswith(".pt"):
                rels.append(os.path.join(folder, name))
        if len(rels) >= 12:
            break
    assert rels, "cache directory is empty"

    for path in rels[:12]:
        t = torch.load(path, map_location="cpu")
        low = float(t.min())
        assert low >= -0.28, f"{path} dips to {low}, below SiLU's floor"
        assert low < 0.0, f"{path} has min exactly 0.0 -- a ReLU backbone?"
        assert float(t.max()) < 65504.0
    assert SILU_FLOOR == pytest.approx(-0.278464556)


@needs_cache
def test_every_cached_file_is_its_own_size():
    """The view bug, caught on the real cache: one size, no spread."""
    sizes = set()
    seen = 0
    for folder, _, files in os.walk(GRID):
        for name in files:
            if name.endswith(".pt"):
                sizes.add(os.path.getsize(os.path.join(folder, name)))
                seen += 1
        if seen >= 200:
            break
    assert len(sizes) == 1, f"{len(sizes)} distinct file sizes: {sorted(sizes)}"
    assert abs(sizes.pop() - 15 * 1280 * 49 * 2) < 4096


@needs_cache
def test_the_manifest_agrees_with_the_files():
    path = os.path.join(CACHE, "manifest.jsonl")
    rows = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and json.loads(line).get("ok"):
                rows += 1
    files = 0
    for _, _, names in os.walk(GRID):
        for name in names:
            files += name.endswith(".pt")
            assert not name.endswith(".tmp")
    assert rows == files


# --- FeatureDataset --------------------------------------------------------

@needs_cache
def test_both_views_come_from_one_file_and_hold_the_same_numbers():
    flat = FeatureDataset(SPLIT, view="flat")
    grid = FeatureDataset(SPLIT, view="grid")
    a, _ = flat[0]
    b, _ = grid[0]
    assert a.shape == (15, 62720)
    assert b.shape == (15, 1280, 7, 7)
    assert torch.equal(a, b.flatten(1))          # flatten(1) is the inverse
    assert a.dtype == torch.float32


@needs_cache
def test_feature_dataset_drops_the_same_clips_as_the_video_one():
    """The property that makes a cached run comparable to the baseline."""
    video = CricShotDataset(SPLIT)
    feats = FeatureDataset(SPLIT, view="flat")
    assert len(feats) == len(video)
    assert feats.missing == video.missing
    assert feats.classes == video.classes


@needs_cache
def test_items_match_the_layout_train_py_indexes():
    """train.py's limit_clips reads dataset.items[i][1] as the label."""
    feats = FeatureDataset(SPLIT, view="flat")
    path, label, clip = feats.items[0]
    assert isinstance(label, int) and 0 <= label < len(feats.classes)
    assert path.endswith(".pt") and clip.endswith(".avi")


@needs_cache
def test_an_unknown_view_is_refused_before_any_file_is_opened():
    with pytest.raises(ValueError, match="view must be one of"):
        FeatureDataset(SPLIT, view="average")


@needs_cache
def test_the_avg_view_names_the_command_that_would_create_it():
    if os.path.isdir(os.path.join(CACHE, "avg")):
        pytest.skip("avg already derived")
    with pytest.raises(FileNotFoundError, match="--derive avg"):
        FeatureDataset(SPLIT, view="avg")


# --- extract is resumable --------------------------------------------------

@needs_clips
def test_a_second_run_over_the_same_clips_does_nothing(tmp_path):
    out = str(tmp_path / "cache")
    first = extract(limit=2, out_dir=out, batch_size=2, quiet=True)
    assert first == 0
    made = []
    for folder, _, files in os.walk(os.path.join(out, "grid")):
        for name in files:
            made.append(os.path.join(folder, name))
    assert len(made) == 2
    stamps = []
    for path in made:
        stamps.append(os.path.getmtime(path))

    assert extract(limit=2, out_dir=out, batch_size=2, quiet=True) == 0
    for path, was in zip(made, stamps):
        assert os.path.getmtime(path) == was, "a skipped clip was rewritten"


@needs_clips
def test_the_cache_does_not_depend_on_the_worker_count(tmp_path):
    """A regression for the subtlest bug of Day 8, found 2026-10-15.

    `dataset.py` builds its tensor with `.permute(0, 3, 1, 2)`, which makes it
    **channels_last**, and that memory format propagates through the
    elementwise normalisation. The DataLoader then decides it silently: at
    `--workers 0` the collated batch stays channels_last, at `--workers 4` the
    shared-memory collate rewrites it to NCHW. oneDNN picks a different kernel
    for each, so the written cache differed by **0.39** depending on a flag
    that is only supposed to affect speed.

    `.clone()` does not fix it -- clone preserves memory format. `.contiguous()`
    does.
    """
    first = str(tmp_path / "w0")
    second = str(tmp_path / "w4")
    assert extract(limit=3, out_dir=first, batch_size=3, workers=0,
                   quiet=True) == 0
    assert extract(limit=3, out_dir=second, batch_size=3, workers=2,
                   quiet=True) == 0

    rels = clip_list(PROCESSED_BOX)[:3]
    for rel in rels:
        a = torch.load(out_path(first, "grid", rel), map_location="cpu")
        b = torch.load(out_path(second, "grid", rel), map_location="cpu")
        assert torch.equal(a, b), f"{rel} differs between worker counts"


@needs_clips
def test_the_dataset_really_does_hand_over_a_channels_last_tensor():
    """The cause, pinned separately from the fix. If dataset.py ever stops
    doing this, the .contiguous() in encode_clip can be reconsidered -- and
    this test is what will say so."""
    from src.data.dataset import CricShotDataset

    video = CricShotDataset(SPLIT)
    x, _ = video[0]
    assert not x.is_contiguous(), "no longer channels_last; see encode_clip"
    assert x.is_contiguous(memory_format=torch.channels_last)
