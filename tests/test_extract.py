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


# --- 8.8: the pooled view, derived on the CPU --------------------------------

def test_pool_grid_averages_the_spatial_axes_and_nothing_else():
    """dim=(1, 2) would give (T, H) and still save without complaint."""
    from src.extract import pool_grid

    grid = torch.zeros(15, 4, 7, 7, dtype=torch.float16)
    for c in range(4):
        grid[:, c] = float(c)                      # channel c is all c
    out = pool_grid(grid)
    assert out.shape == (15, 4)
    assert out.dtype == torch.float16
    assert out[0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_pool_grid_is_the_float32_mean_stored_as_float16():
    from src.extract import pool_grid

    torch.manual_seed(0)
    grid = (torch.rand(15, 1280, 7, 7) * 30).half()
    want = grid.float().mean(dim=(-1, -2)).half()
    assert torch.equal(pool_grid(grid), want)


def test_pool_grid_refuses_a_tensor_that_is_not_a_grid():
    from src.extract import pool_grid

    with pytest.raises(ValueError, match="expected"):
        pool_grid(torch.zeros(15, 1280, dtype=torch.float16))


@needs_clips
def test_derive_makes_both_halves_and_does_not_redo_them(tmp_path):
    from src.extract import derive, manifest_path

    out = str(tmp_path / "cache")
    extract(limit=2, out_dir=out, batch_size=2, quiet=True)
    extract(limit=2, out_dir=out, batch_size=2, mirror=True, quiet=True)
    assert derive(out_dir=out, quiet=True) == 0

    rel = clip_list(PROCESSED_BOX)[0]
    plain = torch.load(out_path(out, "avg", rel))
    mirrored = torch.load(out_path(out, "avg_mirror", rel))
    grid = torch.load(out_path(out, "grid", rel))
    assert plain.shape == (15, 1280)
    assert not torch.equal(plain, mirrored)
    assert torch.equal(plain, grid.float().mean(dim=(-1, -2)).half())
    # one manifest per output, or the second half reads as already done
    assert os.path.exists(manifest_path(out, "avg"))
    assert os.path.exists(manifest_path(out, "avg_mirror"))

    stamp = os.path.getmtime(out_path(out, "avg", rel))
    assert derive(out_dir=out, quiet=True) == 0
    assert os.path.getmtime(out_path(out, "avg", rel)) == stamp


needs_avg = pytest.mark.skipif(
    not os.path.isdir(os.path.join(CACHE, "avg")),
    reason="no avg view; run src.extract --derive avg")


@needs_avg
def test_the_avg_view_reads_back_1280_wide_and_matches_the_grid():
    feats = FeatureDataset(SPLIT, view="avg")
    grid = FeatureDataset(SPLIT, view="grid")
    a, label_a = feats[0]
    g, label_g = grid[0]
    assert a.shape == (15, 1280)
    assert feats.feature_dim == 1280
    assert label_a == label_g
    gap = (a - g.mean(dim=(-1, -2))).abs().max()
    assert float(gap) < 5e-3                     # fp16 storage, nothing more
    assert len(feats) == len(grid)


@needs_avg
def test_the_avg_view_can_flip():
    feats = FeatureDataset(SPLIT, view="avg", mirror=0.5)
    torch.manual_seed(0)
    seen = set()
    for _ in range(12):
        seen.add(float(feats[0][0].sum()))
    assert len(seen) == 2


# --- 8.10: other backbones, and the avg view straight from the encoder -------

def test_each_backbone_gets_its_own_folder_by_default():
    from src.extract import cache_dir

    assert cache_dir() == os.path.join("data", "features", "effnetv2s")
    assert cache_dir("convnext_tiny") == os.path.join("data", "features",
                                                      "convnext_tiny")
    assert cache_dir("convnext_tiny", name="cnx") == os.path.join(
        "data", "features", "cnx")
    assert cache_dir("convnext_tiny", out_dir="x") == "x"


def test_a_cache_folder_refuses_a_second_backbone_before_loading_it(tmp_path):
    """Without this a second backbone finds the manifest full and does nothing."""
    from src.extract import _hold_source

    out = str(tmp_path / "cache")
    _hold_source(out, "efficientnet_v2_s", "box")
    _hold_source(out, "efficientnet_v2_s", "box")            # same: fine
    with pytest.raises(SystemExit, match="--name"):
        _hold_source(out, "efficientnet_v2_s", "seg")
    # extract checks before it builds the model, so this costs no download
    with pytest.raises(SystemExit, match="convnext_tiny"):
        extract(backbone="convnext_tiny", pool="avg", out_dir=out, quiet=True)


@needs_clips
def test_pool_avg_writes_the_avg_view_directly_and_verifies(tmp_path):
    from src.extract import manifest_path, verify

    out = str(tmp_path / "cache")
    assert extract(limit=2, out_dir=out, pool="avg", batch_size=2,
                   quiet=True) == 0
    assert extract(limit=2, out_dir=out, pool="avg", batch_size=2,
                   mirror=True, quiet=True) == 0
    assert not os.path.isdir(os.path.join(out, "grid"))
    assert os.path.exists(manifest_path(out, "avg"))
    assert os.path.exists(manifest_path(out, "avg_mirror"))

    rel = clip_list(PROCESSED_BOX)[0]
    plain = torch.load(out_path(out, "avg", rel))
    assert plain.shape == (15, 1280)
    assert plain.dtype == torch.float16
    assert not torch.equal(plain, torch.load(out_path(out, "avg_mirror", rel)))
    _check_size(out_path(out, "avg", rel), plain)

    assert verify(pool="avg", out_dir=out, clips=2, quiet=True) == 0
    # and it can fail: a sign-flipped file breaks both checks
    torch.save(-plain, out_path(out, "avg", rel))
    assert verify(pool="avg", out_dir=out, clips=2, quiet=True) == 2


CONVNEXT = os.path.join("data", "features", "convnext_tiny")
needs_convnext = pytest.mark.skipif(
    not os.path.isdir(os.path.join(CONVNEXT, "avg")),
    reason="no ConvNeXt cache; run src.extract --backbone convnext_tiny")


@needs_convnext
def test_the_convnext_cache_is_768_wide_and_complete():
    from src.extract import manifest_path
    from src.utils.checkpoint import read_results

    ok = 0
    for row in read_results(manifest_path(CONVNEXT, "avg")):
        if row.get("ok"):
            ok += 1
    assert ok == len(clip_list(PROCESSED_BOX))
    feats = FeatureDataset(SPLIT, features=CONVNEXT, view="avg")
    assert feats.item_shape == (15, 768)
    assert feats.feature_dim == 768
    assert feats.missing == FeatureDataset(SPLIT, view="avg").missing


# --- 8.10 part 2: r3d_18, one vector per clip -----------------------------------

def _weights_cached(name):
    from torchvision.models import get_model_weights

    url = get_model_weights(name).DEFAULT.url
    return os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints",
                                       os.path.basename(url)))


needs_r3d_weights = pytest.mark.skipif(
    not _weights_cached("r3d_18"), reason="r3d_18 weights not downloaded")


def test_the_pool_is_picked_from_the_backbone():
    from src.extract import _pick_pool

    assert _pick_pool("efficientnet_v2_s", None) == "flatten"
    assert _pick_pool("convnext_tiny", "avg") == "avg"
    assert _pick_pool("r3d_18", None) == "video"
    assert _pick_pool("videomae_base", None) == "video"
    with pytest.raises(ValueError, match="whole clip"):
        _pick_pool("r3d_18", "avg")
    with pytest.raises(ValueError, match="frame model"):
        _pick_pool("convnext_tiny", "video")


@needs_clips
def test_clips_are_resized_and_recoloured_for_a_video_model():
    from src.models.video_encoder import input_spec

    rel = clip_list(PROCESSED_BOX)[0]
    size, mean, std = input_spec("r3d_18")
    small = _Clips(PROCESSED_BOX, [rel], mean=mean, std=std, size=size)[0][0]
    plain = _Clips(PROCESSED_BOX, [rel])[0][0]
    assert small.shape == (15, 3, 112, 112)
    assert plain.shape == (15, 3, 224, 224)
    # the default path is untouched, or the EfficientNet cache stops matching
    frames = torch.from_numpy(np.load(os.path.join(PROCESSED_BOX, rel + ".npy")))
    x = frames.permute(0, 3, 1, 2).float().div_(255)
    m = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    assert torch.equal(plain, (x - m) / s)


@needs_clips
@needs_r3d_weights
def test_an_r3d_cache_end_to_end(tmp_path):
    from src.extract import manifest_path, verify

    out = str(tmp_path / "cache")
    assert extract(backbone="r3d_18", limit=2, out_dir=out, batch_size=2,
                   quiet=True) == 0
    assert os.path.exists(manifest_path(out, "video"))
    rel = clip_list(PROCESSED_BOX)[0]
    one = torch.load(out_path(out, "video", rel))
    assert one.shape == (512,)
    assert one.dtype == torch.float16
    _check_size(out_path(out, "video", rel), one)
    assert verify(backbone="r3d_18", out_dir=out, clips=2, quiet=True) == 0

    feats = FeatureDataset(SPLIT, features=out, view="video")
    assert feats.item_shape == (512,)
    assert feats.feature_dim == 512          # not 1, the empty product

    # the same files under the wrong view are refused, not misread
    os.rename(os.path.join(out, "video"), os.path.join(out, "avg"))
    with pytest.raises(ValueError, match="expects 2-D"):
        FeatureDataset(SPLIT, features=out, view="avg")


R3D = os.path.join("data", "features", "r3d_18")
needs_r3d_cache = pytest.mark.skipif(
    not os.path.isdir(os.path.join(R3D, "video")),
    reason="no r3d_18 cache; run src.extract --backbone r3d_18")


@needs_r3d_cache
def test_the_r3d_cache_is_512_per_clip_and_complete():
    from src.extract import manifest_path
    from src.utils.checkpoint import read_results

    ok = 0
    for row in read_results(manifest_path(R3D, "video")):
        if row.get("ok"):
            ok += 1
    assert ok == len(clip_list(PROCESSED_BOX))
    feats = FeatureDataset(SPLIT, features=R3D, view="video")
    assert feats.item_shape == (512,)
    assert feats.missing == FeatureDataset(SPLIT, view="avg").missing


# --- 8.10 part 3: VideoMAE, 16 frames from 15 -----------------------------------

def _mae_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache("MCG-NJU/videomae-base-finetuned-kinetics",
                                 "model.safetensors")
    return isinstance(hit, str)


@needs_clips
@pytest.mark.skipif(not _mae_cached(), reason="VideoMAE not downloaded")
def test_a_videomae_cache_end_to_end(tmp_path):
    from src.extract import manifest_path, verify

    out = str(tmp_path / "cache")
    assert extract(backbone="videomae_base", limit=2, out_dir=out,
                   batch_size=2, quiet=True) == 0
    assert os.path.exists(manifest_path(out, "video"))
    one = torch.load(out_path(out, "video", clip_list(PROCESSED_BOX)[0]))
    assert one.shape == (768,)
    assert one.dtype == torch.float16
    assert verify(backbone="videomae_base", out_dir=out, clips=2,
                  quiet=True) == 0


MAE = os.path.join("data", "features", "videomae_base")
needs_mae_cache = pytest.mark.skipif(
    not os.path.isdir(os.path.join(MAE, "video")),
    reason="no VideoMAE cache; run src.extract --backbone videomae_base")


@needs_mae_cache
def test_the_videomae_cache_is_768_per_clip_and_complete():
    from src.extract import manifest_path
    from src.utils.checkpoint import read_results

    ok = 0
    for row in read_results(manifest_path(MAE, "video")):
        if row.get("ok"):
            ok += 1
    assert ok == len(clip_list(PROCESSED_BOX))
    feats = FeatureDataset(SPLIT, features=MAE, view="video")
    assert feats.item_shape == (768,)
    assert feats.missing == FeatureDataset(SPLIT, view="avg").missing

