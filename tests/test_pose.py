"""Day 28: the pose cache and the two pose heads (Family E)."""
import json
import os

import numpy as np
import pytest
import torch

from src import train as T
from src.data.features import FeatureDataset
from src.methods import cached_head as M
from src.models.head import STGCNHead
from src.preprocess import pose as P

CLASSES = ["Cover Drive", "Sweep", "Pull"]


def _raw_pose(seed, shift=0.0):
    """A made-up (33, 3) pose: x, y in 0..1, visibility 0.5..1."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((P.JOINTS, P.CHANNELS), dtype=np.float32)
    frame[:, 0] = 0.3 + 0.4 * rng.random(P.JOINTS) + shift
    frame[:, 1] = 0.2 + 0.6 * rng.random(P.JOINTS)
    frame[:, 2] = 0.5 + 0.5 * rng.random(P.JOINTS)
    return frame


def _geometric_mirror(frame):
    """What the flipped image's pose would be: x -> 1 - x, sides swapped."""
    if frame is None:
        return None
    out = frame.copy()
    out[:, 0] = 1.0 - frame[:, 0]
    for left, right in P.LEFT_RIGHT:
        out[left] = frame[right]
        out[right] = frame[left]
        out[left, 0] = 1.0 - frame[right, 0]
        out[right, 0] = 1.0 - frame[left, 0]
    return out


# ---- the skeleton ----------------------------------------------------------

def test_the_adjacency_is_symmetric_with_self_loops_and_one_piece():
    a = P.adjacency()
    assert a.shape == (33, 33)
    assert torch.equal(a, a.T)
    assert torch.all(torch.diagonal(a) == 1)
    # every joint reachable from the nose: the head is not cut off
    seen = {0}
    frontier = [0]
    while frontier:
        joint = frontier.pop()
        for other in range(33):
            if a[joint, other] and other not in seen:
                seen.add(other)
                frontier.append(other)
    assert len(seen) == 33


def test_left_right_pairs_cover_each_side_once():
    used = []
    for left, right in P.LEFT_RIGHT:
        used.append(left)
        used.append(right)
    assert len(used) == len(set(used)) == 32        # all but the nose
    assert 0 not in used


# ---- normalising -----------------------------------------------------------

def test_the_clip_is_centred_on_its_hips_and_scaled_by_its_torso():
    frames = [_raw_pose(0), _raw_pose(1), _raw_pose(2)]
    out = P.normalise(frames)
    hips = []
    for t in range(3):
        hips.append((out[t, 23, :2] + out[t, 24, :2]) / 2)
    assert np.allclose(np.mean(hips, axis=0), 0.0, atol=1e-5)
    torsos = []
    for t in range(3):
        shoulder = (out[t, 11, :2] + out[t, 12, :2]) / 2
        hip = (out[t, 23, :2] + out[t, 24, :2]) / 2
        torsos.append(np.linalg.norm(shoulder - hip))
    assert np.median(torsos) == pytest.approx(1.0, abs=1e-5)


def test_moving_inside_the_crop_survives_normalising():
    """Per clip, not per frame: a step forward must still show."""
    still = P.normalise([_raw_pose(0), _raw_pose(0)])
    stepping = P.normalise([_raw_pose(0), _raw_pose(0, shift=0.1)])
    assert np.allclose(still[0], still[1])
    assert not np.allclose(stepping[0, :, :2], stepping[1, :, :2])


def test_crop_position_and_batter_size_drop_out():
    frames = [_raw_pose(0), _raw_pose(1)]
    moved = []
    for frame in frames:
        bigger = frame.copy()
        bigger[:, :2] = 0.1 + 0.5 * frame[:, :2]    # smaller, elsewhere
        moved.append(bigger)
    assert np.allclose(P.normalise(frames), P.normalise(moved), atol=1e-5)


def test_a_frame_with_no_pose_is_all_zeros_visibility_too():
    out = P.normalise([_raw_pose(0), None, _raw_pose(1)])
    assert np.all(out[1] == 0)
    assert np.all(out[0, :, 2] > 0)                 # found frames are visible
    assert np.all(P.normalise([None, None]) == 0)


# ---- mirroring ---------------------------------------------------------------

def test_mirroring_the_cache_equals_the_pose_of_the_flipped_clip():
    """Exact: the same numbers as normalising the flipped clip's pose."""
    frames = [_raw_pose(0), None, _raw_pose(1), _raw_pose(2)]
    flipped = []
    for frame in frames:
        flipped.append(_geometric_mirror(frame))
    assert np.allclose(P.mirror(P.normalise(frames)), P.normalise(flipped),
                       atol=1e-6)


def test_mirroring_twice_gives_the_clip_back_for_arrays_and_tensors():
    pose = P.normalise([_raw_pose(0), _raw_pose(1)])
    assert np.array_equal(P.mirror(P.mirror(pose)), pose)
    as_tensor = torch.from_numpy(pose)
    assert torch.equal(P.mirror(P.mirror(as_tensor)), as_tensor)


# ---- the cache -----------------------------------------------------------------

@pytest.fixture
def clips(tmp_path):
    """Tiny box clips, and a split over them."""
    proc = tmp_path / "processed"
    split = tmp_path / "splits"
    split.mkdir()
    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(4):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            path = proc / "box" / clip.replace("/", os.sep)[:-4]
            path.parent.mkdir(parents=True, exist_ok=True)
            frames = np.full((5, 8, 8, 3), 40 + 70 * label, np.uint8)
            frames[2] = 0               # a frame the detector will miss
            np.save(str(path) + ".npy", frames)
    for name in ("train", "val"):
        text = "clip,class,label,match\n"
        for clip, cls, label in rows:
            text += f"{clip},{cls},{label},vid{label}\n"
        (split / f"{name}.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return str(proc), str(split), str(tmp_path / "pose")


class FakeDetector:
    """A pose from the frame's brightness; none for a black frame."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        level = int(frame.mean())
        if level == 0:
            return None
        return _raw_pose(level)


def test_the_cache_holds_both_halves_and_counts_what_was_found(clips):
    proc, _, out = clips
    assert P.build(FakeDetector(), proc, out, quiet=True) == 0
    rel = os.path.join("Sweep", "vid10_0.pt")
    pose = torch.load(os.path.join(out, "pose", rel))
    flipped = torch.load(os.path.join(out, "pose_mirror", rel))
    assert pose.shape == (5, 33, 3)
    assert torch.allclose(flipped, P.mirror(pose))
    assert P.coverage(out) == {"Cover Drive": (16, 20), "Pull": (16, 20),
                               "Sweep": (16, 20)}


def test_the_cache_resumes_instead_of_starting_again(clips):
    proc, _, out = clips
    P.build(FakeDetector(), proc, out, limit=5, quiet=True)
    second = FakeDetector()
    P.build(second, proc, out, quiet=True)
    assert second.calls == 7 * 5             # only the 7 clips not yet done


def test_the_pose_view_gives_99_numbers_a_frame_and_mirrors_by_file(clips):
    proc, split, out = clips
    P.build(FakeDetector(), proc, out, quiet=True)
    data = FeatureDataset(os.path.join(split, "train.csv"), features=out,
                          view="pose", mirror=1.0)
    x, _ = data[0]
    assert x.shape == (5, 99)
    assert data.feature_dim == 99
    plain = FeatureDataset(os.path.join(split, "train.csv"), features=out,
                           view="pose")
    assert torch.allclose(x.view(5, 33, 3), P.mirror(plain[0][0].view(5, 33, 3)))


# ---- the heads ---------------------------------------------------------------

def test_stgcn_gives_15_scores_and_trains():
    torch.manual_seed(0)
    head = STGCNHead(99, P.adjacency(), 15)
    x = torch.randn(4, 15, 99)
    out = head(x)
    assert out.shape == (4, 15)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    before = head.blocks[0].graph.weight.detach().clone()
    torch.nn.functional.cross_entropy(out, torch.zeros(4).long()).backward()
    opt.step()
    assert not torch.equal(before, head.blocks[0].graph.weight)


def test_stgcn_normalises_a_symmetric_adjacency():
    head = STGCNHead(99, P.adjacency(), 15)
    a = head.adjacency
    assert torch.allclose(a, a.T)
    assert torch.all(torch.diagonal(a) > 0)
    with pytest.raises(ValueError, match="not 33 joints"):
        STGCNHead(100, P.adjacency(), 15)


def test_a_joint_only_hears_from_its_neighbours_in_one_block():
    """Change the left ankle; within one graph step only joints joined to it
    can move."""
    a = STGCNHead(99, P.adjacency(), 15).adjacency
    for joint in range(33):
        joined = P.adjacency()[27, joint] > 0
        assert bool(a[27, joint] != 0) == bool(joined)


def test_stgcn_only_reads_the_pose_view():
    cfg = dict(T.DEFAULTS)
    cfg.update({"method": "cached_head", "head": "stgcn", "view": "avg"})
    assert M.check(cfg) == "stgcn reads joints; it needs the pose view"
    cfg["view"] = "pose"
    assert M.check(cfg) is None


def test_e01_and_e02_wait_for_the_pose_cache(monkeypatch, tmp_path):
    from src.experiments import registry as R
    over = {"features": str(tmp_path / "pose")}
    assert R.status(R.get("E01_pose_gru"), over) == "needs cache pose"
    assert R.status(R.get("E02_stgcn"), over) == "needs cache pose"


def test_stgcn_trains_through_the_loop(clips):
    proc, split, out = clips
    P.build(FakeDetector(), proc, out, quiet=True)
    cfg = {"name": "e", "method": "cached_head", "head": "stgcn",
           "features": out, "view": "pose", "split": split,
           "out": os.path.join(os.path.dirname(out), "exp"),
           "batch_size": 3, "epochs": 1, "patience": 99, "lr": 1e-3,
           "workers": 0}
    row = T.train(cfg, quiet=True)
    assert row["epochs_run"] == 1
    assert row["method"] == "cached_head"
