"""Day 30: bat and ball tracks from the authors' detectors (Family W)."""
import json
import os

import numpy as np
import pytest
import torch

from src.data.features import FeatureDataset
from src.preprocess import tracks as K
from src.preprocess.detect import Detection
from src.preprocess import pose as P

SIZE = 32


def _line(points):
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for x, y in points:
        mask[y, x] = 1
    return mask


def _flip_det(det):
    """The same detection in the mirrored frame."""
    x1, y1, x2, y2 = det.box
    mask = None
    if det.mask is not None:
        mask = det.mask[:, ::-1].copy()
    return Detection(det.cls, det.conf, (SIZE - x2, y1, SIZE - x1, y2), mask)


# ---- the bat's angle ---------------------------------------------------------

def test_the_axis_of_a_line_is_its_direction_doubled():
    across = []
    down = []
    diagonal = []
    for i in range(4, 28):
        across.append((i, 16))
        down.append((16, i))
        diagonal.append((i, i))
    assert np.allclose(K.axis(_line(across)), (1, 0))
    assert np.allclose(K.axis(_line(down)), (-1, 0))      # 90 deg -> 180
    assert np.allclose(K.axis(_line(diagonal)), (0, 1))   # 45 deg -> 90


def test_a_speck_has_no_axis():
    assert K.axis(_line([(3, 3), (4, 4)])) is None
    assert K.axis(np.zeros((SIZE, SIZE), np.uint8)) is None


# ---- the bat track ---------------------------------------------------------------

def _bat_frames():
    """Three frames: a bat with a mask, a bat without one, nothing."""
    diagonal = []
    for i in range(6, 20):
        diagonal.append((i, i + 4))
    with_mask = Detection("Bat", 0.9, (6.0, 10.0, 20.0, 24.0),
                          _line(diagonal))
    boxes = [[Detection("Cricket Bat", 0.8, (6.0, 10.0, 20.0, 24.0))],
             [Detection("Cricket Bat", 0.7, (8.0, 12.0, 22.0, 26.0))],
             []]
    segs = [[Detection("Striker", 0.9, (0.0, 0.0, 30.0, 30.0)), with_mask],
            [],
            []]
    return boxes, segs


def test_a_bat_row_is_normalised_to_the_crop():
    boxes, segs = _bat_frames()
    track = K.bat_track(boxes, segs, SIZE)
    assert track.shape == (3, len(K.BAT_COLUMNS))
    assert track[0, 0] == 1 and track[0, 7] == 1          # box and mask
    assert track[0, 1] == pytest.approx(13 / SIZE)        # centre x
    assert track[0, 3] == pytest.approx(14 / SIZE)        # width
    assert track[1, 7] == 0                               # no mask: no angle
    assert np.all(track[2] == 0)                          # nothing found


def test_a_step_needs_both_frames():
    boxes, segs = _bat_frames()
    track = K.bat_track(boxes, segs, SIZE)
    assert track[1, 8] == pytest.approx(2 / SIZE)         # 13 -> 15
    assert track[0, 8] == 0 and track[2, 8] == 0


def test_mirroring_the_bat_equals_tracking_the_flipped_frames():
    boxes, segs = _bat_frames()
    flipped_boxes = []
    for frame in boxes:
        row = []
        for det in frame:
            row.append(_flip_det(det))
        flipped_boxes.append(row)
    flipped_segs = []
    for frame in segs:
        row = []
        for det in frame:
            row.append(_flip_det(det))
        flipped_segs.append(row)
    want = K.bat_track(flipped_boxes, flipped_segs, SIZE)
    got = K.mirror_bat(K.bat_track(boxes, segs, SIZE))
    assert np.allclose(got, want, atol=1e-6)
    assert np.all(got[2] == 0)                            # missing stays zeros


# ---- the ball track --------------------------------------------------------------

def _ball(x, y):
    return Detection("Ball", 0.5, (x - 2.0, y - 2.0, x + 2.0, y + 2.0))


def test_the_ball_is_resampled_to_the_nearest_frame_with_a_ball():
    raw = []
    for i in range(29):
        raw.append([])
    raw[2] = [_ball(100, 50)]
    raw[16] = [_ball(200, 60)]
    track = K.ball_track(raw, (100, 400), 15)             # a step of 2 frames
    assert track.shape == (15, len(K.BALL_COLUMNS))
    found = []
    for p in range(15):
        if track[p, 0]:
            found.append(p)
    assert found == [1, 8]                                # frames 2 and 16
    assert track[1, 1] == pytest.approx(100 / 400)
    assert track[1, 2] == pytest.approx(50 / 100)


def test_mirroring_the_ball_equals_tracking_the_flipped_frames():
    raw = []
    for i in range(20):
        if i % 3 == 0:
            raw.append([_ball(40 + 10 * i, 30)])
        else:
            raw.append([])
    flipped = []
    for frame in raw:
        row = []
        for det in frame:
            x1, y1, x2, y2 = det.box
            row.append(Detection("Ball", det.conf, (400 - x2, y1, 400 - x1, y2)))
        flipped.append(row)
    want = K.ball_track(flipped, (100, 400), 15)
    got = K.mirror_ball(K.ball_track(raw, (100, 400), 15))
    assert np.allclose(got, want, atol=1e-6)


def test_mirroring_twice_gives_the_tracks_back_for_tensors_too():
    boxes, segs = _bat_frames()
    bat = K.bat_track(boxes, segs, SIZE)
    assert np.allclose(K.mirror_bat(K.mirror_bat(bat)), bat)
    as_tensor = torch.from_numpy(bat)
    assert torch.allclose(K.mirror_bat(K.mirror_bat(as_tensor)), as_tensor)


# ---- the cache -------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    """Box clips, raw clips, a pose cache, and fake detectors."""
    import src.data.video as video
    import src.preprocess.detect as detect_module

    proc = tmp_path / "processed"
    pose_dir = tmp_path / "pose"
    split = tmp_path / "splits"
    split.mkdir()
    classes = ["Pull", "Hook"]
    rows = []
    for label, cls in enumerate(classes):
        for k in range(3):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            stem = clip.replace("/", os.sep)[:-4]
            path = proc / "box" / stem
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(path) + ".npy", np.full((5, SIZE, SIZE, 3), 50 + label,
                                                np.uint8))
            pose = np.random.default_rng(k).normal(size=(5, 33, 3))
            pose = pose.astype(np.float32)
            for leaf, value in (("pose", pose), ("pose_mirror", P.mirror(pose))):
                out = pose_dir / leaf / (stem + ".pt")
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.from_numpy(value), str(out))
    text = "clip,class,label,match\n"
    for clip, cls, label in rows:
        text += f"{clip},{cls},{label},vid{label}\n"
    (split / "train.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(classes), encoding="utf-8")

    def fake_load(path):
        return os.path.basename(path)

    def fake_detect(frames, model=None, **kwargs):
        out = []
        for i in range(len(frames)):
            if model.startswith("Bat"):
                out.append([Detection("Cricket Bat", 0.9,
                                      (4.0 + i, 6.0, 14.0 + i, 20.0))])
            elif model.startswith("Ball"):
                if i % 2 == 0:
                    out.append([_ball(10 + 5 * i, 20)])
                else:
                    out.append([])
            else:
                out.append([])
        return out

    def fake_read(path, rgb=True, retry=True):
        return np.zeros((9, 60, 120, 3), np.uint8)

    monkeypatch.setattr(detect_module, "load_detector", fake_load)
    monkeypatch.setattr(detect_module, "detect", fake_detect)
    monkeypatch.setattr(video, "read_clip", fake_read)
    return str(proc), str(pose_dir), str(split), str(tmp_path / "tracks")


def test_the_cache_holds_three_tracks_and_their_mirrors(world):
    proc, pose_dir, _, out = world
    assert K.build(proc, "raw", out, pose_dir=pose_dir, quiet=True) == 0
    rel = os.path.join("Pull", "vid00_0.pt")
    bat = torch.load(os.path.join(out, "bat", rel))
    ball = torch.load(os.path.join(out, "ball", rel))
    joined = torch.load(os.path.join(out, "bat_ball_pose", rel))
    joined_mirror = torch.load(os.path.join(out, "bat_ball_pose_mirror", rel))
    assert bat.shape == (5, 10) and ball.shape == (5, 5)
    assert joined.shape == (5, 10 + 5 + 99)
    pose_mirror = torch.load(os.path.join(pose_dir, "pose_mirror",
                                          "Pull", "vid00_0.pt"))
    want = torch.cat([K.mirror_bat(bat), K.mirror_ball(ball),
                      pose_mirror.flatten(1)], dim=1)
    assert torch.allclose(joined_mirror, want)
    found = K.coverage(out)
    assert found["Pull"]["bat"] == 15 and found["Pull"]["frames"] == 15


def test_the_cache_resumes(world):
    proc, pose_dir, _, out = world
    K.build(proc, "raw", out, pose_dir=pose_dir, limit=2, quiet=True)
    K.build(proc, "raw", out, pose_dir=pose_dir, quiet=True)
    rows = 0
    with open(os.path.join(out, "manifest.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows += 1
    assert rows == 6                       # each clip once, none twice


def test_the_track_views_read_as_frames(world):
    proc, pose_dir, split, out = world
    K.build(proc, "raw", out, pose_dir=pose_dir, quiet=True)
    for view, width in (("bat", 10), ("ball", 5), ("bat_ball_pose", 114)):
        data = FeatureDataset(os.path.join(split, "train.csv"), features=out,
                              view=view, mirror=0.5)
        assert data[0][0].shape == (5, width)
        assert data.feature_dim == width


def test_w02_to_w04_wait_for_the_tracks(tmp_path):
    from src.experiments import registry as R
    over = {"features": str(tmp_path / "tracks")}
    for exp_id in ("W02_bat", "W03_ball", "W04_bat_ball_pose"):
        assert R.status(R.get(exp_id), over) == "needs cache tracks"
