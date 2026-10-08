"""The striker detector. Mostly guarding the two settings that silently ruin it."""
import os

import numpy as np
import pytest

from src.preprocess.detect import (
    BATCH,
    CLASSES,
    IMGSZ,
    Detection,
    MODEL_PATH,
    detect,
    load_detector,
)

R = "data/CricShoot10kShootDataset"
needs = pytest.mark.skipif(
    not (os.path.isdir(R) and os.path.exists(MODEL_PATH)),
    reason="dataset or detector not present")


def test_detection_geometry():
    d = Detection("Striker", 0.9, (10.0, 20.0, 110.0, 220.0))
    assert d.area == 100 * 200
    assert d.centre == (60.0, 120.0)


def test_imgsz_is_not_the_checkpoint_default():
    """224 is what the checkpoint says; it drops striker recall to near zero."""
    assert IMGSZ == 896
    assert BATCH <= 8


@needs
def test_the_model_knows_only_striker_and_bat():
    m = load_detector()
    assert tuple(m.names[i] for i in sorted(m.names)) == CLASSES


@needs
def test_a_single_frame_returns_a_flat_list():
    from src.data.video import read_clip
    frame = read_clip(os.path.join(R, "Sweep/vid306_8.avi"))[5]
    dets = detect(frame)
    assert isinstance(dets, list)
    assert all(isinstance(d, Detection) for d in dets)
    assert any(d.cls == "Striker" for d in dets)


@needs
def test_a_clip_returns_one_list_per_frame():
    from src.data.video import read_clip
    clip = read_clip(os.path.join(R, "Sweep/vid306_8.avi"))
    res = detect(list(clip))
    assert len(res) == len(clip)


@needs
def test_native_resolution_beats_the_checkpoint_default():
    """The whole reason IMGSZ is pinned. vid302_23 scores 0/24 at 224."""
    from src.data.video import read_clip
    clip = list(read_clip(os.path.join(R, "Cover Drive/vid302_23.avi")))
    small = detect(clip, imgsz=224)
    native = detect(clip)
    n = lambda r: sum(1 for fr in r if any(d.cls == "Striker" for d in fr))
    assert n(small) < 3
    assert n(native) > 12


@needs
def test_boxes_lie_inside_the_frame():
    from src.data.video import read_clip
    clip = read_clip(os.path.join(R, "Sweep/vid306_8.avi"))
    h, w = clip.shape[1:3]
    for fr in detect(list(clip[:8])):
        for d in fr:
            x1, y1, x2, y2 = d.box
            assert 0 <= x1 < x2 <= w + 1 and 0 <= y1 < y2 <= h + 1
            assert 0.0 < d.conf <= 1.0
