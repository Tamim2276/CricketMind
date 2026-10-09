"""Selection has no ground truth in the dataset, so these pin down the
behaviour the measured rule is supposed to have, not its accuracy."""
import numpy as np
import pytest

from src.preprocess.detect import Detection
from src.preprocess.striker import EDGE_PX, pick_striker, plausible

W, H = 896, 540


def person(cx, cy, w=80, h=200, conf=0.5, cls="Striker"):
    return Detection(cls, conf, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def test_picks_the_central_person():
    far = person(100, 450)
    near = person(450, 280)
    assert pick_striker([far, near], (H, W)) is near


def test_centre_beats_a_bigger_box():
    """The non-striker is often nearer the camera, so bigger. Centre wins."""
    huge = person(820, 400, w=200, h=400)
    small = person(450, 270, w=60, h=150)
    assert pick_striker([huge, small], (H, W)) is small


def test_centre_beats_higher_confidence():
    assert pick_striker([person(820, 100, conf=0.95),
                         person(450, 270, conf=0.26)], (H, W)).conf == 0.26


def test_bats_are_not_candidates():
    bat = person(450, 270, w=20, h=90, cls="Bat")
    batter = person(300, 300)
    assert pick_striker([bat, batter], (H, W)) is batter


def test_no_striker_returns_none():
    assert pick_striker([], (H, W)) is None
    assert pick_striker([person(450, 270, cls="Bat")], (H, W)) is None


def test_station_logo_against_the_edge_is_rejected():
    logo = Detection("Striker", 0.9, (W - 60, 100, W, 260))
    assert not plausible(logo, W, H)
    assert pick_striker([logo], (H, W)) is None


def test_scorebar_sliver_is_rejected():
    bar = Detection("Striker", 0.8, (100, 520, 700, 538))   # wide and flat
    assert not plausible(bar, W, H)


def test_too_small_to_crop_is_rejected():
    speck = person(450, 270, w=20, h=40)
    assert not plausible(speck, W, H)


def test_a_real_batter_passes():
    assert plausible(person(450, 270), W, H)


def test_accepts_a_frame_as_shape():
    frame = np.zeros((H, W, 3), np.uint8)
    assert pick_striker([person(450, 270)], frame) is not None


@pytest.mark.parametrize("x1", [0, 1, EDGE_PX])
def test_touching_the_left_border_is_rejected(x1):
    assert not plausible(Detection("Striker", 0.9, (x1, 100, x1 + 80, 300)), W, H)
