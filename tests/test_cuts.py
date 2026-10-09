import numpy as np
import pytest

from src.preprocess.cuts import find_cuts, frame_diffs, shots

H, W = 120, 200


def solid(v, n):
    return np.full((n, H, W, 3), v, np.uint8)


def drifting(n, start=40, step=2):
    """A shot that slowly brightens -- motion, but no cut."""
    return np.stack([np.full((H, W, 3), start + step * i, np.uint8)
                     for i in range(n)])


def test_no_cut_in_a_steady_clip():
    assert find_cuts(solid(80, 20)) == []


def test_no_cut_when_the_scene_merely_moves():
    assert find_cuts(drifting(20)) == []


def test_finds_an_obvious_cut():
    clip = np.concatenate([solid(30, 10), solid(220, 10)])
    assert find_cuts(clip) == [10]


def test_finds_two_cuts():
    clip = np.concatenate([solid(20, 6), solid(200, 6), solid(40, 6)])
    assert find_cuts(clip) == [6, 12]


def test_shots_split_at_the_cut():
    clip = np.concatenate([solid(30, 10), solid(220, 14)])
    assert [list(s)[0] for s in shots(clip)] == [0, 10]
    assert sum(len(s) for s in shots(clip)) == 24


def test_a_clip_with_no_cut_is_one_shot():
    assert len(shots(solid(80, 24))) == 1


def test_too_short_to_judge():
    assert find_cuts(solid(80, 2)) == []


def test_diffs_length_is_one_less_than_frames():
    assert len(frame_diffs(solid(80, 9))) == 8


@pytest.mark.parametrize("n", [3, 24, 49])
def test_any_length(n):
    assert isinstance(find_cuts(solid(80, n)), list)
