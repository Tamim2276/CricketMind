import numpy as np
import pytest

from src.preprocess.sample import (DUP_TOL, TARGET, drop_duplicates, sample,
                                   sample_indices)

H, W = 40, 60


def frames(vals):
    return np.stack([np.full((H, W, 3), v, np.uint8) for v in vals])


# ---- duplicates ---------------------------------------------------------

def test_an_exact_repeat_is_dropped():
    clip, keep = drop_duplicates(frames([10, 10, 40, 70]))
    assert keep == [0, 2, 3] and len(clip) == 3


def test_a_near_repeat_is_dropped_too():
    """Nothing in this dataset is bit-identical; the clips were re-encoded."""
    a = np.full((H, W, 3), 50, np.uint8)
    b = a.copy()
    b[0, 0] = 60                                   # mean diff far below 1.0
    clip, keep = drop_duplicates(np.stack([a, b]))
    assert keep == [0]


def test_real_motion_survives():
    assert drop_duplicates(frames([10, 20, 30, 40]))[1] == [0, 1, 2, 3]


def test_a_run_of_repeats_collapses_to_one():
    assert drop_duplicates(frames([10, 10, 10, 10, 90]))[1] == [0, 4]


def test_duplicates_are_compared_to_the_last_kept_frame():
    """Otherwise a slow drift of sub-threshold steps deletes the whole clip."""
    clip = frames([10, 10.6, 11.2, 11.8, 12.4]).astype(np.uint8)
    assert len(drop_duplicates(clip)[1]) > 1


def test_a_single_frame_clip():
    clip, keep = drop_duplicates(frames([10]))
    assert keep == [0] and len(clip) == 1


def test_an_empty_clip():
    assert drop_duplicates(np.empty((0, H, W, 3), np.uint8))[1] == []


# ---- sampling -----------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 8, 14, 15, 16, 24, 28, 49])
def test_always_exactly_fifteen(n):
    assert len(sample_indices(n)) == TARGET


@pytest.mark.parametrize("n", [15, 24, 28, 49])
def test_first_and_last_are_always_included(n):
    idx = sample_indices(n)
    assert idx[0] == 0 and idx[-1] == n - 1


def test_indices_never_go_backwards():
    for n in (3, 15, 24, 49):
        idx = sample_indices(n)
        assert idx == sorted(idx)


def test_exactly_fifteen_frames_is_the_identity():
    assert sample_indices(15) == list(range(15))


def test_a_long_clip_is_spread_not_truncated():
    """The swing is in the middle; the first 15 would miss it."""
    idx = sample_indices(49)
    assert max(idx) == 48
    assert 20 <= idx[len(idx) // 2] <= 28


def test_a_short_clip_repeats_rather_than_failing():
    idx = sample_indices(9)
    assert len(idx) == TARGET and max(idx) == 8 and len(set(idx)) == 9


def test_sample_returns_frames():
    out = sample(frames(list(range(0, 240, 10))))
    assert out.shape == (TARGET, H, W, 3)


def test_no_frames_at_all_raises():
    with pytest.raises(ValueError):
        sample_indices(0)
