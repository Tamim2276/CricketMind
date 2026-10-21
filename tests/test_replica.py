"""tools/replica.py: the authors' preprocessing as their notebooks run it."""
import csv
import os

import numpy as np
import pytest

from tools.replica import (as_ints, authors_box, even_indices,
                           mjpg_round_trip, stretch, write_split)


def test_padding_goes_sideways_only():
    """Their notebook pads 10% of the width left and right, and crops the
    rows exactly at the box."""
    assert authors_box((100, 50, 200, 300), None, 1280) == (90, 50, 210, 300)


def test_the_bat_box_joins_the_striker_box():
    box = authors_box((100, 50, 200, 300), (180, 20, 260, 120), 1280)
    # union (100, 20, 260, 300), width 160, pad 16
    assert box == (84, 20, 276, 300)


def test_the_padding_stops_at_the_frame_edge():
    assert authors_box((5, 0, 95, 100), None, 100) == (0, 0, 100, 100)


def test_a_tall_box_is_stretched_to_a_square_not_kept_in_shape():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[100:400, 600:700] = 255                 # 100 wide, 300 tall
    out = stretch(frame, (600, 100, 700, 400))
    assert out.shape == (224, 224, 3)
    assert out.min() == 255                       # the whole box fills it


def test_an_empty_box_is_dropped():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert stretch(frame, (50, 10, 50, 90)) is None


def test_fifteen_even_frames_ends_included():
    idx = even_indices(24)
    assert len(idx) == 15
    assert idx[0] == 0 and idx[-1] == 23
    short = even_indices(5)                       # fewer frames than 15
    assert len(short) == 15
    assert list(short) == sorted(short)
    assert sorted(set(short.tolist())) == [0, 1, 2, 3, 4]


def test_boxes_are_truncated_like_astype_int():
    assert as_ints((10.9, 20.1, 30.5, 40.99)) == (10, 20, 30, 40)


def test_mjpg_gives_back_every_frame_close_to_what_went_in(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(4):
        base = rng.integers(0, 255, (1, 1, 3))
        frames.append(np.full((224, 224, 3), base, np.uint8))
    back = mjpg_round_trip(frames, str(tmp_path))
    assert len(back) == 4
    for a, b in zip(frames, back):
        assert np.abs(a.astype(int) - b.astype(int)).max() <= 3


def test_ultralytics_still_paints_striker_blue_and_bat_cyan():
    """The replica's overlay is ultralytics' default palette, as theirs was.
    If an upgrade changes it, the replica stops being theirs."""
    from ultralytics.utils.plotting import colors

    assert colors(0, False) == (4, 42, 255)        # class 0, Striker
    assert colors(1, False) == (11, 219, 235)      # class 1, Bat


def test_the_split_keeps_only_clips_that_came_out(tmp_path):
    rows = [{"clip": "Pull/vid1_1.avi", "class": "Pull", "label": "8",
             "match": "vid1"},
            {"clip": "Hook/vid2_3.avi", "class": "Hook", "label": "4",
             "match": "vid2"}]
    path, kept = write_split(rows, {"Hook/vid2_3.avi"}, out=str(tmp_path))
    assert kept == 1
    with open(path, encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert [r["clip"] for r in got] == ["Hook/vid2_3.avi"]
    assert os.path.exists(os.path.join(str(tmp_path), "classes.json"))


def test_a_full_build_refuses_a_disk_that_cannot_hold_it(monkeypatch):
    import shutil as sh

    from tools import replica

    class Usage:
        free = 10 * 1024 ** 3                     # 10 GB free

    monkeypatch.setattr(sh, "disk_usage", lambda path: Usage())
    assert replica.room_for(100) is None          # 0.2 GB: fine
    told = replica.room_for(9880)                 # 20.8 GiB: no
    assert "need 20.8 GB" in told and "10.0 GB free" in told


def test_every_clip_is_the_author_split_whole():
    from tools.replica import every_clip

    rows = every_clip()
    names = set()
    for row in rows:
        names.add(row["clip"])
    assert len(names) == len(rows)                # no clip twice
    assert len(rows) > 9000
