"""The frame grid. Day 3.4 judges crops with it, so it has to be right."""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from src.data.viz import compare, frame_grid          # noqa: E402


def _clip(t=7, h=40, w=60):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (t, h, w, 3), dtype=np.uint8)


def test_grid_geometry_matches_the_frame_count(tmp_path):
    out = frame_grid(_clip(t=7), str(tmp_path / "g.png"), cols=3, width=60)
    img = cv2.imread(out)
    assert img.shape[1] == 3 * 60                  # 3 columns
    assert img.shape[0] == 3 * 40                  # ceil(7/3) = 3 rows


def test_the_last_row_is_padded_not_cropped(tmp_path):
    """7 frames in 3 columns leaves 2 blanks -- all 7 must still be present."""
    out = frame_grid(_clip(t=7), str(tmp_path / "g.png"), cols=3, width=60)
    img = cv2.imread(out)
    assert img[-40:, -60:].max() == 0              # bottom-right tile is blank


def test_it_accepts_a_path_as_well_as_an_array(tmp_path):
    R = "data/CricShoot10kShootDataset/Cover Drive/vid1_0.avi"
    if not os.path.exists(R):
        pytest.skip("dataset not present")
    out = frame_grid(R, str(tmp_path / "g.png"), cols=6, width=100)
    assert cv2.imread(out).shape[1] == 600


def test_compare_puts_two_frames_side_by_side(tmp_path):
    out = compare(_clip(h=40), _clip(h=80), str(tmp_path / "c.png"), width=100)
    img = cv2.imread(out)
    assert img.shape[1] == 200                     # two panes
