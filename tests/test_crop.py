"""Each test pins one of the four stages: rogue rejection, span choice,
smoothing, and the square crop geometry."""
import numpy as np
import pytest

from src.preprocess.crop import (SIZE, _fill, _square, crop_clip, segment,
                                 smooth_boxes, track_boxes)
from src.preprocess.striker import pick_striker
from src.preprocess.detect import Detection

FH, FW = 540, 896


def det(cx, cy, w=100, h=220, cls="Striker", conf=0.6):
    return Detection(cls, conf, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def clip(n, shift=0):
    """A visually continuous clip -- random noise would read as a cut per frame."""
    base = np.full((FH, FW, 3), 90 + shift, np.uint8)
    return np.stack([np.clip(base + i % 3, 0, 255).astype(np.uint8)
                     for i in range(n)])


def cut_clip(before, after):
    """Two visually different shots joined -- a real cut at `before`."""
    return np.concatenate([clip(before), clip(after, shift=130)])


# ---- stage 1: rogue rejection -------------------------------------------

def test_a_box_miles_from_the_rest_is_dropped():
    per = [[det(450, 270)] for _ in range(8)]
    per[5] = [det(200, 120)]                 # a fielder in a crowd shot
    boxes = track_boxes(per, (FH, FW))
    assert boxes[5] is None
    assert sum(b is not None for b in boxes) == 7


def test_the_batter_walking_is_not_a_rogue():
    """He moves down the pitch; that is signal, not noise."""
    per = [[det(430 + 4 * i, 250 + 3 * i)] for i in range(12)]
    assert all(b is not None for b in track_boxes(per, (FH, FW)))


def test_too_few_detections_to_judge():
    per = [[det(450, 270)], [], [det(100, 100)]]
    assert sum(b is not None for b in track_boxes(per, (FH, FW))) == 2


# ---- stage 2: span ------------------------------------------------------

def test_a_long_gap_with_no_cut_is_filled_not_trimmed():
    """79% of long gaps are the detector blinking. Keep those frames."""
    per = [[det(450, 270)] for _ in range(16)]
    for i in range(6, 11):
        per[i] = []                         # five frames missed, camera steady
    r = crop_clip(clip(16), per)
    assert r.idxs == list(range(16))
    assert "filled 5" in r.note


def test_a_gap_at_a_real_cut_still_trims():
    per = [[det(450, 270)] for _ in range(12)] + [[] for _ in range(12)]
    r = crop_clip(cut_clip(12, 12), per)
    assert r.idxs == list(range(12))


def test_a_mostly_undetected_shot_is_refused():
    """Interpolating across 80% of a clip would be inventing the track."""
    per = [[] for _ in range(24)]
    for i in (0, 4, 23):
        per[i] = [det(450, 270)]
    r = crop_clip(clip(24), per)
    assert r.frames.shape[0] == 0 and "detected" in r.note


def test_fill_interpolates_a_blink():
    out = _fill([(0, 0, 10, 10), None, (10, 10, 20, 20)], 0, 2)
    assert out[1] == (5.0, 5.0, 15.0, 15.0)


def test_clip_stops_at_a_camera_cut():
    per = [[det(450, 270)] for _ in range(20)] + [[] for _ in range(9)]
    r = crop_clip(cut_clip(20, 9), per)
    assert r.idxs == list(range(20))
    assert "trimmed 9" in r.note


def test_no_striker_at_all():
    r = crop_clip(clip(24), [[] for _ in range(24)])
    assert r.frames.shape[0] == 0 and r.idxs == []
    assert "no striker" in r.note


def test_a_shot_too_short_to_use():
    per = [[] for _ in range(24)]
    for i in range(4):
        per[i] = [det(450, 270)]
    r = crop_clip(cut_clip(4, 20), per)
    assert r.frames.shape[0] == 0 and "4 frames" in r.note


# ---- stage 3: smoothing -------------------------------------------------

def test_smoothing_removes_shake_but_keeps_the_drift():
    jitter = [(400 + 6 * (i % 2), 200, 500 + 6 * (i % 2), 420) for i in range(12)]
    sm = smooth_boxes(jitter)
    wobble = np.abs(np.diff(sm[:, 0])).max()
    assert wobble < 3, wobble


def test_smoothing_keeps_the_box_count():
    assert len(smooth_boxes([(0, 0, 10, 10)] * 7)) == 7


# ---- stage 4: crop geometry ---------------------------------------------

def test_crop_is_square_and_sized():
    per = [[det(450, 270)] for _ in range(16)]
    r = crop_clip(clip(16), per)
    assert r.frames.shape == (16, SIZE, SIZE, 3)


def test_one_size_for_the_whole_clip():
    """The box doubles in width mid-clip; the crop must not pulse."""
    per = [[det(450, 270, w=100)] for _ in range(6)] + \
          [[det(450, 270, w=200)] for _ in range(6)]
    sides = set()
    for k in range(12):
        box = (450 - 50, 160, 450 + 50, 380)
        sides.add(_square(box, 300.0, FW, FH)[2])
    assert len(sides) == 1
    assert crop_clip(clip(12), per).frames.shape[0] == 12


def test_square_is_nudged_back_inside_the_frame():
    x, y, s = _square((10, 10, 60, 60), 300.0, FW, FH)
    assert x >= 0 and y >= 0 and x + s <= FW and y + s <= FH


def test_square_never_exceeds_the_frame():
    x, y, s = _square((448, 270, 460, 290), 5000.0, FW, FH)
    assert s == FH and 0 <= x and x + s <= FW


@pytest.mark.parametrize("n", [8, 15, 24, 49])
def test_any_clip_length_works(n):
    per = [[det(450, 270)] for _ in range(n)]
    assert crop_clip(clip(n), per).frames.shape == (n, SIZE, SIZE, 3)


# ---- the temporal link --------------------------------------------------

# the batter stands below centre; a fielder stands above it, nearer the middle
# of the frame, so centrality prefers the fielder wherever he is detected
BATTER, FIELDER = (430, 390), (450, 200)


def test_the_track_outvotes_a_run_of_wrong_picks():
    """Centrality picks a fielder for 5 frames; the other 13 outvote him."""
    per = [[det(*FIELDER, w=90, h=200), det(*BATTER)] if 6 <= i < 11
           else [det(*BATTER)] for i in range(18)]
    assert pick_striker(per[7], (FH, FW)).box == det(*FIELDER, w=90, h=200).box
    boxes = track_boxes(per, (FH, FW))
    for i in range(18):
        assert boxes[i] is not None, i
        assert abs((boxes[i][1] + boxes[i][3]) / 2 - BATTER[1]) < 20, i


def test_one_bad_frame_does_not_poison_the_rest():
    """The reason for two passes rather than chaining off the last frame."""
    per = [[det(*BATTER)] for _ in range(14)]
    per[1] = [det(*FIELDER, w=90, h=200), det(*BATTER)]
    boxes = track_boxes(per, (FH, FW))
    assert all(b is not None for b in boxes)
    assert all(abs((b[1] + b[3]) / 2 - BATTER[1]) < 20 for b in boxes[2:])


def test_a_frame_with_nobody_on_the_track_stays_empty():
    per = [[det(*BATTER)] for _ in range(12)]
    per[5] = [det(120, 150)]                # a different person entirely
    assert track_boxes(per, (FH, FW))[5] is None


def test_the_batter_moving_is_still_tracked():
    per = [[det(400 + 6 * i, 330 + 4 * i)] for i in range(12)]
    assert all(b is not None for b in track_boxes(per, (FH, FW)))


def test_too_few_detections_to_build_a_track():
    per = [[det(450, 270)], [], []]
    assert sum(b is not None for b in track_boxes(per, (FH, FW))) == 1


def test_swapped_frames_are_reported():
    per = [[det(*FIELDER, w=90, h=200), det(*BATTER)] if 5 <= i < 9
           else [det(*BATTER)] for i in range(16)]
    assert "4 swapped" in crop_clip(clip(16), per).note


# ---- the segmented variant ----------------------------------------------

def _with_mask(d, region):
    m = np.zeros((FH, FW), np.uint8)
    y1, y2, x1, x2 = region
    m[y1:y2, x1:x2] = 1
    return d._replace(mask=m)


def test_segment_keeps_the_batter_and_drops_the_rest():
    frame = np.full((FH, FW, 3), 200, np.uint8)
    s = _with_mask(det(*BATTER), (280, 500, 380, 480))
    out = segment(frame, s)
    assert out[300, 400].tolist() == [200, 200, 200]      # inside the mask
    assert out[50, 50].tolist() == [0, 0, 0]              # background


def test_segment_keeps_his_bat_too():
    """Dropping the bat would throw away the cue that names the shot."""
    frame = np.full((FH, FW, 3), 200, np.uint8)
    s = _with_mask(det(*BATTER), (280, 500, 380, 480))
    bat = _with_mask(det(500, 330, w=20, h=90, cls="Bat"), (300, 360, 490, 510))
    assert segment(frame, s, [s, bat])[330, 500].tolist() == [200, 200, 200]


def test_a_distant_bat_is_somebody_elses():
    frame = np.full((FH, FW, 3), 200, np.uint8)
    s = _with_mask(det(*BATTER), (280, 500, 380, 480))
    far = _with_mask(det(860, 60, w=20, h=90, cls="Bat"), (20, 100, 850, 875))
    assert segment(frame, s, [s, far])[60, 860].tolist() == [0, 0, 0]


def test_segment_without_a_mask_is_a_no_op():
    frame = np.full((FH, FW, 3), 200, np.uint8)
    assert np.array_equal(segment(frame, det(*BATTER)), frame)
    assert np.array_equal(segment(frame, None), frame)


def test_segmented_crop_runs_and_reports_missing_masks():
    per = [[_with_mask(det(*BATTER), (280, 500, 380, 480))] for _ in range(12)]
    per[3] = [det(*BATTER)]                       # one frame with no mask
    r = crop_clip(clip(12), per, segmented=True)
    assert r.frames.shape == (12, SIZE, SIZE, 3)
    assert "1 unsegmented" in r.note


def test_unsegmented_crop_is_unaffected_by_masks():
    per = [[_with_mask(det(*BATTER), (280, 500, 380, 480))] for _ in range(12)]
    a = crop_clip(clip(12), per)
    b = crop_clip(clip(12), per, segmented=True)
    assert not np.array_equal(a.frames, b.frames)
    assert a.frames.shape == b.frames.shape
