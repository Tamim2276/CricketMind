"""Reading a clip: the real length, the real colours, and loud failures."""
import os

import numpy as np
import pytest

from src.data.video import ClipError, clip_shape, read_clip

R = "data/CricShoot10kShootDataset"
needs_data = pytest.mark.skipif(not os.path.isdir(R), reason="dataset not present")

# one clip per distinct (frames, h, w) in the dataset
SHAPES = [
    ("Cover Drive/vid40_17.avi", (22, 540, 896)),
    ("Flick/vid52_4.avi", (23, 540, 896)),
    ("Cover Drive/vid121_13.avi", (24, 360, 598)),      # off-resolution match
    ("Defensive/vid22_2.avi", (24, 494, 896)),          # off-resolution match
    ("Cover Drive/vid100_41.avi", (24, 540, 896)),
    ("Cover Drive/vid203_13.avi", (28, 540, 896)),
    ("Cover Drive/vid1_0.avi", (29, 540, 896)),
    ("Cover Drive/vid294_11.avi", (49, 540, 896)),
]


@needs_data
@pytest.mark.parametrize("rel,want", SHAPES)
def test_the_real_length_comes_back_not_24(rel, want):
    """Hardcoding 24 would silently mangle 2,434 of the 10,091 clips."""
    clip = read_clip(os.path.join(R, rel))
    assert clip.shape == (*want, 3)
    assert clip.dtype == np.uint8


@needs_data
@pytest.mark.parametrize("rel,want", SHAPES)
def test_header_and_decoded_count_agree_here(rel, want):
    """They agree in this dataset. read_clip still counts, because headers lie."""
    assert clip_shape(os.path.join(R, rel))[0] == want[0]


@needs_data
def test_pixels_span_the_full_byte_range():
    clip = read_clip(os.path.join(R, "Cover Drive/vid1_0.avi"))
    assert clip.min() == 0 and clip.max() == 255


@needs_data
def test_rgb_and_bgr_differ_by_a_channel_swap():
    p = os.path.join(R, "Cover Drive/vid1_0.avi")
    rgb = read_clip(p, rgb=True)
    bgr = read_clip(p, rgb=False)
    assert not np.array_equal(rgb, bgr), "rgb flag did nothing"
    assert np.array_equal(rgb, bgr[:, :, :, ::-1])


@needs_data
def test_the_result_survives_becoming_a_tensor():
    """A [::-1] slice would be a negative-stride view and fail here."""
    torch = pytest.importorskip("torch")
    clip = read_clip(os.path.join(R, "Cover Drive/vid1_0.avi"))
    t = torch.from_numpy(clip)
    assert t.shape == clip.shape


@needs_data
def test_two_reads_of_one_clip_are_identical():
    p = os.path.join(R, "Cover Drive/vid1_0.avi")
    assert np.array_equal(read_clip(p), read_clip(p))


def test_a_missing_file_raises_cliperror_naming_the_path():
    with pytest.raises(ClipError, match="nope.avi"):
        read_clip("nope.avi")
    with pytest.raises(ClipError, match="nope.avi"):
        clip_shape("nope.avi")


def test_a_file_that_is_not_a_video_raises_cliperror(tmp_path):
    """Day 4 needs this distinguishable from a bug in its own code."""
    junk = tmp_path / "junk.avi"
    junk.write_bytes(b"this is not a video")
    with pytest.raises(ClipError):
        read_clip(str(junk))


def test_cliperror_is_catchable_as_a_runtime_error():
    with pytest.raises(RuntimeError):
        read_clip("nope.avi")


def test_a_persistent_failure_still_raises(tmp_path):
    """A genuinely bad file must not become a silent success."""
    junk = tmp_path / "junk.avi"
    junk.write_bytes(b"not a video")
    with pytest.raises(ClipError):
        read_clip(str(junk))


def test_a_transient_failure_is_retried_once_and_counted(monkeypatch):
    """Guards Day 4: a one-off decoder hiccup must not condemn a good clip."""
    import src.data.video as V
    good = np.zeros((29, 4, 4, 3), np.uint8)
    calls = {"n": 0}

    def flaky(path, rgb):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ClipError(f"opened but decoded 0 frames: {path}")
        return good

    monkeypatch.setattr(V, "_read_once", flaky)
    V.reset_retries()
    out = V.read_clip(__file__)             # a file that exists
    assert calls["n"] == 2, "did not retry"
    assert V.retries() == 1, "retry was not counted"
    assert np.array_equal(out, good)


def test_two_failures_in_a_row_still_raise(monkeypatch):
    """Retrying must not turn a genuinely broken clip into a silent success."""
    import src.data.video as V

    def always_bad(path, rgb):
        raise ClipError("nope")

    monkeypatch.setattr(V, "_read_once", always_bad)
    with pytest.raises(ClipError):
        V.read_clip(__file__)


def test_a_missing_file_is_not_retried(monkeypatch):
    """No point retrying something that does not exist."""
    import src.data.video as V
    calls = {"n": 0}
    real = V._read_once

    def counting(path, rgb):
        calls["n"] += 1
        return real(path, rgb)

    monkeypatch.setattr(V, "_read_once", counting)
    with pytest.raises(ClipError):
        V.read_clip("definitely_not_here.avi")
    assert calls["n"] == 1, "retried a missing file"


def test_retry_can_be_turned_off():
    with pytest.raises(ClipError):
        read_clip("nope.avi", retry=False)


def test_the_counter_survives_a_from_import():
    """The reason retries() is a function and not a bare global."""
    from src.data.video import reset_retries, retries
    reset_retries()
    assert retries() == 0
