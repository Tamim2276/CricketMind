"""The point of this module is surviving an interruption, so that is what
most of these check."""
import json
import os

import numpy as np
import pytest

from src.preprocess import build_dataset as bd
from src.preprocess.detect import Detection

FH, FW = 540, 896


def det(cx=430, cy=390, w=100, h=220, cls="Striker", mask=False):
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    m = None
    if mask:
        m = np.zeros((FH, FW), np.uint8)
        m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
    return Detection(cls, 0.6, box, m)


class FakeModel:
    """Stands in for YOLO: every frame has one batter, in the same place."""
    names = {0: "Striker", 1: "Bat"}


def fake_detect(frames, **kw):
    return [[det(mask=kw.get("masks", False))] for _ in frames]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Three one-class clips of continuous, non-duplicate frames."""
    root = tmp_path / "clips" / "Sweep"
    root.mkdir(parents=True)
    clips = {}
    for i in range(3):
        name = f"vid{i}_0.avi"
        (root / name).write_bytes(b"not really a video")
        rng = np.random.default_rng(i)
        clips[f"Sweep/{name}"] = np.stack(
            [np.full((FH, FW, 3), 40 + 7 * k, np.uint8) for k in range(20)]
        ) | rng.integers(0, 2, (20, FH, FW, 3), dtype=np.uint8)

    monkeypatch.setattr(bd, "read_clip", lambda p: clips[
        os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)])
    monkeypatch.setattr(bd, "detect", fake_detect)
    monkeypatch.setattr(bd, "load_detector", lambda: FakeModel())
    monkeypatch.setattr(bd, "get_device", lambda: "cpu")
    return str(tmp_path / "clips"), str(tmp_path / "out")


def rows(out):
    """Readable rows only -- a test deliberately writes a damaged one."""
    got = []
    with open(os.path.join(out, "manifest.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            try:
                got.append(json.loads(line))
            except ValueError:
                pass
    return got


def test_processes_every_clip(dataset):
    root, out = dataset
    assert bd.build(root=root, out_dir=out) == 0
    assert len(rows(out)) == 3
    assert all(r["ok"] for r in rows(out))


def test_writes_both_variants_at_fifteen_frames(dataset):
    root, out = dataset
    bd.build(root=root, out_dir=out)
    for v in bd.VARIANTS:
        p = os.path.join(out, v, "Sweep", "vid0_0.npy")
        assert np.load(p).shape == (15, 224, 224, 3)


def test_a_second_run_skips_what_is_done(dataset):
    """The power-cut case: start again, lose nothing, redo nothing."""
    root, out = dataset
    bd.build(root=root, out_dir=out)
    before = os.path.getmtime(os.path.join(out, "box", "Sweep", "vid0_0.npy"))
    bd.build(root=root, out_dir=out)
    assert len(rows(out)) == 3                      # no duplicate rows
    assert os.path.getmtime(os.path.join(out, "box", "Sweep", "vid0_0.npy")) == before


def test_an_interrupted_run_resumes_where_it_stopped(dataset):
    root, out = dataset
    bd.build(root=root, out_dir=out, limit=1)       # "power cut" after one
    assert len(rows(out)) == 1
    bd.build(root=root, out_dir=out)
    assert len(rows(out)) == 3
    assert len({r["clip"] for r in rows(out)}) == 3


def test_a_truncated_manifest_line_does_not_break_resume(dataset):
    root, out = dataset
    bd.build(root=root, out_dir=out, limit=1)
    with open(os.path.join(out, "manifest.jsonl"), "a", encoding="utf-8") as fh:
        fh.write('{"clip": "Sweep/vid1_0.avi", "ok": tr')    # cut mid-write
    bd.build(root=root, out_dir=out)
    assert {r["clip"] for r in rows(out)} == {f"Sweep/vid{i}_0.avi" for i in range(3)}


def test_one_unreadable_clip_does_not_stop_the_run(dataset):
    root, out = dataset
    real = bd.read_clip

    def boom(p):
        if "vid1_0" in p:
            raise RuntimeError("decoder said no")
        return real(p)

    bd.read_clip = boom
    try:
        assert bd.build(root=root, out_dir=out) == 1
    finally:
        bd.read_clip = real
    bad = [r for r in rows(out) if not r["ok"]]
    assert len(bad) == 1 and "decoder said no" in bad[0]["reason"]
    assert len(rows(out)) == 3                      # the other two still ran


def test_a_half_written_npy_is_never_left_behind(tmp_path):
    p = str(tmp_path / "x.npy")
    bd._save(np.zeros((2, 3), np.uint8), p)
    assert np.load(p).shape == (2, 3)
    assert not os.path.exists(p + ".tmp")


def test_only_the_box_variant(dataset):
    root, out = dataset
    bd.build(root=root, out_dir=out, variants=("box",))
    assert os.path.isdir(os.path.join(out, "box"))
    assert not os.path.exists(os.path.join(out, "seg"))


def test_clip_list_is_sorted_and_classed(tmp_path):
    for cls in ("Sweep", "Pull"):
        d = tmp_path / cls
        d.mkdir()
        for n in ("vid2_0.avi", "vid1_0.avi", "notes.txt"):
            (d / n).write_text("x")
    got = bd.clip_list(str(tmp_path))
    assert got == [("Pull", "Pull/vid1_0.avi"), ("Pull", "Pull/vid2_0.avi"),
                   ("Sweep", "Sweep/vid1_0.avi"), ("Sweep", "Sweep/vid2_0.avi")]


# ---- the progress line --------------------------------------------------

def test_the_bar_fills_up():
    import time as _t
    t0 = _t.time() - 10
    start = bd._progress(0, 100, 0, t0, True)
    half = bd._progress(50, 100, 0, t0, True)
    end = bd._progress(100, 100, 0, t0, True)
    dot = chr(0x25CF)
    assert start.count(dot) == 0
    assert half.count(dot) == bd.BAR // 2
    assert end.count(dot) == bd.BAR


def test_the_bar_rewrites_one_line():
    line = bd._progress(5, 100, 0, __import__("time").time() - 5, True)
    assert line.startswith("\r") and "\n" not in line


def test_no_escape_codes_when_output_is_not_a_terminal():
    """A log file should not fill up with colour codes and carriage returns."""
    line = bd._progress(5, 100, 2, __import__("time").time() - 5, False)
    assert "\033" not in line and "\r" not in line
    assert "5/100" in line and "2 failed" in line


def test_failures_are_coloured_only_when_there_are_some():
    t0 = __import__("time").time() - 5
    assert bd._RED in bd._progress(5, 100, 1, t0, True)
    assert bd._RED not in bd._progress(5, 100, 0, t0, True)
