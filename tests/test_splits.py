"""match_id, and the two split schemes it drives."""
import csv
import json
import os
from collections import Counter

import pytest

from src.utils.splits import (
    leaks,
    load_clips,
    make_all_splits,
    match_id,
    split_clips,
    write_splits,
)

DATA_ROOT = "data/CricShoot10kShootDataset"
needs_data = pytest.mark.skipif(not os.path.isdir(DATA_ROOT),
                                reason="dataset not present")


# match_id

def test_the_basic_shape():
    assert match_id("vid100_41.avi") == "vid100"
    assert match_id("vid1_0.avi") == "vid1"
    assert match_id("vid536_53.avi") == "vid536"


def test_a_path_works_not_just_a_bare_name():
    assert match_id("Cover Drive/vid100_41.avi") == "vid100"
    assert match_id(r"data\x\Late Cut\vid7_3.avi") == "vid7"     # windows


def test_clips_from_one_match_share_the_key():
    assert len({match_id(n) for n in
                ["vid1_0.avi", "vid1_4.avi", "vid1_43.avi"]}) == 1


def test_the_match_number_is_not_confused_with_the_shot_number():
    assert match_id("vid1_100.avi") == "vid1"
    assert match_id("vid100_1.avi") == "vid100"


def test_augmented_copies_keep_their_original_match():
    """Day 7.2 flips the training set. A flip must not land opposite its original."""
    assert match_id("vid100_41_flip.avi") == "vid100"
    assert match_id("vid100_41_flip_x2.avi") == "vid100"


@pytest.mark.parametrize("bad", [
    "vid100.avi", "vid100_41", "vid100_41.mp4", "100_41.avi",
    "vidabc_41.avi", "vid100_abc.avi", "VID100_41.avi", "vid100-41.avi",
    "", "Cover Drive",
])
def test_anything_unexpected_raises(bad):
    with pytest.raises(ValueError):
        match_id(bad)


def test_the_error_names_the_offending_input():
    with pytest.raises(ValueError, match="nonsense.txt"):
        match_id("nonsense.txt")


# the splits, on synthetic data

def _toy(n_matches=40, per_match=10, n_classes=5):
    """Matches that each span several classes, like the real data."""
    items = []
    for m in range(1, n_matches + 1):
        for s in range(per_match):
            items.append((f"class{s % n_classes}/vid{m}_{s}.avi",
                          f"class{s % n_classes}"))
    return items


def test_grouped_never_puts_a_match_in_two_splits():
    a = split_clips(_toy(), scheme="grouped", seed=1)
    assert leaks(a) == {}


def test_author_does_let_a_match_straddle():
    """Not a bug -- it is the protocol being reproduced."""
    a = split_clips(_toy(), scheme="author", seed=1)
    assert len(leaks(a)) > 0


def test_author_keeps_one_match_and_class_together():
    a = split_clips(_toy(), scheme="author", seed=1)
    where = {}
    for split, rows in a.items():
        for clip, cls in rows:
            key = (match_id(clip), cls)
            assert where.setdefault(key, split) == split, f"{key} straddles"


def test_every_clip_lands_exactly_once():
    items = _toy()
    for scheme in ("grouped", "author"):
        a = split_clips(items, scheme=scheme, seed=1)
        placed = [c for rows in a.values() for c, _ in rows]
        assert sorted(placed) == sorted(c for c, _ in items)
        assert len(placed) == len(set(placed))


def test_proportions_are_close_to_the_targets():
    items = _toy(n_matches=200, per_match=10)
    for scheme in ("grouped", "author"):
        a = split_clips(items, scheme=scheme, seed=1)
        n = len(items)
        for split, target in [("train", 0.64), ("val", 0.16), ("test", 0.20)]:
            assert abs(len(a[split]) / n - target) < 0.03, \
                f"{scheme}/{split}: {len(a[split])/n:.3f} vs {target}"


def test_every_class_reaches_every_split():
    """A class missing from val or test would make its metrics undefined."""
    items = _toy(n_matches=100)
    for scheme in ("grouped", "author"):
        a = split_clips(items, scheme=scheme, seed=1)
        for split, rows in a.items():
            assert len({c for _, c in rows}) == 5, f"{scheme}/{split} lost a class"


def test_the_same_seed_gives_the_same_split():
    items = _toy()
    assert split_clips(items, seed=7) == split_clips(items, seed=7)


def test_a_different_seed_gives_a_different_split():
    items = _toy()
    assert split_clips(items, seed=7) != split_clips(items, seed=8)


# the files on disk

def test_written_csvs_round_trip(tmp_path):
    items = _toy()
    classes = sorted({c for _, c in items})
    a = split_clips(items, scheme="grouped", seed=1)
    write_splits(str(tmp_path), a, classes)

    assert json.loads((tmp_path / "classes.json").read_text()) == classes
    seen = 0
    for split in ("train", "val", "test"):
        with open(tmp_path / f"{split}.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            assert r["match"] == match_id(r["clip"])
            # the label index is frozen in the file, not re-derived later
            assert classes[int(r["label"])] == r["class"]
        seen += len(rows)
    assert seen == len(items)


# the real dataset

@needs_data
def test_every_real_filename_parses():
    items, classes = load_clips(DATA_ROOT)
    assert len(items) == 10091
    assert len(classes) == 15
    assert len({match_id(c) for c, _ in items}) == 534


@needs_data
def test_the_real_splits_hold_up(tmp_path):
    report = make_all_splits(DATA_ROOT, str(tmp_path), seed=42, verbose=False)

    assert report["grouped"]["straddling_matches"] == 0
    # ~474 of 534; section 6 of the plan predicts ~490. Near zero would mean
    # two grouped splits by accident, and the comparison would measure nothing.
    assert report["author"]["straddling_matches"] > 400

    for scheme in ("grouped", "author"):
        sizes = report[scheme]["sizes"]
        n = sum(sizes.values())
        assert n == 10091
        for split, target in [("train", 0.64), ("val", 0.16), ("test", 0.20)]:
            assert abs(sizes[split] / n - target) < 0.02


@needs_data
def test_no_class_is_skewed_in_the_real_splits():
    items, _ = load_clips(DATA_ROOT)
    totals = Counter(c for _, c in items)
    for scheme in ("grouped", "author"):
        a = split_clips(items, scheme=scheme, seed=42)
        for split, target in [("train", 0.64), ("val", 0.16), ("test", 0.20)]:
            got = Counter(c for _, c in a[split])
            for cls, n in totals.items():
                assert abs(got[cls] / n - target) < 0.04, \
                    f"{scheme}/{split}/{cls}: {got[cls]/n:.3f} vs {target}"
