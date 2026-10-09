import json
import os

import numpy as np
import pytest
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, CricShotDataset

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def split(tmp_path):
    """A split CSV, a classes.json, and three of its four clips on disk."""
    proc = tmp_path / "processed"
    rows = [("Cover Drive/vid1_0.avi", 0), ("Sweep/vid2_0.avi", 1),
            ("Pull/vid3_0.avi", 2), ("Sweep/vid9_0.avi", 1)]
    for clip, _ in rows[:3]:                    # vid9_0 deliberately absent
        for v in ("box", "seg"):
            p = proc / v / clip.replace("/", os.sep)[:-4]
            p.parent.mkdir(parents=True, exist_ok=True)
            fill = 200 if v == "box" else 10
            np.save(str(p) + ".npy",
                    np.full((15, 224, 224, 3), fill, np.uint8))

    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "clip,class,label,match\n" + "".join(
            f"{c},{CLASSES[l]},{l},{c.split('/')[1].split('_')[0]}\n"
            for c, l in rows), encoding="utf-8")
    (tmp_path / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    return str(csv_path), str(proc)


# ---- the shape contract -------------------------------------------------

def test_one_item_is_channels_first_float(split):
    csv_path, proc = split
    x, y = CricShotDataset(csv_path, processed=proc)[0]
    assert x.shape == (15, 3, 224, 224)
    assert x.dtype == torch.float32
    assert isinstance(y, int) and 0 <= y < len(CLASSES)


def test_normalised_with_imagenet_statistics(split):
    """Not 0-1: the pretrained encoder was fitted on these exact statistics."""
    csv_path, proc = split
    x, _ = CricShotDataset(csv_path, processed=proc)[0]
    want = (200 / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert x[0, 0].mean().item() == pytest.approx(want, abs=1e-4)


def test_each_channel_gets_its_own_mean(split):
    csv_path, proc = split
    x, _ = CricShotDataset(csv_path, processed=proc)[0]
    assert x[0, 0].mean() != x[0, 1].mean() != x[0, 2].mean()


def test_labels_match_the_csv(split):
    csv_path, proc = split
    d = CricShotDataset(csv_path, processed=proc)
    assert [y for _, y in (d[i] for i in range(len(d)))] == [0, 1, 2]


# ---- the clips that failed preprocessing --------------------------------

def test_a_clip_with_no_output_is_dropped(split):
    """211 of 10,091 failed. Drop them here, not on epoch 3."""
    csv_path, proc = split
    d = CricShotDataset(csv_path, processed=proc)
    assert len(d) == 3
    assert d.missing == ["Sweep/vid9_0.avi"]


def test_strict_mode_refuses_instead(split):
    csv_path, proc = split
    with pytest.raises(FileNotFoundError, match="vid9_0"):
        CricShotDataset(csv_path, processed=proc, strict=True)


def test_nothing_usable_is_an_error(split, tmp_path):
    csv_path, _ = split
    with pytest.raises(RuntimeError, match="no usable clips"):
        CricShotDataset(csv_path, processed=str(tmp_path / "empty"))


# ---- variants -----------------------------------------------------------

def test_the_two_variants_hold_the_same_clips(split):
    csv_path, proc = split
    box = CricShotDataset(csv_path, processed=proc, variant="box")
    seg = CricShotDataset(csv_path, processed=proc, variant="seg")
    assert len(box) == len(seg)
    assert not torch.equal(box[0][0], seg[0][0])
    assert box[0][1] == seg[0][1]


def test_an_unknown_variant_is_not_silently_empty(split):
    csv_path, proc = split
    with pytest.raises(RuntimeError):
        CricShotDataset(csv_path, processed=proc, variant="nope")


# ---- the rest -----------------------------------------------------------

def test_classes_come_from_beside_the_split(split):
    csv_path, proc = split
    assert CricShotDataset(csv_path, processed=proc).classes == CLASSES


def test_missing_classes_json_is_an_error(tmp_path, split):
    csv_path, proc = split
    moved = tmp_path / "elsewhere.csv"
    moved.write_text(open(csv_path, encoding="utf-8").read(), encoding="utf-8")
    os.remove(os.path.join(os.path.dirname(csv_path), "classes.json"))
    with pytest.raises(FileNotFoundError, match="classes.json"):
        CricShotDataset(str(moved), processed=proc)


def test_label_counts_ignore_the_missing(split):
    csv_path, proc = split
    assert CricShotDataset(csv_path, processed=proc).label_counts() == {
        "Cover Drive": 1, "Sweep": 1, "Pull": 1}


def test_a_transform_is_applied(split):
    csv_path, proc = split
    plain = CricShotDataset(csv_path, processed=proc)
    flipped = CricShotDataset(csv_path, processed=proc,
                              transform=lambda x: torch.flip(x, dims=[-1]))
    assert plain.transform is None
    assert torch.equal(flipped[0][0], torch.flip(plain[0][0], dims=[-1]))


def test_it_works_in_a_dataloader(split):
    from torch.utils.data import DataLoader
    csv_path, proc = split
    x, y = next(iter(DataLoader(CricShotDataset(csv_path, processed=proc),
                                batch_size=2)))
    assert x.shape == (2, 15, 3, 224, 224) and y.shape == (2,)
