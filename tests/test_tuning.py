"""Family T (Days 38-40): one change from C06 or F02, and each change real.

A setting that quietly does nothing costs a whole run, so most of this checks
that each one reaches the model, the loss or the clip it claims to.
"""
import io
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src import train as T
from src.data.augment import (ClipColour, ClipCrop, ClipFlip, ClipTimeJitter,
                              build_transform)
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.experiments import registry as R
from src.experiments import run as runner
from src.methods import video3d
from src.models.video_classifier import VideoClassifier
from tools import leaderboard as L
from tools import tuning as U


def _clip(frames=4, size=8):
    torch.manual_seed(0)
    return torch.rand(frames, 3, size, size)


# ---- the registry -----------------------------------------------------------

def test_every_tuning_run_is_grouped_only():
    entries = R.entries("T")
    assert len(entries) == 26
    for entry in entries:
        assert R.splits_of(entry) == ("grouped",), entry["id"]
    # every other family still runs on both
    assert R.splits_of(R.get("C06_mvitv2_s")) == tuple(R.SPLITS)


@pytest.mark.parametrize("base, original", [("tune_c06", "C06_mvitv2_s"),
                                            ("tune_f02", "F02_multiscale_tf")])
def test_a_tuning_run_differs_from_its_original_only_in_its_delta(base,
                                                                  original):
    want = R.resolve(R.get(original), "grouped")
    seen = 0
    for entry in R.entries("T"):
        if entry["base"] != base:
            continue
        seen += 1
        got = R.resolve(entry, "grouped")
        differ = R.config_differences(want, got)
        assert differ == sorted(entry["cfg"]), entry["id"]
    assert seen == 13


def test_every_tuning_run_is_ready():
    for entry in R.entries("T"):
        assert R.status(entry) == "ready", entry["id"]


def test_the_family_runs_on_grouped_only_even_if_both_are_asked_for():
    out = io.StringIO()
    runner.run_many(["T01_c06_lr3e5", "T14_f02_lr3e5"], list(R.SPLITS),
                    dry_run=True, out=out)
    text = out.getvalue()
    assert "T01_c06_lr3e5__grouped" in text
    assert "T14_f02_lr3e5__grouped" in text
    assert "__author" not in text


def test_the_leaderboard_expects_no_author_run_from_tuning():
    data = L.collect([], "T")
    assert data["total"] == 26
    for _, split in data["pending"]:
        assert split == "grouped"
    row = {"name": "T01_c06_lr3e5__grouped", "best_val_top1": 0.85,
           "split": "data/splits_grouped"}
    data = L.collect([row], "T")
    text = L.render(data)
    assert "done 1 of 26 runs; 25 still to run" in text
    for line in text.splitlines():
        if line.strip().startswith("T01_c06_lr3e5"):
            assert "pending" not in line
            assert "85.00%" in line
    assert "one split left" not in text


# ---- the loss ---------------------------------------------------------------

class _Items:
    def __init__(self, labels):
        self.items = []
        for label in labels:
            self.items.append(("path", label, "clip"))


def test_no_new_setting_means_the_loss_of_every_earlier_run():
    criterion = T._make_criterion(dict(T.DEFAULTS), _Items([0, 1]), 2)
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.0
    assert criterion.weight is None


def test_label_smoothing_reaches_the_loss():
    cfg = dict(T.DEFAULTS)
    cfg["label_smoothing"] = 0.1
    criterion = T._make_criterion(cfg, _Items([0, 1]), 2)
    assert criterion.label_smoothing == 0.1


def test_class_weights_favour_the_rare_class_and_average_one():
    # three of class 0, one of class 1
    weights = T._class_weights(_Items([0, 0, 0, 1]), 2)
    assert weights.tolist() == pytest.approx([4 / 6, 2.0])
    average = (3 * weights[0] + 1 * weights[1]) / 4
    assert average.item() == pytest.approx(1.0)


def test_class_weights_see_through_a_subset():
    from torch.utils.data import Subset
    data = _Items([0, 0, 0, 1])
    weights = T._class_weights(Subset(data, [0, 3]), 2)
    assert weights.tolist() == pytest.approx([1.0, 1.0])


def test_a_batch_of_only_rare_clips_really_costs_more():
    """torch's weight= divides by the batch's weights, which cancels here."""
    logits = torch.zeros(4, 2)
    rare = torch.ones(4, dtype=torch.long)
    weights = torch.tensor([0.5, 2.0])
    plain = nn.functional.cross_entropy(logits, rare).item()
    ours = T.BalancedLoss(weights)(logits, rare).item()
    theirs = nn.CrossEntropyLoss(weight=weights)(logits, rare).item()
    assert ours == pytest.approx(2.0 * plain)
    assert theirs == pytest.approx(plain)


def test_balanced_loss_keeps_the_smoothing_too():
    logits = torch.tensor([[2.0, 0.0]])
    y = torch.tensor([0])
    weights = torch.tensor([1.0, 1.0])
    want = nn.functional.cross_entropy(logits, y, label_smoothing=0.1).item()
    got = T.BalancedLoss(weights, 0.1)(logits, y).item()
    assert got == pytest.approx(want)


@pytest.fixture
def workspace(tmp_path):
    proc = tmp_path / "processed"
    split = tmp_path / "splits"
    split.mkdir()
    rng = np.random.default_rng(0)
    classes = ["Cover Drive", "Sweep", "Pull"]
    rows = []
    for label, cls in enumerate(classes):
        for k in range(6):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            p = proc / "box" / clip.replace("/", os.sep)[:-4]
            p.parent.mkdir(parents=True, exist_ok=True)
            frames = np.clip(40 + 70 * label + rng.integers(0, 20, (4, 8, 8, 3)),
                             0, 255).astype(np.uint8)
            np.save(str(p) + ".npy", frames)
    for name in ("train", "val"):
        text = "clip,class,label,match\n"
        for clip, cls, label in rows:
            text += f"{clip},{cls},{label},vid{label}\n"
        (split / f"{name}.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(classes), encoding="utf-8")
    return {"name": "t", "split": str(split), "out": str(tmp_path / "exp"),
            "processed": str(proc), "pretrained": False, "backbone": "resnet18",
            "batch_size": 3, "epochs": 1, "patience": 99, "lr": 1e-3,
            "workers": 0, "figures": None}


def test_val_loss_stays_plain_whatever_the_training_loss_is(workspace,
                                                            monkeypatch):
    """Or val loss stops comparing across the ledger."""
    seen = []
    real = T.evaluate

    def spy(model, loader, criterion=None, *args, **kwargs):
        seen.append(criterion)
        return real(model, loader, criterion, *args, **kwargs)

    monkeypatch.setattr(T, "evaluate", spy)
    cfg = dict(workspace)
    cfg["label_smoothing"] = 0.1
    cfg["class_balanced"] = True
    row = T.train(cfg, quiet=True)
    assert row["cfg"]["label_smoothing"] == 0.1
    assert isinstance(seen[0], nn.CrossEntropyLoss)
    assert seen[0].label_smoothing == 0.0
    assert seen[0].weight is None


def test_a_run_with_every_new_setting_trains_and_resumes_its_config(workspace):
    cfg = dict(workspace)
    cfg.update({"flip": 0.5, "colour_jitter": 0.2, "random_crop": 0.9,
                "time_jitter": 1, "label_smoothing": 0.05,
                "class_balanced": True})
    row = T.train(cfg, quiet=True)
    assert row["epochs_run"] == 1
    # an older resume file has none of these keys; they count as their defaults
    old = {}
    full = dict(T.DEFAULTS)
    full.update(workspace)
    for key in full:
        if key not in ("label_smoothing", "class_balanced", "colour_jitter",
                       "random_crop", "time_jitter", "head_dropout"):
            old[key] = full[key]
    assert T._resume_differs(old, full) == []


# ---- backbone_lr on a whole video model --------------------------------------

@pytest.mark.parametrize("name", ["r3d_18", "mvit_v2_s", "swin3d_t"])
def test_backbone_lr_slows_all_but_the_new_last_layer(name):
    model = VideoClassifier(name, 3, pretrained=False)
    cfg = dict(T.DEFAULTS)
    cfg["lr"] = 1e-4
    cfg["backbone_lr"] = 1e-5
    opt = T._make_optimizer(cfg, model)
    assert len(opt.param_groups) == 2
    head, backbone = opt.param_groups
    assert head["name"] == "head" and head["lr"] == 1e-4
    assert backbone["name"] == "backbone" and backbone["lr"] == 1e-5
    # the last layer is 3 classes wide, and it is alone up there
    assert model.last_layer().out_features == 3
    want = []
    for p in model.last_layer().parameters():
        want.append(id(p))
    got = []
    for p in head["params"]:
        got.append(id(p))
    assert sorted(got) == sorted(want)
    # nothing lost, nothing in two groups
    seen = []
    for group in opt.param_groups:
        for p in group["params"]:
            seen.append(id(p))
    everything = []
    for p in model.parameters():
        everything.append(id(p))
    assert sorted(seen) == sorted(everything)


def test_backbone_lr_on_a_model_with_no_split_fails_loudly():
    """It used to do nothing, silently: a run on no change at all."""
    cfg = dict(T.DEFAULTS)
    cfg["backbone_lr"] = 1e-5
    with pytest.raises(ValueError, match="backbone_lr"):
        T._param_groups(cfg, nn.Linear(2, 2))


def test_no_backbone_lr_keeps_one_group():
    model = VideoClassifier("r3d_18", 3, pretrained=False)
    opt = T._make_optimizer(dict(T.DEFAULTS), model)
    assert len(opt.param_groups) == 1


# ---- MViT's own dropout and stochastic depth ---------------------------------

def test_mvit_keeps_its_own_settings_unless_asked():
    """C06's checkpoints say dropout 0.0, which never reached MViT: its own
    0.5 was on all along. head_dropout None must go on meaning that."""
    model = VideoClassifier("mvit_v2_s", 3, pretrained=False)
    assert model.backbone.head[0].p == 0.5
    assert model.backbone.blocks[-1].stochastic_depth.p == pytest.approx(0.2)


def test_mvit_takes_both_settings_and_its_shape_does_not_change():
    plain = VideoClassifier("mvit_v2_s", 3, pretrained=False)
    tuned = VideoClassifier("mvit_v2_s", 3, pretrained=False, head_dropout=0.7,
                            drop_path=0.4)
    assert tuned.backbone.head[0].p == 0.7
    assert tuned.backbone.blocks[-1].stochastic_depth.p == pytest.approx(0.4)
    assert list(plain.state_dict()) == list(tuned.state_dict())


def test_only_mvit_takes_them():
    with pytest.raises(ValueError, match="mvit_v2_s"):
        VideoClassifier("r3d_18", 3, pretrained=False, drop_path=0.4)
    cfg = R.resolve(R.get("C02_r3d_18"), "grouped")
    cfg["drop_path"] = 0.4
    assert "not r3d_18" in video3d.check(cfg)


def test_the_method_hands_the_settings_to_the_model():
    cfg = R.resolve(R.get("T09_c06_drop07"), "grouped")
    cfg["pretrained"] = False
    model = video3d.model(cfg, None, 3)
    assert model.backbone.head[0].p == 0.7
    cfg = R.resolve(R.get("T08_c06_sd04"), "grouped")
    cfg["pretrained"] = False
    model = video3d.model(cfg, None, 3)
    assert model.backbone.blocks[-1].stochastic_depth.p == pytest.approx(0.4)


# ---- the three augmentations -------------------------------------------------

def test_nothing_asked_means_no_transform_and_flip_alone_is_unchanged():
    assert build_transform() is None
    assert build_transform(0.0) is None
    assert isinstance(build_transform(0.5), ClipFlip)


def test_the_steps_run_in_a_fixed_order():
    chain = build_transform(0.5, colour=0.2, crop=0.9, time=1)
    kinds = []
    for step in chain.steps:
        kinds.append(type(step))
    assert kinds == [ClipTimeJitter, ClipCrop, ClipFlip, ClipColour]
    torch.manual_seed(0)
    assert chain(_clip()).shape == _clip().shape


def _normalised(frame_value):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = torch.full((2, 3, 4, 4), frame_value)
    x[:, 0] = frame_value * 0.8
    return (x - mean) / std


def test_colour_changes_the_clip_but_every_frame_alike():
    x = _normalised(0.5)
    torch.manual_seed(1)
    out = ClipColour(0.4)(x)
    assert not torch.equal(out, x)
    assert torch.equal(out[0], out[1])       # one draw for the clip
    assert out.shape == x.shape


def test_colour_stays_a_real_picture():
    x = _normalised(0.95)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for seed in range(20):
        torch.manual_seed(seed)
        back = ClipColour(0.9)(x) * std + mean
        assert back.min() >= -1e-5 and back.max() <= 1 + 1e-5


def test_colour_varies_from_clip_to_clip():
    x = _normalised(0.5)
    torch.manual_seed(2)
    one = ClipColour(0.4)(x)
    two = ClipColour(0.4)(x)
    assert not torch.equal(one, two)


def test_crop_uses_one_box_for_the_whole_clip():
    x = _clip(frames=1, size=32).repeat(3, 1, 1, 1)
    torch.manual_seed(3)
    out = ClipCrop(0.5)(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x)
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])


def test_crop_keeps_at_least_the_area_it_promises():
    # a left-to-right ramp: the stretched result's range shows the box width
    ramp = torch.linspace(0, 1, 100).view(1, 1, 1, 100).repeat(1, 3, 100, 1)
    for seed in range(30):
        torch.manual_seed(seed)
        out = ClipCrop(0.9)(ramp)
        width = (out.max() - out.min()).item()
        assert width >= 0.9 ** 0.5 - 0.03, seed


def test_time_jitter_keeps_the_order_and_stays_near():
    # each frame holds its own index
    x = torch.arange(15, dtype=torch.float32).view(15, 1, 1, 1).repeat(1, 3, 2, 2)
    moved = False
    for seed in range(30):
        torch.manual_seed(seed)
        out = ClipTimeJitter(1)(x)
        assert out.shape == x.shape
        order = out[:, 0, 0, 0].tolist()
        assert order == sorted(order)
        for i, value in enumerate(order):
            assert abs(value - i) <= 2       # sorting can add one to the reach
        if order != list(range(15)):
            moved = True
    assert moved


def test_none_of_them_changes_the_clip_they_were_given():
    x = _normalised(0.5)
    keep = x.clone()
    torch.manual_seed(4)
    ClipColour(0.4)(x)
    ClipCrop(0.8)(x)
    ClipTimeJitter(1)(x)
    assert torch.equal(x, keep)


@pytest.mark.parametrize("kind, value", [
    (ClipColour, 0.0), (ClipColour, 1.5), (ClipCrop, 0.0), (ClipCrop, 1.0),
    (ClipTimeJitter, 0)])
def test_nonsense_values_are_refused(kind, value):
    with pytest.raises(ValueError):
        kind(value)


def test_the_settings_reach_the_training_clips_only():
    cfg = R.resolve(R.get("T11_c06_colour"), "grouped")
    train_set, val_set = video3d.datasets(cfg)
    kinds = []
    for step in train_set.transform.steps:
        kinds.append(type(step))
    assert ClipColour in kinds
    assert val_set.transform is None


# ---- the report --------------------------------------------------------------

def _ledger_row(name, top1):
    return {"name": name, "best_val_top1": top1}


def _write_probs(out, name, right):
    """A fake run over four clips: `right` says which it gets right."""
    folder = os.path.join(out, name)
    os.makedirs(folder, exist_ok=True)
    probs = np.zeros((4, 2), dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    for i, label in enumerate(labels):
        if right[i]:
            probs[i, label] = 0.9
            probs[i, 1 - label] = 0.1
        else:
            probs[i, label] = 0.2
            probs[i, 1 - label] = 0.8
    np.savez(os.path.join(folder, U.CACHE),
             clips=np.array(["a", "b", "c", "d"]), labels=labels, probs=probs,
             classes=np.array(["x", "y"]))


def _board():
    rows = []
    for name, top1 in [("C06_mvitv2_s__grouped", 0.8451),
                       ("C06_mvitv2_s__grouped__seed1", 0.8509),
                       ("C06_mvitv2_s__grouped__seed2", 0.8560),
                       ("F02_multiscale_tf__grouped", 0.8343),
                       ("F02_multiscale_tf__grouped__seed1", 0.8337),
                       ("F02_multiscale_tf__grouped__seed2", 0.8337),
                       ("T05_c06_ls005__grouped", 0.8700),
                       ("T18_f02_ls005__grouped", 0.8400)]:
        rows.append(_ledger_row(name, top1))
    return U._ledger(rows)


def test_base_names_put_the_original_first_and_ignore_other_splits():
    latest = U._ledger([_ledger_row("C06_mvitv2_s__grouped__seed2", 0.8),
                        _ledger_row("C06_mvitv2_s__author", 0.8),
                        _ledger_row("C06_mvitv2_s__grouped__seed1", 0.8),
                        _ledger_row("C06_mvitv2_s__grouped", 0.8),
                        _ledger_row("mimic_author", 0.8)])
    assert U.base_names("C06_mvitv2_s", latest) == [
        "C06_mvitv2_s__grouped", "C06_mvitv2_s__grouped__seed1",
        "C06_mvitv2_s__grouped__seed2"]


def test_the_marks_match_the_plan():
    latest = U._ledger([
        _ledger_row("C06_mvitv2_s__grouped", 1326 / 1569),
        _ledger_row("C06_mvitv2_s__grouped__seed1", 1335 / 1569),
        _ledger_row("C06_mvitv2_s__grouped__seed2", 1343 / 1569),
        _ledger_row("F02_multiscale_tf__grouped", 1309 / 1569),
        _ledger_row("F02_multiscale_tf__grouped__seed1", 1308 / 1569),
        _ledger_row("F02_multiscale_tf__grouped__seed2", 1308 / 1569)])
    c06 = U.rows_for("tune_c06", latest, "no_such_folder")
    f02 = U.rows_for("tune_f02", latest, "no_such_folder")
    assert c06["mean"] == pytest.approx(85.06, abs=0.01)
    assert c06["mark"] == pytest.approx(86.15, abs=0.01)
    # F02 borrows C06's spread, not its own 0.04
    assert f02["sd"] == pytest.approx(0.04, abs=0.01)
    assert f02["mark"] == pytest.approx(84.47, abs=0.01)


def test_a_team_averages_one_run_with_the_mean_of_the_others(tmp_path):
    out = str(tmp_path)
    _write_probs(out, "one", [True, True, False, False])
    _write_probs(out, "two", [False, False, True, True])
    one = U.val_probs("one", out)
    two = U.val_probs("two", out)
    # 0.9 against 0.2 for the right answer: the confident run wins each clip
    assert U.team_top1(one, [two]) == pytest.approx(100.0)
    assert U.team_top1(one, [one]) == pytest.approx(50.0)


def test_the_report_says_what_counts_and_compares_the_team(tmp_path):
    out = str(tmp_path)
    for name in ("C06_mvitv2_s__grouped", "C06_mvitv2_s__grouped__seed1",
                 "C06_mvitv2_s__grouped__seed2"):
        _write_probs(out, name, [True, True, True, False])
    for name in ("F02_multiscale_tf__grouped",
                 "F02_multiscale_tf__grouped__seed1",
                 "F02_multiscale_tf__grouped__seed2"):
        _write_probs(out, name, [True, True, False, True])
    _write_probs(out, "T05_c06_ls005__grouped", [True, True, True, True])

    model = U.rows_for("tune_c06", _board(), out)
    text = "\n".join(U.render(model))
    assert "T05_c06_ls005" in text
    assert "label_smoothing 0.05" in text
    assert "YES" in text                     # 87.00 is over the mark
    assert model["runs"][0]["team"] == pytest.approx(100.0)
    assert model["team_base"] is not None


def test_a_run_not_scored_yet_still_shows_its_val(tmp_path):
    model = U.rows_for("tune_f02", _board(), str(tmp_path))
    assert model["team_base"] is None
    assert model["runs"][0]["team"] is None
    text = "\n".join(U.render(model))
    assert "T18_f02_ls005" in text and "84.00" in text


def test_a_retrained_run_is_not_read_from_its_old_answers(tmp_path):
    out = str(tmp_path)
    _write_probs(out, "run", [True, True, True, True])
    folder = os.path.join(out, "run")
    ckpt = os.path.join(folder, "best.pt")
    with open(ckpt, "wb") as fh:
        fh.write(b"x")
    cache = os.path.join(folder, U.CACHE)
    os.utime(ckpt, (os.path.getmtime(cache) + 100,
                    os.path.getmtime(cache) + 100))
    assert U.val_probs("run", out) is None


def test_the_report_never_opens_a_test_file():
    with open(os.path.join("tools", "tuning.py"), encoding="utf-8") as fh:
        text = fh.read()
    assert "report_test" not in text
    assert "test.csv" not in text
