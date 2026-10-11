"""The figure is a deliverable, so it gets the same treatment as the metrics:
checked for the things that would make it quietly wrong or quietly missing."""
import json
import os

import numpy as np
import pytest

from src.plot import FIGURES, THEMES, plot_history, plot_run


def history(n=12, cut_at=8):
    rows = []
    for i in range(1, n + 1):
        if i < cut_at:
            lr = 1e-4
        else:
            lr = 1e-5
        rows.append({
            "epoch": i,
            "train_loss": 2.0 - 0.1 * i,
            "train_top1": 0.3 + 0.03 * i,
            "val_loss": 2.1 - 0.09 * i,
            "val_top1": 0.32 + 0.025 * i,
            "val_top3": 0.60 + 0.02 * i,
            "lr": lr,
            "pinned": 0.01,
            "dead": 0.0,
            "grad_norm": 20.0 - 0.2 * i,
            "clipped": 0,
            "steps": 100,
            "secs": 400.0,
        })
    return rows


# ---- it draws something, in both themes ---------------------------------

def test_a_figure_is_written(tmp_path):
    out = str(tmp_path / "history.png")
    assert plot_history(history(), out) == out
    assert os.path.getsize(out) > 5000       # not a blank canvas


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_every_theme_renders(tmp_path, theme):
    out = str(tmp_path / (theme + ".png"))
    plot_history(history(), out, theme=theme)
    assert os.path.getsize(out) > 5000


def test_an_unknown_theme_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown theme"):
        plot_history(history(), str(tmp_path / "x.png"), theme="solarized")


def test_the_light_and_dark_themes_are_different_pictures(tmp_path):
    light = str(tmp_path / "l.png")
    dark = str(tmp_path / "d.png")
    plot_history(history(), light, theme="light")
    plot_history(history(), dark, theme="dark")
    assert open(light, "rb").read() != open(dark, "rb").read()


def test_the_dark_theme_is_its_own_steps_not_a_flip():
    """The reference palette steps the same hues for the dark surface; if
    these were equal somebody has copied the light values across."""
    assert THEMES["light"]["series"] != THEMES["dark"]["series"]
    assert len(THEMES["light"]["series"]) == 3
    assert len(THEMES["dark"]["series"]) == 3


# ---- the things that would make it wrong --------------------------------

def test_an_empty_history_is_refused(tmp_path):
    """Better than a blank axes nobody notices is empty."""
    with pytest.raises(ValueError, match="no epochs"):
        plot_history([], str(tmp_path / "x.png"))


def test_a_single_epoch_still_draws(tmp_path):
    """An interrupted run has one row, and that is when you most want to look."""
    out = str(tmp_path / "one.png")
    plot_history(history(n=1), out)
    assert os.path.getsize(out) > 5000


def test_a_run_from_before_the_diagnostics_existed_still_draws(tmp_path):
    """history.json from the older runs has no grad_norm or pinned keys."""
    rows = history()
    for row in rows:
        del row["grad_norm"]
        del row["pinned"]
    out = str(tmp_path / "old.png")
    plot_history(rows, out)
    assert os.path.getsize(out) > 5000


def test_plot_run_reads_history_json_and_names_the_files(tmp_path):
    run = tmp_path / "myrun"
    run.mkdir()
    (run / "history.json").write_text(json.dumps(history()), encoding="utf-8")

    written = plot_run(str(run), mirror=str(tmp_path / "kept"))
    assert os.path.join(str(run), "history.png") in written
    assert os.path.join(str(run), "history_dark.png") in written
    for path in written:
        assert os.path.exists(path)


def test_the_directory_is_created_if_it_is_not_there(tmp_path):
    out = str(tmp_path / "nested" / "deeper" / "history.png")
    plot_history(history(), out)
    assert os.path.exists(out)


# ---- a finished run leaves a figure behind ------------------------------

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def tiny_run(tmp_path):
    """A real dataset small enough to train twice in a second."""
    proc = tmp_path / "processed"
    split = tmp_path / "splits"
    split.mkdir()
    rng = np.random.default_rng(0)

    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(6):
            clip = cls + "/vid" + str(label) + str(k) + "_0.avi"
            rows.append((clip, cls, label))
            target = proc / "box" / clip.replace("/", os.sep)[:-4]
            target.parent.mkdir(parents=True, exist_ok=True)
            base = 40 + 70 * label
            frames = np.clip(base + rng.integers(0, 20, (4, 8, 8, 3)),
                             0, 255).astype(np.uint8)
            np.save(str(target) + ".npy", frames)

    header = "clip,class,label,match"
    for name in ("train", "val"):
        lines = [header]
        for clip, cls, label in rows:
            lines.append(clip + "," + cls + "," + str(label) + ",vid" + str(label))
        (split / (name + ".csv")).write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")

    return {"name": "t", "split": str(split), "out": str(tmp_path / "exp"),
            "processed": str(proc), "pretrained": False,
            "backbone": "resnet18", "batch_size": 3, "epochs": 2,
            "patience": 99, "lr": 1e-3, "workers": 0,
            "figures": str(tmp_path / "kept")}


def test_a_finished_training_run_writes_its_figure(tiny_run):
    """The point of the whole file: nobody has to remember to draw it."""
    from src import train as T

    T.train(tiny_run, quiet=True)
    out = os.path.join(tiny_run["out"], tiny_run["name"])
    assert os.path.exists(os.path.join(out, "history.png"))
    assert os.path.exists(os.path.join(out, "history_dark.png"))


def test_a_broken_figure_does_not_fail_the_run(tiny_run, monkeypatch):
    """Six hours of training must not be lost to a drawing bug."""
    from src import train as T

    def boom(*args, **kwargs):
        raise RuntimeError("matplotlib exploded")

    monkeypatch.setattr(T, "plot_run", boom)
    row = T.train(tiny_run, quiet=True)

    assert row["best_val_top1"] >= 0.0          # the run still returned
    out = os.path.join(tiny_run["out"], tiny_run["name"])
    assert os.path.exists(os.path.join(out, "history.json"))
    assert not os.path.exists(os.path.join(out, "resume.pt"))


# ---- the tracked copy ---------------------------------------------------

def test_a_mirror_copy_is_written_under_the_run_name(tmp_path):
    """experiments/ is gitignored, so a figure that lives only there is lost
    on a clone. The kept copy is named after the run, not 'history'."""
    run = tmp_path / "mimic_author_seg"
    run.mkdir()
    (run / "history.json").write_text(json.dumps(history()), encoding="utf-8")
    kept = tmp_path / "kept"

    plot_run(str(run), mirror=str(kept))

    assert os.path.exists(str(kept / "mimic_author_seg.png"))
    assert os.path.exists(str(kept / "mimic_author_seg_dark.png"))


def test_the_mirror_can_be_turned_off(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "history.json").write_text(json.dumps(history()), encoding="utf-8")

    written = plot_run(str(run), mirror=None, themes=("light",))

    assert written == [os.path.join(str(run), "history.png")]


def test_the_default_mirror_is_the_tracked_directory():
    """If this moves back under an ignored path the figures stop surviving."""
    assert FIGURES == os.path.join("notes", "figures", "runs")


def test_a_finished_run_puts_its_figure_somewhere_git_keeps(tiny_run, tmp_path):
    from src import train as T

    kept = str(tmp_path / "kept")
    tiny_run["figures"] = kept
    T.train(tiny_run, quiet=True)

    assert os.path.exists(os.path.join(kept, "t.png"))
    assert os.path.exists(os.path.join(kept, "t_dark.png"))
