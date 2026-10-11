"""One figure per run, drawn from history.json.

Four stacked panels sharing the epoch axis, because loss, accuracy, learning
rate and gradient norm are four different scales and a twin y-axis would be a
lie about which curve belongs to which ruler.

The lr cuts get a vertical hairline across every panel. Every sharp bend in
this project's curves has been a plateau cut -- epoch 16 of the box run was
worth 6.5 points -- and without the marker the bend looks like the model
suddenly deciding to learn.

`pin` is a number, not a curve: it sits at 0-2% on a healthy run and the only
interesting thing about it is its peak, so it goes in the subtitle. A panel
showing a flat line at zero would be four panels pretending to be five.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # no display on this machine
import matplotlib.pyplot as plt

__all__ = ["plot_history", "plot_run", "THEMES", "FIGURES"]

# git keeps this one; experiments/ holds the checkpoints and is ignored
FIGURES = os.path.join("notes", "figures", "runs")

# From the data-viz reference palette. Categorical slots 1/2/3 in fixed order;
# the dark column is the same three hues stepped for the dark surface, not a
# flip of the light one.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}

LINE = 1.6          # thin marks; the grid is a hairline under them
MARK = 6.0


def _column(history, key, scale=1.0):
    xs = []
    ys = []
    for row in history:
        value = row.get(key)
        if value is not None:
            xs.append(row["epoch"])
            ys.append(value * scale)
    return xs, ys


def _lr_cuts(history):
    """The epochs where the rate changed. These explain every sharp bend."""
    cuts = []
    last = None
    for row in history:
        lr = row.get("lr")
        if last is not None and lr is not None and lr != last:
            cuts.append(row["epoch"])
        last = lr
    return cuts


def _style_axes(ax, ink, last_row):
    ax.set_facecolor(ink["surface"])
    ax.grid(True, color=ink["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=ink["muted"], labelsize=8, length=3, width=0.8)
    ax.set_xlim(0.5, last_row + 0.5)
    if not last_row:
        return


def _panel_title(ax, text, ink):
    ax.set_title(text, color=ink["secondary"], fontsize=9, loc="left",
                 pad=6)


def plot_history(history, path, name="run", theme="light"):
    """Draw one run's curves. Returns the path written."""
    if not history:
        raise ValueError("no epochs to plot")
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; have {sorted(THEMES)}")

    ink = THEMES[theme]
    blue, orange, aqua = ink["series"]
    last = history[-1]["epoch"]
    cuts = _lr_cuts(history)

    best = history[0]
    for row in history:
        if row["val_top1"] > best["val_top1"]:
            best = row

    peak_pin = 0.0
    for row in history:
        if row.get("pinned"):
            if row["pinned"] > peak_pin:
                peak_pin = row["pinned"]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.5, 1.5],
                                          "hspace": 0.28})
    fig.patch.set_facecolor(ink["surface"])

    for ax in axes:
        _style_axes(ax, ink, last)
        for epoch in cuts:
            ax.axvline(epoch, color=ink["grid"], linewidth=0.8, zorder=0)

    # ---- loss -----------------------------------------------------------
    ax = axes[0]
    ax.plot(*_column(history, "train_loss"), color=blue, linewidth=LINE,
            label="train")
    ax.plot(*_column(history, "val_loss"), color=orange, linewidth=LINE,
            label="validation")
    _panel_title(ax, "Cross-entropy loss", ink)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right",
                    labelcolor=ink["secondary"])
    for text in leg.get_texts():
        text.set_color(ink["secondary"])

    # ---- accuracy -------------------------------------------------------
    ax = axes[1]
    ax.plot(*_column(history, "train_top1", 100), color=blue, linewidth=LINE,
            label="train top-1")
    ax.plot(*_column(history, "val_top1", 100), color=orange, linewidth=LINE,
            label="validation top-1")
    xs3, ys3 = _column(history, "val_top3", 100)
    ax.plot(xs3, ys3, color=aqua, linewidth=LINE, label="validation top-3")
    _panel_title(ax, "Accuracy (%)", ink)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right",
                    labelcolor=ink["secondary"])
    for text in leg.get_texts():
        text.set_color(ink["secondary"])

    # the extreme, direct-labelled: the only number anyone quotes
    ax.plot([best["epoch"]], [100 * best["val_top1"]], marker="o",
            markersize=MARK, color=orange, markeredgecolor=ink["surface"],
            markeredgewidth=1.5, zorder=5)
    # flip the label inward near the right edge, or it runs off the figure
    if best["epoch"] > 0.66 * last:
        side = "right"
        nudge = -8
    else:
        side = "left"
        nudge = 8
    ax.annotate(f"best {100 * best['val_top1']:.2f}%  (epoch {best['epoch']})",
                xy=(best["epoch"], 100 * best["val_top1"]),
                xytext=(nudge, -15), textcoords="offset points", ha=side,
                color=ink["primary"], fontsize=8.5)

    # aqua sits under 3:1 on the light surface, so it carries a visible label
    # rather than relying on its hue -- the relief rule
    if xs3:
        ax.annotate(f"top-3 {ys3[-1]:.1f}%", xy=(xs3[-1], ys3[-1]),
                    xytext=(-4, 7), textcoords="offset points",
                    color=ink["secondary"], fontsize=8, ha="right")

    # ---- learning rate --------------------------------------------------
    ax = axes[2]
    ax.plot(*_column(history, "lr"), color=blue, linewidth=LINE,
            drawstyle="steps-post")
    ax.set_yscale("log")
    _panel_title(ax, "Learning rate", ink)

    # ---- gradient norm --------------------------------------------------
    ax = axes[3]
    xs, ys = _column(history, "grad_norm")
    if xs:
        ax.plot(xs, ys, color=blue, linewidth=LINE)
        _panel_title(ax, "Gradient norm (mean, before clipping)", ink)
    else:
        _panel_title(ax, "Gradient norm -- not recorded for this run", ink)
    ax.set_xlabel("epoch", color=ink["secondary"], fontsize=9)

    # ---- the frame ------------------------------------------------------
    fig.text(0.055, 0.978, name, color=ink["primary"], fontsize=13,
             ha="left", va="top", fontweight="medium")
    subtitle = (f"{len(history)} epochs   "
                f"best validation top-1 {100 * best['val_top1']:.2f}% "
                f"at epoch {best['epoch']}   "
                f"peak GRU saturation {100 * peak_pin:.0f}%")
    if cuts:
        joined = ""
        for epoch in cuts:
            if joined:
                joined += ", "
            joined += str(epoch)
        subtitle += f"   lr cut at epoch {joined}"
    fig.text(0.055, 0.953, subtitle, color=ink["muted"], fontsize=8.5,
             ha="left", va="top")

    fig.subplots_adjust(left=0.075, right=0.975, top=0.918, bottom=0.055)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=ink["surface"])
    plt.close(fig)
    return path


def plot_run(run_dir, name=None, themes=("light", "dark"), mirror=FIGURES):
    """Draw history.json in `run_dir`. Returns the paths written.

    Also writes a copy into `mirror` (notes/figures/runs, which git keeps),
    because experiments/ holds the checkpoints and is ignored -- a figure that
    lives only there does not survive a clone or a disk clear-out.
    """
    source = os.path.join(run_dir, "history.json")
    with open(source, encoding="utf-8") as fh:
        history = json.load(fh)
    if name is None:
        name = os.path.basename(os.path.normpath(run_dir))

    written = []
    for theme in themes:
        if theme == "light":
            leaf = "history.png"
            kept = f"{name}.png"
        else:
            leaf = f"history_{theme}.png"
            kept = f"{name}_{theme}.png"
        written.append(plot_history(history, os.path.join(run_dir, leaf),
                                    name=name, theme=theme))
        if mirror:
            written.append(plot_history(history, os.path.join(mirror, kept),
                                        name=name, theme=theme))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="+",
                    help="an experiments/<name> directory")
    ap.add_argument("--light_only", action="store_true")
    ap.add_argument("--no_mirror", action="store_true",
                    help="skip the tracked copy in " + FIGURES)
    args = ap.parse_args(argv)

    if args.light_only:
        themes = ("light",)
    else:
        themes = ("light", "dark")

    if args.no_mirror:
        mirror = None
    else:
        mirror = FIGURES

    for run_dir in args.run_dir:
        for path in plot_run(run_dir, themes=themes, mirror=mirror):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
