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

__all__ = ["plot_history", "plot_run", "plot_report", "plot_report_run",
           "sequential_cmap", "THEMES", "FIGURES"]

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
        # blue 100 -> 700, low to high
        "ramp": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                 "#184f95", "#0d366b"),
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
        # blue 700 -> 250: on a dark surface, near zero is the dark end
        "ramp": ("#0d366b", "#104281", "#184f95", "#1c5cab", "#2a78d6",
                 "#5598e7", "#86b6ef"),
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


def plot_run(run_dir, name=None, themes=("light", "dark"), mirror=FIGURES,
             mirror_themes=("light",)):
    """Draw history.json in `run_dir`. Returns the paths written.

    Also writes a copy into `mirror` (notes/figures/runs, which git keeps),
    because experiments/ holds the checkpoints and is ignored -- a figure that
    lives only there does not survive a clone or a disk clear-out. Only the
    light copy by default: 92 runs in two themes is 60 MB of PNGs in git.
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
        if mirror and theme in mirror_themes:
            written.append(plot_history(history, os.path.join(mirror, kept),
                                        name=name, theme=theme))
    return written


def sequential_cmap(ink):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ramp", list(ink["ramp"]))


def _confusion_panel(ax, report, ink):
    """Each row as a percentage of its true class, so small classes show."""
    import numpy as np

    matrix = np.array(report["confusion"], dtype=float)
    support = matrix.sum(axis=1)
    pct = np.zeros_like(matrix)
    for i in range(len(matrix)):
        if support[i] > 0:
            pct[i] = 100.0 * matrix[i] / support[i]

    n = len(report["classes"])
    # a 1px surface gap between cells, not a border
    mesh = ax.pcolormesh(pct, cmap=sequential_cmap(ink), vmin=0, vmax=100,
                         edgecolors=ink["surface"], linewidth=1.0)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")

    rows = []
    for i in range(n):
        rows.append(f"{report['classes'][i]} (n={int(support[i])})")
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(rows, fontsize=7.5, color=ink["secondary"])
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(report["classes"], fontsize=7.5, rotation=50,
                       ha="right", rotation_mode="anchor",
                       color=ink["secondary"])
    ax.tick_params(length=0)
    for side in ax.spines:
        ax.spines[side].set_visible(False)

    # numbers only where they matter: the diagonal, and confusions of 10%+
    for i in range(n):
        for j in range(n):
            value = pct[i][j]
            if i != j and value < 10:
                continue
            if value == 0:
                continue
            if value > 55:
                color = ink["surface"]
            else:
                color = ink["primary"]
            ax.text(j + 0.5, i + 0.5, f"{value:.0f}", ha="center",
                    va="center", fontsize=6.5, color=color)
    ax.set_xlabel("predicted", color=ink["secondary"], fontsize=8.5)
    ax.set_ylabel("true class (val clips)", color=ink["secondary"],
                  fontsize=8.5)
    _panel_title(ax, "Confusion, % of each true class", ink)
    return mesh


def _f1_panel(ax, report, ink):
    """Per-class F1, worst at the top; the worst three get their value."""
    f1 = report["per_class"]["f1"]
    order = []
    for i in range(len(f1)):
        order.append(i)

    def by_f1(i):
        return f1[i]
    order.sort(key=by_f1)

    names = []
    values = []
    for i in order:
        names.append(report["classes"][i])
        values.append(f1[i])
    ys = list(range(len(values)))
    ax.barh(ys, values, height=0.68, color=ink["series"][0])
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.5, color=ink["secondary"])
    ax.set_ylim(len(values) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_facecolor(ink["surface"])
    ax.grid(True, axis="x", color=ink["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ink["axis"])
    ax.tick_params(colors=ink["muted"], labelsize=8, length=0)
    for k in range(min(3, len(values))):
        ax.annotate(f"{values[k]:.2f}", xy=(values[k], k), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color=ink["primary"])
    _panel_title(ax, "F1 per class", ink)


def plot_report(report, path, name="run", theme="light"):
    """One run's report card: the confusion matrix beside per-class F1."""
    if theme not in THEMES:
        raise ValueError(f"unknown theme {theme!r}; have {sorted(THEMES)}")
    ink = THEMES[theme]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2),
                             gridspec_kw={"width_ratios": [2.1, 1],
                                          "wspace": 0.42})
    fig.patch.set_facecolor(ink["surface"])
    axes[0].set_facecolor(ink["surface"])
    mesh = _confusion_panel(axes[0], report, ink)
    _f1_panel(axes[1], report, ink)
    bar = fig.colorbar(mesh, ax=axes[0], fraction=0.03, pad=0.02,
                       ticks=[0, 50, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=ink["muted"], labelsize=7.5, length=0)
    bar.set_label("% of true class", color=ink["muted"], fontsize=7.5)

    fig.text(0.02, 0.975, name, color=ink["primary"], fontsize=13,
             ha="left", va="top", fontweight="medium")
    subtitle = (f"validation   top-1 {100 * report['top1']:.2f}%   "
                f"macro-F1 {report['macro_f1']:.3f}   "
                f"{report['clips']} clips")
    fig.text(0.02, 0.94, subtitle, color=ink["muted"], fontsize=8.5,
             ha="left", va="top")

    fig.subplots_adjust(left=0.17, right=0.975, top=0.88, bottom=0.17)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=ink["surface"])
    plt.close(fig)
    return path


def plot_report_run(run_dir, name=None, themes=("light", "dark"),
                    mirror=FIGURES, mirror_themes=("light",)):
    """Report card from `run_dir`/report_val.json. Validation only: test is
    never drawn before Days 32-33 (rule 5). Returns the paths written."""
    with open(os.path.join(run_dir, "report_val.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    if name is None:
        name = os.path.basename(os.path.normpath(run_dir))

    written = []
    for theme in themes:
        if theme == "light":
            leaf = "report_card.png"
        else:
            leaf = f"report_card_{theme}.png"
        written.append(plot_report(report, os.path.join(run_dir, leaf),
                                   name=name, theme=theme))
        if mirror and theme in mirror_themes:
            kept = os.path.join(mirror, f"{name}_report.png")
            written.append(plot_report(report, kept, name=name, theme=theme))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="+",
                    help="an experiments/<name> directory")
    ap.add_argument("--light_only", action="store_true")
    ap.add_argument("--no_mirror", action="store_true",
                    help="skip the tracked copy in " + FIGURES)
    ap.add_argument("--report", action="store_true",
                    help="draw the report card from report_val.json instead")
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
        if args.report:
            written = plot_report_run(run_dir, themes=themes, mirror=mirror)
        else:
            written = plot_run(run_dir, themes=themes, mirror=mirror)
        for path in written:
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
