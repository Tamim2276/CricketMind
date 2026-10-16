"""Every comparison figure, rebuilt from the ledger, and the missing report cards.

    python -m tools.figures
    python -m tools.figures --redraw      # redraw every report card too

Writes notes/figures/compare/ and notes/figures/index.md, and overwrites
them on every run: if a figure is wrong, fix this file and rerun, never
edit a PNG. Validation numbers only; test is not drawn before Days 32-33.
Author is always blue and grouped always orange, in every figure.
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments import registry as R
from src.plot import FIGURES, THEMES, plot_report_run, sequential_cmap
from src.utils.checkpoint import RESULTS_PATH, read_results
from tools.leaderboard import collect

__all__ = ["build", "main"]

NOTES = os.path.join("notes", "figures")
COMPARE = os.path.join(NOTES, "compare")
INDEX = os.path.join(NOTES, "index.md")
EXPERIMENTS = "experiments"
SPLIT_SLOT = {"author": 0, "grouped": 1}


# ---- small helpers ---------------------------------------------------------

def _style(ax, ink):
    ax.set_facecolor(ink["surface"])
    ax.grid(True, color=ink["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=ink["muted"], labelsize=8, length=3, width=0.8)


def _title(fig, ink, title, subtitle):
    # fixed inches from the top, so short and tall figures look the same
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / height, title, color=ink["primary"],
             fontsize=12.5, ha="left", va="top", fontweight="medium")
    fig.text(0.02, 1 - 0.42 / height, subtitle, color=ink["muted"],
             fontsize=8.5, ha="left", va="top")


def _panel_title(ax, text, ink):
    ax.set_title(text, color=ink["secondary"], fontsize=9, loc="left", pad=6)


def _save(fig, path, ink):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=ink["surface"])
    plt.close(fig)


def _top_margin(height):
    # room for the title block, whatever the figure height
    return 1 - 0.95 / height


def _val(row):
    return 100 * row["best_val_top1"]


def _ids_with_rows(fam, runs):
    found = []
    for entry in R.entries(fam):
        if (entry["id"], "author") in runs or (entry["id"], "grouped") in runs:
            found.append(entry["id"])
    return found


def _base_id(fam):
    """The family's reference: the entry that changes nothing."""
    for entry in R.entries(fam):
        if not entry["cfg"]:
            return entry["id"]
    return None


# ---- one figure per family -------------------------------------------------

def family_figure(fam, runs, path, theme):
    ink = THEMES[theme]
    ids = _ids_with_rows(fam, runs)
    base = _base_id(fam)
    height = 1.9 + 0.34 * len(ids)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, height), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(ink["surface"])

    values = []
    for key in runs:
        if key[0] in ids:
            values.append(_val(runs[key]))
    low = max(0.0, math.floor(min(values) / 5) * 5 - 5)
    high = min(100.0, math.ceil(max(values) / 5) * 5 + 5)

    for k, split in enumerate(R.SPLITS):
        ax = axes[k]
        _style(ax, ink)
        color = ink["series"][SPLIT_SLOT[split]]
        best = None
        for y, exp_id in enumerate(ids):
            row = runs.get((exp_id, split))
            if row is None:
                continue
            ax.plot([_val(row)], [y], marker="o", markersize=7, color=color,
                    markeredgecolor=ink["surface"], markeredgewidth=1.5,
                    linestyle="none", zorder=3)
            if best is None or _val(row) > best[0]:
                best = (_val(row), y)
        if base is not None and (base, split) in runs:
            x = _val(runs[(base, split)])
            ax.axvline(x, color=ink["muted"], linewidth=1.0, zorder=1)
            ax.annotate(f"base {base}", xy=(x, -0.6), xytext=(3, 0),
                        textcoords="offset points", fontsize=7.5,
                        color=ink["muted"], va="center")
        if best is not None:
            ax.annotate(f"{best[0]:.2f}%", xy=(best[0], best[1]),
                        xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ink["primary"])
        ax.set_xlim(low, high)
        top = -0.6
        if base is not None:
            top = -1.1
        ax.set_ylim(len(ids) - 0.4, top)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=8, color=ink["secondary"])
        ax.set_xlabel("val top-1 (%)", color=ink["secondary"], fontsize=8.5)
        _panel_title(ax, f"{split} split", ink)

    question, day = R.FAMILIES[fam]
    _title(fig, ink, f"Family {fam}: {question}",
           "validation top-1, one dot per experiment; the splits are never"
           " averaged")
    fig.subplots_adjust(left=0.17, right=0.97, top=_top_margin(height),
                        bottom=0.9 / height)
    _save(fig, path, ink)


# ---- author vs grouped, per experiment ------------------------------------

def gap_figure(runs, path, theme):
    ink = THEMES[theme]
    ids = []
    for entry in R.ENTRIES:
        if (entry["id"], "author") in runs and (entry["id"], "grouped") in runs:
            ids.append(entry["id"])
    height = 1.9 + 0.34 * len(ids)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(9, height))
    fig.patch.set_facecolor(ink["surface"])
    _style(ax, ink)

    for y, exp_id in enumerate(ids):
        a = _val(runs[(exp_id, "author")])
        g = _val(runs[(exp_id, "grouped")])
        ax.plot([a, g], [y, y], color=ink["grid"], linewidth=2.0, zorder=1)
    for split in R.SPLITS:
        xs = []
        ys = []
        for y, exp_id in enumerate(ids):
            xs.append(_val(runs[(exp_id, split)]))
            ys.append(y)
        ax.plot(xs, ys, marker="o", markersize=7, linestyle="none",
                color=ink["series"][SPLIT_SLOT[split]],
                markeredgecolor=ink["surface"], markeredgewidth=1.5,
                label=split, zorder=3)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8, color=ink["secondary"])
    ax.set_ylim(len(ids) - 0.5, -0.5)
    ax.set_xlabel("val top-1 (%)", color=ink["secondary"], fontsize=8.5)
    # above the plot, where no dot can ever land
    leg = ax.legend(frameon=False, fontsize=8, loc="lower left",
                    bbox_to_anchor=(0, 1.0), ncol=2)
    for text in leg.get_texts():
        text.set_color(ink["secondary"])
    _title(fig, ink, "Author split against grouped split",
           "same experiment, two questions: seen footage vs unseen matches")
    fig.subplots_adjust(left=0.22, right=0.97, top=_top_margin(height),
                        bottom=0.9 / height)
    _save(fig, path, ink)


# ---- which change helped which shot ---------------------------------------

def per_class_figure(runs, reports, path, theme):
    ink = THEMES[theme]
    import numpy as np

    names = []
    classes = None
    for entry in R.ENTRIES:
        for split in R.SPLITS:
            name = R.run_name(entry["id"], split)
            if name not in reports or (entry["id"], split) not in runs:
                continue
            if entry["id"] not in names:
                names.append(entry["id"])
            if classes is None:
                classes = reports[name]["classes"]

    height = 3.2 + 0.3 * len(names)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(14, height), sharey=True)
    fig.patch.set_facecolor(ink["surface"])

    mesh = None
    for k, split in enumerate(R.SPLITS):
        ax = axes[k]
        ax.set_facecolor(ink["surface"])
        grid = np.full((len(names), len(classes)), np.nan)
        for i, exp_id in enumerate(names):
            name = R.run_name(exp_id, split)
            if name in reports and (exp_id, split) in runs:
                grid[i] = reports[name]["per_class"]["f1"]
        # a run without this split stays blank, the surface showing through
        mesh = ax.pcolormesh(np.ma.masked_invalid(grid),
                             cmap=sequential_cmap(ink), vmin=0, vmax=1,
                             edgecolors=ink["surface"], linewidth=1.0)
        ax.set_ylim(len(names), 0)
        ax.set_yticks([i + 0.5 for i in range(len(names))])
        ax.set_yticklabels(names, fontsize=7.5, color=ink["secondary"])
        ax.set_xticks([i + 0.5 for i in range(len(classes))])
        ax.set_xticklabels(classes, fontsize=7.5, rotation=50, ha="right",
                           rotation_mode="anchor", color=ink["secondary"])
        ax.tick_params(length=0)
        for side in ax.spines:
            ax.spines[side].set_visible(False)
        _panel_title(ax, f"{split} split", ink)

    # its own axis, so it can never sit on top of a panel
    cax = fig.add_axes([0.93, 1.4 / height, 0.012, 1 - 2.35 / height])
    bar = fig.colorbar(mesh, cax=cax, ticks=[0, 0.5, 1])
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=ink["muted"], labelsize=7.5, length=0)
    bar.set_label("F1", color=ink["muted"], fontsize=7.5)
    _title(fig, ink, "F1 per shot class, per experiment",
           "validation; darker means better in light mode, brighter in dark")
    fig.subplots_adjust(left=0.1, right=0.9, top=_top_margin(height),
                        bottom=1.4 / height, wspace=0.06)
    _save(fig, path, ink)


# ---- which runs earn their GPU minutes ------------------------------------

def cost_figure(runs, path, theme):
    ink = THEMES[theme]
    height = 5.0
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, height), sharey=True,
                             sharex=True, gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(ink["surface"])

    spread = []
    for key in runs:
        spread.append(max(runs[key].get("minutes", 0.0), 0.1))
    left = 10 ** math.floor(math.log10(min(spread)))
    right = 10 ** math.ceil(math.log10(max(spread)))
    middle = math.sqrt(left * right)
    ticks = []
    for value in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000):
        if left <= value <= right:
            ticks.append(value)

    for k, split in enumerate(R.SPLITS):
        ax = axes[k]
        _style(ax, ink)
        xs = []
        ys = []
        best = None
        for key in runs:
            if key[1] != split:
                continue
            row = runs[key]
            minutes = max(row.get("minutes", 0.0), 0.1)
            xs.append(minutes)
            ys.append(_val(row))
            if best is None or _val(row) > best[1]:
                best = (minutes, _val(row), key[0])
        if xs:
            ax.plot(xs, ys, marker="o", markersize=7, linestyle="none",
                    color=ink["series"][SPLIT_SLOT[split]],
                    markeredgecolor=ink["surface"], markeredgewidth=1.5)
            if best[0] > middle:
                side = "right"
                nudge = -6
            else:
                side = "left"
                nudge = 6
            ax.annotate(f"{best[2]}  {best[1]:.2f}%", xy=(best[0], best[1]),
                        xytext=(nudge, 8), textcoords="offset points",
                        ha=side, fontsize=8, color=ink["primary"])
        ax.set_xscale("log")
        ax.set_xlim(left, right)
        ax.set_xticks(ticks)
        labels = []
        for value in ticks:
            labels.append(f"{value:g}")
        ax.set_xticklabels(labels)
        ax.minorticks_off()
        ax.set_xlabel("training minutes (log scale)", color=ink["secondary"],
                      fontsize=8.5)
        _panel_title(ax, f"{split} split", ink)
    axes[0].set_ylabel("val top-1 (%)", color=ink["secondary"], fontsize=8.5)
    _title(fig, ink, "Cost against accuracy",
           "one dot per run; frozen-feature runs take minutes, end-to-end"
           " runs hours")
    fig.subplots_adjust(left=0.08, right=0.97, top=_top_margin(height),
                        bottom=0.15)
    _save(fig, path, ink)


# ---- the probe ------------------------------------------------------------

def probe_figure(runs, path, theme):
    ink = THEMES[theme]
    ids = _ids_with_rows("P", runs)

    def by_author(exp_id):
        row = runs.get((exp_id, "author"))
        if row is None:
            return -1.0
        return row["best_val_top1"]
    ids.sort(key=by_author, reverse=True)

    height = 1.9 + 0.42 * len(ids)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, height), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor(ink["surface"])
    for k, split in enumerate(R.SPLITS):
        ax = axes[k]
        _style(ax, ink)
        ax.grid(False, axis="y")
        for y, exp_id in enumerate(ids):
            row = runs.get((exp_id, split))
            if row is None:
                continue
            ax.barh([y], [_val(row)], height=0.62,
                    color=ink["series"][SPLIT_SLOT[split]])
            ax.annotate(f"{_val(row):.2f}%", xy=(_val(row), y),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ink["primary"])
        ax.set_xlim(0, 100)
        ax.set_ylim(len(ids) - 0.5, -0.5)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids, fontsize=8, color=ink["secondary"])
        ax.set_xlabel("val top-1 (%)", color=ink["secondary"], fontsize=8.5)
        _panel_title(ax, f"{split} split", ink)
    _title(fig, ink, "The probe: one Linear on each frozen cache",
           "same head, same training; the only thing that changes is the"
           " features")
    fig.subplots_adjust(left=0.15, right=0.97, top=_top_margin(height),
                        bottom=0.9 / height)
    _save(fig, path, ink)


# ---- report cards and the index -------------------------------------------

def _load_reports(exp_dir):
    reports = {}
    if not os.path.isdir(exp_dir):
        return reports
    for name in sorted(os.listdir(exp_dir)):
        path = os.path.join(exp_dir, name, "report_val.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                reports[name] = json.load(fh)
    return reports


def _report_cards(exp_dir, reports, redraw, mirror):
    drawn = []
    for name in reports:
        folder = os.path.join(exp_dir, name)
        if redraw or not os.path.exists(os.path.join(folder, "report_card.png")):
            plot_report_run(folder, name=name, mirror=mirror)
            drawn.append(name)
    return drawn


def _index(made, runs, earlier, notes):
    def exists(rel):
        return os.path.exists(os.path.join(notes, rel))

    lines = ["# Figures", "",
             "Rebuilt by `python -m tools.figures` from"
             " `experiments/results.jsonl`. Validation numbers only (rule 5)."
             " Do not edit; fix the code and rerun.", "",
             "## Comparisons", ""]
    for stem, caption in made:
        lines.append(f"- [{stem}](compare/{stem}.png) — {caption}"
                     f" ([dark](compare/{stem}_dark.png))")
    if not made:
        lines.append("- none yet")

    lines += ["", "## Per run", "",
              "| run | val top-1 | why | figures |", "|---|---|---|---|"]
    for entry in R.ENTRIES:
        for split in R.SPLITS:
            row = runs.get((entry["id"], split))
            if row is None:
                continue
            name = R.run_name(entry["id"], split)
            links = []
            if exists(f"runs/{name}.png"):
                links.append(f"[curves](runs/{name}.png)")
            if exists(f"runs/{name}_report.png"):
                links.append(f"[report card](runs/{name}_report.png)")
            lines.append(f"| {name} | {_val(row):.2f}% | {entry['why']} | "
                         f"{' · '.join(links)} |")

    if earlier:
        lines += ["", "## Earlier runs, before the registry", "",
                  "| run | val top-1 | figures |", "|---|---|---|"]
        for name in earlier:
            row = earlier[name][0]
            links = []
            if exists(f"runs/{name}.png"):
                links.append(f"[curves](runs/{name}.png)")
            if exists(f"runs/{name}_report.png"):
                links.append(f"[report card](runs/{name}_report.png)")
            lines.append(f"| {name} | {_val(row):.2f}% | {' · '.join(links)} |")
    return "\n".join(lines) + "\n"


def build(results=RESULTS_PATH, exp_dir=EXPERIMENTS, notes=NOTES,
          mirror=FIGURES, redraw=False, out=sys.stdout):
    """Draw everything. Returns the stems of the comparison figures made."""
    data = collect(read_results(results))
    runs = data["runs"]
    reports = _load_reports(exp_dir)
    compare = os.path.join(notes, "compare")

    cards = _report_cards(exp_dir, reports, redraw, mirror)
    out.write(f"report cards: {len(cards)} drawn, "
              f"{len(reports) - len(cards)} already there\n")

    made = []
    jobs = []
    if _ids_with_rows("P", runs):
        jobs.append(("probe", "which cache carries the most shot information?",
                     probe_figure, (runs,)))
    for fam in R.FAMILIES:
        if _ids_with_rows(fam, runs):
            jobs.append((f"family_{fam}",
                         f"family {fam}: did each change beat its base?",
                         family_figure, (fam, runs)))
        else:
            out.write(f"family {fam}: no rows yet\n")
    both = 0
    for entry in R.ENTRIES:
        if (entry["id"], "author") in runs and (entry["id"], "grouped") in runs:
            both += 1
    if both:
        jobs.append(("split_gap", "how much does the author split flatter"
                     " each model?", gap_figure, (runs,)))
    if runs:
        jobs.append(("cost_accuracy", "which runs earn their GPU minutes?",
                     cost_figure, (runs,)))
    with_reports = 0
    for key in runs:
        if R.run_name(key[0], key[1]) in reports:
            with_reports += 1
    if with_reports:
        jobs.append(("per_class_f1", "which change helped which shot?",
                     per_class_figure, (runs, reports)))

    for stem, caption, draw, args in jobs:
        for theme in THEMES:
            leaf = f"{stem}.png"
            if theme != "light":
                leaf = f"{stem}_{theme}.png"
            draw(*args, os.path.join(compare, leaf), theme)
        made.append((stem, caption))
        out.write(f"wrote compare/{stem}.png (+ dark)\n")

    os.makedirs(notes, exist_ok=True)
    with open(os.path.join(notes, "index.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(_index(made, runs, data["earlier"], notes))
    out.write(f"wrote {os.path.join(notes, 'index.md')}\n")
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--redraw", action="store_true",
                    help="redraw every report card, not only missing ones")
    args = ap.parse_args(argv)
    build(redraw=args.redraw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
