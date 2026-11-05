"""Family T against its baselines: does a change count, alone and in the team?

    python -m tools.tuning            what is already known; no GPU
    python -m tools.tuning --score    first scores what is missing, on val

Val only, grouped only. Test stays closed (rule 5). Use --score when nothing
is training: it needs the GPU for about a minute per run.

Two questions per run (plan, "Decided 2026-11-05"):

    alone   is it above the model's mean by 2 sd? F02's own sd of 0.04
            cannot be trusted, so both models use C06's
    team    how does it do beside every run of the other model, against each
            of this model's own runs in the same place?

A team here is one run of the model being tuned plus the mean of all the
other model's runs, half and half, so a tuned run is always compared with one
run in its place, not three.
"""
import argparse
import os
import sys

import numpy as np

from src.evaluate import probabilities, scores_report
from src.experiments import registry as R
from src.methods.fusion import _average
from src.utils.checkpoint import RESULTS_PATH, read_results
from tools.leaderboard import spread

__all__ = ["base_names", "mark", "team_top1", "val_probs", "rows_for", "render",
           "main"]

SPLIT = "grouped"
MODELS = {"tune_c06": ("C06", "C06_mvitv2_s", "tune_f02"),
          "tune_f02": ("F02", "F02_multiscale_tf", "tune_c06")}
SPREAD_FROM = "tune_c06"
CACHE = "val_probs.npz"


def _ledger(rows):
    """name -> its latest row."""
    latest = {}
    for row in rows:
        latest[row.get("name", "")] = row
    return latest


def base_names(exp_id, latest, split=SPLIT):
    """The original run and every repeat of it on one split, original first."""
    names = []
    seeds = []
    for name in latest:
        parsed = R.parse_name(name)
        if parsed is None or parsed[0] != exp_id or parsed[1] != split:
            continue
        if parsed[2] is None:
            names.append(name)
        else:
            seeds.append((parsed[2], name))
    seeds.sort()
    for _, name in seeds:
        names.append(name)
    return names


def _points(names, latest):
    values = []
    for name in names:
        values.append(100 * latest[name]["best_val_top1"])
    return values


def mark(mean, sd):
    """Where a single run starts to count as a real gain."""
    return mean + 2 * sd


def val_probs(name, out, score=False, split=SPLIT):
    """A run's val probabilities, kept beside its checkpoint. None if they are
    not there yet and score is off."""
    folder = os.path.join(out, name)
    cache = os.path.join(folder, CACHE)
    ckpt = os.path.join(folder, "best.pt")
    fresh = os.path.exists(cache)
    if fresh and os.path.exists(ckpt):
        # a retrained run must not be read from the old one's answers
        fresh = os.path.getmtime(cache) >= os.path.getmtime(ckpt)
    if not fresh:
        if not score:
            return None
        part = probabilities(ckpt, os.path.join(R.SPLITS[split], "val.csv"),
                             batch_size=4)
        np.savez(cache, clips=np.asarray(part["clips"]),
                 labels=np.asarray(part["labels"]), probs=part["probs"],
                 classes=np.asarray(part["classes"]))
    saved = np.load(cache)
    return {"clips": list(saved["clips"]), "labels": saved["labels"],
            "probs": saved["probs"], "classes": list(saved["classes"])}


def team_top1(member, others):
    """Val top-1, in points, of one run beside the mean of the other model's."""
    clips, labels, probs = _average(others)
    partner = {"clips": clips, "labels": labels, "probs": probs,
               "classes": others[0]["classes"]}
    _, labels, probs = _average([member, partner])
    return 100 * scores_report(probs, labels, member["classes"])["top1"]


def _all_probs(names, out, score):
    """Every named run's probabilities, or None if any is missing."""
    parts = []
    for name in names:
        part = val_probs(name, out, score)
        if part is None:
            return None
        parts.append(part)
    return parts


def rows_for(base, latest, out, score=False):
    """One model's baselines and one row per finished tuning run."""
    label, base_id, partner_base = MODELS[base]
    names = base_names(base_id, latest)
    values = _points(names, latest)
    mean, sd = spread(values)
    _, spread_id, _ = MODELS[SPREAD_FROM]
    _, used_sd = spread(_points(base_names(spread_id, latest), latest))

    _, partner_id, _ = MODELS[partner_base]
    partners = _all_probs(base_names(partner_id, latest), out, score)
    own = _all_probs(names, out, score)
    team_base = None
    if partners and own:
        scores = []
        for part in own:
            scores.append(team_top1(part, partners))
        team_base = spread(scores)[0]

    runs = []
    for entry in R.ENTRIES:
        if entry["base"] != base:
            continue
        name = R.run_name(entry["id"], SPLIT)
        if name not in latest:
            continue
        val = 100 * latest[name]["best_val_top1"]
        team = None
        if partners:
            part = val_probs(name, out, score)
            if part is not None:
                team = team_top1(part, partners)
        runs.append({"id": entry["id"], "cfg": entry["cfg"], "val": val,
                     "team": team})
    return {"label": label, "values": values, "mean": mean, "sd": sd,
            "mark": mark(mean, used_sd), "team_base": team_base, "runs": runs}


def _change(cfg):
    parts = []
    for key in cfg:
        parts.append(f"{key} {cfg[key]}")
    return ", ".join(parts)


def _score_of(run):
    return run["val"]


def render(model):
    each = ""
    for value in model["values"]:
        each += f" {value:.2f}"
    lines = [f"{model['label']}: mean {model['mean']:.2f}, sd {model['sd']:.2f}"
             f" over{each}; one run counts from {model['mark']:.2f}"]
    if model["team_base"] is not None:
        lines.append(f"  team with its own runs: {model['team_base']:.2f}")
    if not model["runs"]:
        lines.append("  no tuning runs yet")
        return lines
    lines.append(f"  {'id':<20}{'change':<38}{'val':>7}{'+-mean':>8}"
                 f"{'counts':>8}{'team':>8}{'+-team':>8}")
    ordered = sorted(model["runs"], key=_score_of, reverse=True)
    for run in ordered:
        counts = "no"
        if run["val"] >= model["mark"]:
            counts = "YES"
        team = "-"
        gain = "-"
        if run["team"] is not None:
            team = f"{run['team']:.2f}"
            if model["team_base"] is not None:
                gain = f"{run['team'] - model['team_base']:+.2f}"
        lines.append(f"  {run['id']:<20}{_change(run['cfg']):<38}"
                     f"{run['val']:>7.2f}{run['val'] - model['mean']:>+8.2f}"
                     f"{counts:>8}{team:>8}{gain:>8}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score", action="store_true",
                    help="score what is missing on val; needs the GPU")
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args(argv)

    latest = _ledger(read_results(args.results))
    for base in MODELS:
        for line in render(rows_for(base, latest, args.out, args.score)):
            print(line)
        print()
    if not args.score:
        print("a '-' under team: not scored yet. When nothing is training: "
              "python -m tools.tuning --score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
