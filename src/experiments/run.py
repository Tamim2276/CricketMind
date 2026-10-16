"""Run experiments from the registry: both splits, resumable, test scored.

    python -m src.experiments.run --list
    python -m src.experiments.run --exp P03_r3d_18
    python -m src.experiments.run --exp B01_effnetv2s --split grouped
    python -m src.experiments.run --family P
    python -m src.experiments.run --family A --dry_run

Safe to re-run after a power cut: a finished run is skipped, an interrupted
one (resume.pt there) is resumed, and a run trained but not yet scored is
only scored. A crash stops everything, loudly. The one exception is a figure:
a broken report card is reported and the run goes on, because
`python -m tools.figures` can always draw it again.
"""
import argparse
import json
import os
import shutil
import sys

from src.evaluate import StaleReport, evaluate_checkpoint, load_report
from src.experiments import registry as R
from src.plot import plot_report_run, plot_run
from src.train import DEFAULTS, run_dir, split_name, train
from src.utils.checkpoint import append_result, atomic_write_json, read_results

__all__ = ["main", "print_list", "plan", "run_one", "run_many"]

PARTS = ("val", "test")


def print_list(family=None, out=sys.stdout):
    ready = 0
    shown = 0
    for fam in R.FAMILIES:
        if family is not None and fam != family:
            continue
        chosen = R.entries(fam)
        question, day = R.FAMILIES[fam]
        out.write(f"\n{fam}  {question}  (Day {day}, {len(chosen)} IDs)\n")
        for entry in chosen:
            state = R.status(entry)
            if state == "ready":
                ready += 1
            shown += 1
            out.write(f"  {entry['id']:<20} {state:<37} {entry['why']}\n")
    runs = shown * len(R.SPLITS)
    out.write(f"\n{shown} IDs, {runs} runs ({', '.join(R.SPLITS)}), "
              f"{ready} IDs ready now\n")
    return shown


def _ledger(cfg):
    return os.path.join(cfg["out"], "results.jsonl")


def _rows_named(cfg, name):
    found = []
    for row in read_results(_ledger(cfg)):
        if row.get("name") == name:
            found.append(row)
    return found


def _checkpoint(cfg, rows):
    """The best.pt this run's numbers come from. An adopted run's lives
    in the folder of the run it adopted."""
    if rows and rows[-1].get("checkpoint"):
        return rows[-1]["checkpoint"]
    return os.path.join(run_dir(cfg), "best.pt")


def _same_file(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(
        os.path.abspath(b))


def _report_ok(path, ckpt, split_csv):
    """A report that exists, still matches its checkpoint, and scored
    the split it should have."""
    if not os.path.exists(path) or not os.path.exists(ckpt):
        return False
    try:
        report = load_report(path, ckpt)
    except StaleReport:
        return False
    return _same_file(report["split_csv"], split_csv)


def _missing_reports(cfg, ckpt, folder=None):
    if folder is None:
        folder = run_dir(cfg)
    missing = []
    for part in PARTS:
        path = os.path.join(folder, f"report_{part}.json")
        csv_path = os.path.join(cfg["split"], f"{part}.csv")
        if not _report_ok(path, ckpt, csv_path):
            missing.append(part)
    return missing


def plan(entry, split, force=False, overrides=None):
    """What run_one would do, without doing any of it."""
    cfg = R.resolve(entry, split, overrides)
    name = cfg["name"]
    rows = _rows_named(cfg, name)
    resume = os.path.exists(os.path.join(run_dir(cfg), "resume.pt"))
    source = R.ADOPT.get(name)

    if force:
        action = "train"
    elif resume:
        action = "resume"
    elif rows:
        action = None
    elif source and _rows_named(cfg, source):
        action = "adopt"
    else:
        action = "train"

    if action is None:
        missing = _missing_reports(cfg, _checkpoint(cfg, rows))
    elif action == "adopt":
        # the old run's reports come along if they still check out
        old_dir = os.path.join(cfg["out"], source)
        missing = _missing_reports(cfg, os.path.join(old_dir, "best.pt"),
                                   folder=old_dir)
    else:
        missing = list(PARTS)
    return {"cfg": cfg, "name": name, "action": action, "source": source,
            "rows": rows, "score": missing}


def _differs_from_defaults(cfg):
    lines = []
    for key in sorted(cfg):
        if key == "figures":
            continue
        if cfg[key] != DEFAULTS.get(key):
            lines.append(f"      {key}: {cfg[key]!r}")
    return lines


def _describe(step, entry, out):
    cfg = step["cfg"]
    if step["action"] is None and not step["score"]:
        out.write(f"{step['name']}: done, skip\n")
        return
    doing = []
    if step["action"] == "adopt":
        doing.append(f"adopt {step['source']}")
    elif step["action"]:
        doing.append(step["action"])
    if step["score"]:
        doing.append("score " + " + ".join(step["score"]))
    out.write(f"{step['name']}: {', then '.join(doing)}\n")
    out.write(f"    split  {cfg['split']}\n")
    out.write(f"    delta  {json.dumps(entry['cfg'])}\n")
    out.write("    config (where it differs from DEFAULTS)\n")
    for line in _differs_from_defaults(cfg):
        out.write(line + "\n")


def _adopt(step, out):
    """Reuse an older run with the same config, instead of training again."""
    cfg = step["cfg"]
    source = step["source"]
    old = _rows_named(cfg, source)[-1]
    differ = R.config_differences(old["cfg"], cfg)
    if differ:
        raise SystemExit(f"{step['name']} cannot adopt {source}: "
                         f"{', '.join(differ)} differ")

    old_dir = os.path.join(cfg["out"], source)
    folder = run_dir(cfg)
    os.makedirs(folder, exist_ok=True)
    ckpt = os.path.join(old_dir, "best.pt")

    history = os.path.join(old_dir, "history.json")
    if os.path.exists(history):
        shutil.copyfile(history, os.path.join(folder, "history.json"))
    # its reports carry the checkpoint's hash, so a copy is still checkable
    for part in PARTS:
        report = os.path.join(old_dir, f"report_{part}.json")
        csv_path = os.path.join(cfg["split"], f"{part}.csv")
        if _report_ok(report, ckpt, csv_path):
            shutil.copyfile(report, os.path.join(folder, f"report_{part}.json"))

    row = {}
    for key, value in old.items():
        if key != "recorded_at":
            row[key] = value
    row["name"] = step["name"]
    row["method"] = cfg["method"]
    row["split"] = cfg["split"]
    row["split_name"] = split_name(cfg["split"])
    row["cfg"] = cfg
    row["checkpoint"] = os.path.abspath(ckpt)
    row["adopted_from"] = {"name": source,
                           "recorded_at": old.get("recorded_at")}
    append_result(row, _ledger(cfg))
    if os.path.exists(os.path.join(folder, "history.json")):
        plot_run(folder, name=step["name"], mirror=cfg["figures"])
    if os.path.exists(os.path.join(folder, "report_val.json")):
        _report_card(cfg, out)
    out.write(f"  adopted {source} (recorded {old.get('recorded_at')}, "
              f"val {old['best_val_top1']:.2%})\n")
    return row


def _score(cfg, ckpt, parts, out):
    folder = run_dir(cfg)
    os.makedirs(folder, exist_ok=True)
    for part in parts:
        csv_path = os.path.join(cfg["split"], f"{part}.csv")
        report = evaluate_checkpoint(ckpt, csv_path,
                                     batch_size=cfg["batch_size"],
                                     workers=cfg["workers"])
        atomic_write_json(report, os.path.join(folder, f"report_{part}.json"))
        if part == "test":
            # rule 5: scored and stored once, not read until Days 28-30
            out.write("  test scored and stored, not shown\n")
        else:
            out.write(f"  {part} top-1 {report['top1']:.2%}  "
                      f"macro-F1 {report['macro_f1']:.3f}\n")
    if "val" in parts:
        _report_card(cfg, out)


def _report_card(cfg, out):
    try:
        plot_report_run(run_dir(cfg), name=cfg["name"], mirror=cfg["figures"])
    except Exception as err:
        out.write(f"  (no report card: {err})\n")


def run_one(entry, split, force=False, dry_run=False, overrides=None,
            quiet=False, out=sys.stdout):
    """One ID on one split. Returns what it did: 'skipped', 'trained',
    'resumed', 'adopted', 'scored', or 'planned' for a dry run."""
    step = plan(entry, split, force, overrides)
    if dry_run:
        _describe(step, entry, out)
        return "planned"

    cfg = step["cfg"]
    action = step["action"]
    if action is None and not step["score"]:
        row = step["rows"][-1]
        out.write(f"{step['name']}: already done, skipping "
                  f"(recorded {row.get('recorded_at')}, "
                  f"val {row['best_val_top1']:.2%})\n")
        return "skipped"

    out.write(f"\n== {step['name']} ==\n")
    if action in ("train", "resume"):
        reason = R.status(entry, overrides)
        if reason != "ready":
            raise SystemExit(f"{step['name']} cannot start: {reason}")
        train(cfg, fresh=force, quiet=quiet)
    elif action == "adopt":
        _adopt(step, out)

    rows = _rows_named(cfg, step["name"])
    ckpt = _checkpoint(cfg, rows)
    parts = _missing_reports(cfg, ckpt)
    if force:
        parts = list(PARTS)
    if parts:
        _score(cfg, ckpt, parts, out)

    if action == "train":
        return "trained"
    if action == "resume":
        return "resumed"
    if action == "adopt":
        return "adopted"
    return "scored"


def run_many(ids, splits, force=False, dry_run=False, overrides=None,
             quiet=False, out=sys.stdout):
    """Several IDs, each on each split, in registry order. IDs that cannot
    start are named up front and left out; a crash in one stops the rest."""
    chosen = []
    waiting = []
    for exp_id in ids:
        entry = R.get(exp_id)
        state = R.status(entry, overrides)
        if state == "ready":
            chosen.append(entry)
        else:
            waiting.append((exp_id, state))

    if waiting:
        out.write("not ready, left out:\n")
        for exp_id, state in waiting:
            out.write(f"  {exp_id:<20} {state}\n")

    done = {}
    for entry in chosen:
        for split in splits:
            what = run_one(entry, split, force=force, dry_run=dry_run,
                           overrides=overrides, quiet=quiet, out=out)
            done[what] = done.get(what, 0) + 1

    parts = []
    for what in sorted(done):
        parts.append(f"{done[what]} {what}")
    if not parts:
        parts.append("nothing to run")
    out.write(f"\n{', '.join(parts)}; {len(waiting)} IDs not ready\n")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true",
                    help="every experiment, its status and its question")
    ap.add_argument("--exp", nargs="+", help="one or more experiment IDs")
    ap.add_argument("--family", choices=list(R.FAMILIES))
    ap.add_argument("--split", choices=list(R.SPLITS),
                    help="one split only; default is both")
    ap.add_argument("--dry_run", action="store_true",
                    help="say what would run, and write nothing")
    ap.add_argument("--force", action="store_true",
                    help="train again even if a row exists (needs --exp)")
    args = ap.parse_args(argv)

    if args.list:
        print_list(args.family)
        return 0

    if args.exp:
        ids = args.exp
    elif args.family:
        ids = []
        for entry in R.entries(args.family):
            ids.append(entry["id"])
    else:
        ap.print_help()
        return 0
    if args.force and not args.exp:
        raise SystemExit("--force retrains; name the IDs with --exp")

    if args.split:
        splits = [args.split]
    else:
        splits = list(R.SPLITS)
    run_many(ids, splits, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
