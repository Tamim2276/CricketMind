"""The ledger as a table: both splits side by side, and what is still missing.

    python -m tools.leaderboard
    python -m tools.leaderboard --family A
    python -m tools.leaderboard --sort grouped

Val top-1 only. Test is scored by the harness but never shown before
Days 28-30 (rule 5). The splits are never averaged: they answer different
questions, and the gap between them is a result of its own.
"""
import argparse
import sys

from src.experiments import registry as R
from src.train import split_name
from src.utils.checkpoint import RESULTS_PATH, read_results

__all__ = ["collect", "render", "main"]


def _split_of(row):
    if row.get("split_name"):
        return row["split_name"]
    return split_name(row.get("split", ""))


def collect(rows, family=None):
    """Sort ledger rows into registry runs, earlier runs and the rest.

    The latest row wins when a run was recorded more than once."""
    known = set()
    for entry in R.ENTRIES:
        known.add(entry["id"])

    runs = {}            # (id, split) -> latest row
    repeats = {}         # (id, split) -> {seed: latest row}, other seeds
    earlier = {}         # name -> [latest row, how many]
    unknown = {}
    smoke = 0
    for row in rows:
        name = row.get("name", "")
        if name.startswith("smoke_"):
            smoke += 1
            continue
        if "__" in name:
            parsed = R.parse_name(name)
            if parsed is None:
                unknown[name] = row
                continue
            exp_id, split, seed = parsed
            if exp_id not in known or split not in R.SPLITS:
                unknown[name] = row
            elif seed is None:
                runs[(exp_id, split)] = row
            else:
                repeats.setdefault((exp_id, split), {})[seed] = row
            continue
        if name in earlier:
            earlier[name] = [row, earlier[name][1] + 1]
        else:
            earlier[name] = [row, 1]

    pending = []
    total = 0
    for entry in R.ENTRIES:
        if family is not None and R.family_of(entry["id"]) != family:
            continue
        for split in R.SPLITS:
            total += 1
            if (entry["id"], split) not in runs:
                pending.append((entry["id"], split))

    return {"runs": runs, "repeats": repeats, "earlier": earlier,
            "unknown": unknown, "smoke": smoke, "pending": pending,
            "total": total, "family": family}


def spread(values):
    """Mean and sample standard deviation (n - 1), in the values' units."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    total = 0.0
    for v in values:
        total += (v - mean) ** 2
    return mean, (total / (len(values) - 1)) ** 0.5


def _repeat_lines(data):
    """Each repeated run: the original first, then each seed, and the
    mean and sd over all of them, in points. Empty if none to show."""
    lines = []
    for exp_id, split in sorted(data["repeats"]):
        if data["family"] is not None and \
                R.family_of(exp_id) != data["family"]:
            continue
        values = []
        original = data["runs"].get((exp_id, split))
        if original is not None:
            values.append(100 * original["best_val_top1"])
        seeds = data["repeats"][(exp_id, split)]
        for seed in sorted(seeds):
            values.append(100 * seeds[seed]["best_val_top1"])
        mean, sd = spread(values)
        each = ""
        for v in values:
            each += f" {v:.2f}"
        lines.append(f"   {exp_id:<22}{split:>8}{len(values):>6}"
                     f"{mean:>8.2f}{sd:>7.2f}  {each}")
    if not lines:
        return []
    head = ["repeats: the same run with other seeds; mean and sd over all",
            f"   {'id':<22}{'split':>8}{'runs':>6}{'mean':>8}{'sd':>7}"
            f"   each (original first)"]
    return head + lines


def _cell(row):
    if row is None:
        return "pending "
    mark = " "                  # keeps the column straight
    if row.get("adopted_from"):
        mark = "*"
    return f"{row['best_val_top1']:.2%}{mark}"


def _score_for_sort(item):
    return item[0]


def _family_lines(fam, data, sort_by):
    question, day = R.FAMILIES[fam]
    lines = [f"{fam}  {question}  (Day {day})",
             f"   {'id':<22}{'author':>10}{'grouped':>10}{'gap':>8}"]

    chosen = []
    for entry in R.entries(fam):
        author = data["runs"].get((entry["id"], "author"))
        grouped = data["runs"].get((entry["id"], "grouped"))
        if author is None and grouped is None:
            continue
        chosen.append((entry["id"], author, grouped))
    if not chosen:
        return lines + ["   no rows yet"]

    if sort_by:
        keyed = []
        for exp_id, author, grouped in chosen:
            row = author
            if sort_by == "grouped":
                row = grouped
            score = -1.0
            if row is not None:
                score = row["best_val_top1"]
            keyed.append((score, exp_id, author, grouped))
        keyed.sort(key=_score_for_sort, reverse=True)
        chosen = []
        for score, exp_id, author, grouped in keyed:
            chosen.append((exp_id, author, grouped))

    for exp_id, author, grouped in chosen:
        gap = ""
        if author is not None and grouped is not None:
            points = 100 * (author["best_val_top1"] - grouped["best_val_top1"])
            gap = f"{points:+.2f}"
        lines.append(f"   {exp_id:<22}{_cell(author):>10}{_cell(grouped):>10}"
                     f"{gap:>8}")
    return lines


def render(data, sort_by=None):
    lines = ["val top-1; gap = author - grouped, in points;"
             " * = adopted from an earlier run", ""]
    for fam in R.FAMILIES:
        if data["family"] is not None and fam != data["family"]:
            continue
        lines.extend(_family_lines(fam, data, sort_by))
        lines.append("")

    repeat = _repeat_lines(data)
    if repeat:
        lines.extend(repeat)
        lines.append("")

    pending = data["pending"]
    done = data["total"] - len(pending)
    lines.append(f"done {done} of {data['total']} runs; "
                 f"{len(pending)} still to run")
    if pending:
        both = []
        one = []
        seen = {}
        for exp_id, split in pending:
            seen.setdefault(exp_id, []).append(split)
        for exp_id in seen:
            if len(seen[exp_id]) == len(R.SPLITS):
                both.append(exp_id)
            else:
                one.append(f"{exp_id} ({seen[exp_id][0]})")
        if one:
            lines.append("  one split left: " + ", ".join(one))
        if both:
            lines.append(f"  not started ({len(both)}): " + ", ".join(both))

    if data["family"] is None:
        if data["earlier"]:
            lines.append("")
            lines.append("earlier runs, before the registry (latest of each):")
            for name in data["earlier"]:
                row, count = data["earlier"][name]
                runs = ""
                if count > 1:
                    runs = f"  ({count} runs)"
                lines.append(f"   {name:<22}{_split_of(row):>10}"
                             f"{row['best_val_top1']:>10.2%}{runs}")
        if data["unknown"]:
            lines.append("")
            lines.append("not in the registry: " + ", ".join(data["unknown"]))
        lines.append("")
        lines.append(f"{data['smoke']} smoke-test rows skipped")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=list(R.FAMILIES))
    ap.add_argument("--sort", choices=("author", "grouped"),
                    help="best first by this split; default is registry order")
    ap.add_argument("--results", default=RESULTS_PATH)
    args = ap.parse_args(argv)

    data = collect(read_results(args.results), args.family)
    print(render(data, args.sort))
    return 0


if __name__ == "__main__":
    sys.exit(main())
