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
    earlier = {}         # name -> [latest row, how many]
    unknown = {}
    smoke = 0
    for row in rows:
        name = row.get("name", "")
        if name.startswith("smoke_"):
            smoke += 1
            continue
        if "__" in name:
            exp_id, split = name.split("__", 1)
            if exp_id in known and split in R.SPLITS:
                runs[(exp_id, split)] = row
            else:
                unknown[name] = row
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

    return {"runs": runs, "earlier": earlier, "unknown": unknown,
            "smoke": smoke, "pending": pending, "total": total,
            "family": family}


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
