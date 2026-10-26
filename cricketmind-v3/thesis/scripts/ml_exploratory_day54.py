"""EXPLORATORY, not pre-registered (Day 54): where does the trees' gain come from?

Run from backend/ after ml_baseline_day54.py:
    uv run python ../thesis/scripts/ml_exploratory_day54.py ../data/cricket.duckdb

The pre-registered trees (section 7) also see the match: the score, the wickets, the
balls left, the bowler. A question to the engine ("Kohli out within 6 balls?") carries
none of that. Here the same trees learn from only what such a question can carry: his
record, his part of the order, the kind of cricket, the phase and the bowler's type.
Writes thesis/results/ml_exploratory_day54.md and one ledger line. Decides nothing.
"""

import sys
import time

from ledger import RESULTS, add_to_ledger, relative, write_safely
from ml_baseline_day54 import (
    fit_trees,
    load,
    prepare,
    save,
    say,
    side_by_side,
)

from cricketmind_v3.domain.bootstrap import middle_95
from cricketmind_v3.domain.scoring import (
    BINS,
    bin_of,
    brier,
    ece,
    paired_bootstrap,
    per_match,
    reliability,
)
from cricketmind_v3.infrastructure.ml_features import column_names

# what a question to the engine can carry
KEPT = (
    "batter_outs",
    "batter_balls",
    "batter_rate",
    "batter_log_balls",
    "kind=",
    "batting_order=",
    "bowler_type=",
    "phase=",
)


def main(database: str) -> None:
    started = time.time()
    data = prepare(database, started)
    columns = []
    for number, name in enumerate(column_names()):
        for kept in KEPT:
            if name == kept or (kept.endswith("=") and name.startswith(kept)):
                columns.append(number)
    numbers = data["numbers"][:, columns]
    trees = load(data["cache"] / "day54_trees_question_only.joblib", started)
    if trees is None:
        trees = fit_trees(
            numbers, data["outcome"], data["early"], data["late"], data["train"]
        )
        save(trees, data["cache"] / "day54_trees_question_only.joblib", started)
    given = trees.predict_proba(numbers[data["test"]])[:, 1]
    side = side_by_side(data, {"question only": given})
    outcomes = side["outcomes"]
    tables = {}
    for name in ("B", "question only"):
        chances = side["scored"][name]
        tables[name] = per_match(
            side["matches"], chances, outcomes, bin_of(chances), BINS
        )
    resampled = paired_bootstrap(tables)
    brier_gaps = []
    ece_gaps = []
    for (new_brier, new_gap), (old_brier, old_gap) in zip(
        resampled["question only"], resampled["B"], strict=True
    ):
        brier_gaps.append(new_brier - old_brier)
        ece_gaps.append(new_gap - old_gap)
    low, high = middle_95(brier_gaps)
    gap_low, gap_high = middle_95(ece_gaps)
    lines = [
        "# Day 54, EXPLORATORY (not pre-registered): the trees with only the question",
        "",
        (
            "The same trees as section 7, given only what a question to the engine can carry: "
            f"{', '.join(KEPT)}. On the {len(outcomes):,} test balls the engine answers:"
        ),
        "",
        "| model | Brier score | calibration gap (ECE) |",
        "|---|---|---|",
    ]
    for name in ("B", "question only"):
        chances = side["scored"][name]
        lines.append(
            f"| {name} | {brier(chances, outcomes):.6f} | "
            f"{ece(reliability(chances, outcomes)):.5f} |"
        )
    lines += [
        "",
        f"- Brier(question only) − Brier(B): 95% range {low:+.7f} to {high:+.7f}",
        f"- ECE(question only) − ECE(B): 95% range {gap_low:+.6f} to {gap_high:+.6f}",
        (
            f"- The trees have {trees.n_iter_} rounds. This decides nothing; it says where "
            "the pre-registered trees' gain comes from."
        ),
        "",
    ]
    path = RESULTS / "ml_exploratory_day54.md"
    write_safely(path, "\n".join(lines) + "\n")
    add_to_ledger(
        {
            "experiment": "Day 54 EXPLORATORY: the trees with only the question",
            "data": "real: the D-032 snapshot",
            "script": "thesis/scripts/ml_exploratory_day54.py",
        },
        path,
    )
    print("\n".join(lines))
    print(f"wrote {relative(path)}")
    say("done", started)


if __name__ == "__main__":
    main(sys.argv[1])
