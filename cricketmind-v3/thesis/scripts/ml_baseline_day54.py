"""E-034: Day 54's ML baseline (D-062), as pre-registered (thesis/prereg/day53_calibration.md,
section 7).

Run from backend/:
    uv run python ../thesis/scripts/ml_baseline_day54.py ../data/cricket.duckdb
and, for the rule's third part (20 refits, about 15 minutes, resumable):
    uv run python ../thesis/scripts/ml_baseline_day54.py ../data/cricket.duckdb --ranges

Logistic regression and gradient-boosted trees learn, from every train ball (before
2025-10-15) and only what was known before each ball, the chance he is out on it; then
they are scored per ball on the test year beside the engine's own chance (B, Day 53), on
the same balls. Reads the database only. Writes thesis/results/ml_baseline.csv and
ml_baseline_trials.md, one ledger line each. The fitted models are kept in
data/ml_cache/, so a run cut short picks up where it was.
"""

import csv
import hashlib
import io
import sys
import time

import duckdb
import joblib
import numpy as np
from ledger import PROJECT, RESULTS, add_to_ledger, relative, write_safely
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cricketmind_v3.domain.bootstrap import middle_95
from cricketmind_v3.domain.calibration import fit, posterior
from cricketmind_v3.domain.scoring import (
    BINS,
    bin_of,
    brier,
    ece,
    keeps,
    log_loss,
    paired_bootstrap,
    per_match,
    reliability,
)
from cricketmind_v3.domain.shrinkage import population_prior
from cricketmind_v3.domain.stages import ORDERS, by_batter
from cricketmind_v3.domain.stages import in_order as records_in_order
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.ml_features import KINDS, features, matrix

PREREG = PROJECT / "thesis" / "prereg" / "day53_calibration.md"
PREREG_SHA256 = "1a7070c97c281a7bd2237afeedfb409ec322060d4381ccfa94741b872defff34"
CUTOFF = "2025-10-15"  # the test year starts here (section 1)
VALIDATION = "2024-10-15"  # the last train year, only to stop the trees early
SEED = 42
REFITS = 20  # section 7: refits on resampled training matches
CACHE = PROJECT / "data" / "ml_cache"
# each kind's code, as (format, gender) for the engines' subjects
FORMATS = {
    "men_t20": ("T20", "male"),
    "women_t20": ("T20", "female"),
    "men_odi": ("ODI", "male"),
    "men_test": ("Test", "male"),
}
MODELS = ("B", "logistic", "trees")


def main(database: str, ranges: bool) -> None:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256:
        sys.exit(
            "the pre-registration has changed since it was fingerprinted: not running"
        )
    started = time.time()
    data = prepare(database, started)
    numbers = data["numbers"]
    outcome = data["outcome"]
    train = data["train"]
    logistic = load("day54_logistic.joblib", started)
    if logistic is None:
        logistic = fit_logistic(numbers[train], outcome[train])
        save(logistic, "day54_logistic.joblib", started)
    trees = load("day54_trees.joblib", started)
    if trees is None:
        trees = fit_trees(numbers, outcome, data["early"], data["late"], train)
        save(trees, "day54_trees.joblib", started)
    test = data["test"]
    chances = {
        "logistic": logistic.predict_proba(numbers[test])[:, 1],
        "trees": trees.predict_proba(numbers[test])[:, 1],
    }
    say("test year predicted", started)
    side = side_by_side(data, chances)
    write("ml_baseline.csv", scores_text(side, data["totals"]))
    trials = trials_text(side, logistic, trees)
    write("ml_baseline_trials.md", trials)
    print(trials)
    if ranges:
        write("ml_baseline_ranges.md", honest_ranges(data, trees, side, started))
    say("done", started)


def prepare(database: str, started: float) -> dict:
    """The inputs of every ball, and the engine's Beta curve (B) for each batter."""
    db = duckdb.connect(database, read_only=True)
    repo = DuckDBDeliveryRepository(db)
    batter_priors = {}
    bowler_priors = {}
    engine = {}
    for code, (match_format, gender) in FORMATS.items():
        train = CricketSubject(match_format=match_format, gender=gender, before=CUTOFF)
        records = repo.records(train)
        for order, _ in ORDERS:
            prior = population_prior(by_batter(records_in_order(records, order)))
            batter_priors[(code, order)] = (prior.outs, prior.survived)
        bowlers = {}
        for pair in repo.pairs(train):
            outs, faced = bowlers.get(pair.bowler, (0, 0))
            bowlers[pair.bowler] = (outs + pair.outs, faced + pair.faced)
        prior = population_prior(list(bowlers.values()))
        bowler_priors[code] = (prior.outs, prior.survived)
        fitted = fit(records, [])
        engine[code] = {}
        for batter in fitted.totals:
            curve = posterior(fitted, "B", batter, 1)
            if curve is not None:
                engine[code][batter] = (curve, fitted.totals[batter][1])
        say(f"{code}: priors and the engine's chances", started)

    found = features(db, CUTOFF)
    say(f"{len(found['out']):,} balls read", started)
    test = np.asarray(found["test"]).astype(bool)
    dates = np.asarray(found["date"]).astype("datetime64[D]")
    train = ~test
    late = train & (dates >= np.datetime64(VALIDATION))
    totals = {None: int(test.sum())}
    for code in FORMATS:
        totals[code] = int(((found["kind"] == code) & test).sum())
    return {
        "found": found,
        "numbers": matrix(found, batter_priors, bowler_priors),
        "outcome": np.asarray(found["out"]).astype(int),
        "test": test,
        "train": train,
        "late": late,
        "early": train & ~late,
        "engine": engine,
        "totals": totals,
    }


def side_by_side(data: dict, chances: dict) -> dict:
    """The test balls the engine answers, with every model's chance for each."""
    found = data["found"]
    side = {
        "kinds": [],
        "matches": [],
        "outcomes": [],
        "curves": [],
        "balls": [],
        "rows": [],
        "scored": {"B": []},
    }
    for name in chances:
        side["scored"][name] = []
    for row, i in enumerate(np.flatnonzero(data["test"])):
        kind = found["kind"][i]
        mine = data["engine"][kind].get(found["batter"][i])
        if mine is None:
            continue
        (outs, survived), balls = mine
        side["kinds"].append(kind)
        side["matches"].append(found["match_id"][i])
        side["outcomes"].append(int(data["outcome"][i]))
        side["curves"].append((outs, survived))
        side["balls"].append(balls)
        side["rows"].append(row)
        side["scored"]["B"].append(outs / (outs + survived))
        for name, given in chances.items():
            side["scored"][name].append(float(given[row]))
    return side


def fit_logistic(numbers, outcome):
    # section 7: inputs standardised, default settings
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED))
    model.fit(numbers, outcome)
    return model


def fit_trees(numbers, outcome, early, late, every):
    # section 7: default settings, stopped early on the validation year; then grown
    # again, to the same size, on every train ball
    sizing = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True)
    sizing.fit(numbers[early], outcome[early], X_val=numbers[late], y_val=outcome[late])
    model = HistGradientBoostingClassifier(
        random_state=SEED, early_stopping=False, max_iter=sizing.n_iter_
    )
    model.fit(numbers[every], outcome[every])
    return model


def honest_ranges(data: dict, trees, side: dict, started: float) -> str:
    """The rule's third part: the trees refitted on 20 resamples of the training matches
    (seed 42), so each test ball gets a range that carries the doubt of the fit, set
    beside the engine's own range for one ball (its Beta curve). Each refit is kept in
    the cache, so a power cut costs one refit at most."""
    found = data["found"]
    train = data["train"]
    matches, which = np.unique(
        np.asarray(found["match_id"])[train], return_inverse=True
    )
    rng = np.random.default_rng(SEED)
    draws = []  # all drawn first, so a resumed run draws the same
    for _ in range(REFITS):
        draws.append(rng.integers(0, len(matches), len(matches)))
    test_rows = data["numbers"][data["test"]]
    refits = []
    for number, drawn in enumerate(draws):
        path = CACHE / f"day54_refit_{number:02d}.npy"
        if path.exists():
            refits.append(np.load(path))
            continue
        weights = np.bincount(drawn, minlength=len(matches))[which]
        model = HistGradientBoostingClassifier(
            random_state=SEED, early_stopping=False, max_iter=trees.n_iter_
        )
        model.fit(data["numbers"][train], data["outcome"][train], sample_weight=weights)
        given = model.predict_proba(test_rows)[:, 1]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, given)
        refits.append(given)
        say(f"refit {number + 1} of {REFITS}", started)
    every = np.vstack(refits)
    rows = np.asarray(side["rows"])
    low = np.quantile(every[:, rows], 0.025, axis=0)
    high = np.quantile(every[:, rows], 0.975, axis=0)
    curves = np.asarray(side["curves"])
    curve_low = beta.ppf(0.025, curves[:, 0], curves[:, 1])
    curve_high = beta.ppf(0.975, curves[:, 0], curves[:, 1])
    balls = np.asarray(side["balls"])
    lines = [
        "# Day 54: the trees' honest range (the rule's third part)",
        "",
        (
            f"{REFITS} refits on resampled training matches (seed 42); for every test "
            "ball the engine answers, the middle 95% of the refits' chances, beside the "
            "engine's own 95% range for one ball (its Beta curve). Mean widths, in "
            "percentage points, by how many train balls the batter had:"
        ),
        "",
        "| his train balls | test balls | the trees' range | the engine's range |",
        "|---|---|---|---|",
    ]
    for name, smallest, largest in (
        ("under 100", 0, 100),
        ("100 to 999", 100, 1000),
        ("1,000 or more", 1000, 10**9),
    ):
        here = (balls >= smallest) & (balls < largest)
        trees_width = 100 * float((high - low)[here].mean())
        engine_width = 100 * float((curve_high - curve_low)[here].mean())
        lines.append(
            f"| {name} | {int(here.sum()):,} | {trees_width:.3f} | {engine_width:.3f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def load(name: str, started: float):
    # a model fitted by an earlier run is loaded, not fitted again (a power cut)
    path = CACHE / name
    if not path.exists():
        return None
    say(f"{name}: loaded from the cache", started)
    return joblib.load(path)


def save(model, name: str, started: float) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CACHE / name)
    say(f"{name}: fitted", started)


def scores_text(side: dict, totals: dict) -> str:
    kinds = side["kinds"]
    outcomes = side["outcomes"]
    scored = side["scored"]
    rows = [
        [
            "kind",
            "model",
            "answered",
            "test balls",
            "mean chance",
            "share out",
            "brier",
            "log loss",
            "ece",
            "ece / mean chance",
            "calibrated (rule 3)",
        ]
    ]
    names = []
    for code, name, _ in KINDS:
        names.append((code, name))
    names.append((None, "all four"))
    for code, name in names:
        places = []
        for i, kind in enumerate(kinds):
            if code is None or kind == code:
                places.append(i)
        here = []
        for i in places:
            here.append(outcomes[i])
        for model in MODELS:
            chances = []
            for i in places:
                chances.append(scored[model][i])
            mean = sum(chances) / len(chances)
            gap = ece(reliability(chances, here))
            rows.append(
                [
                    name,
                    model,
                    len(chances),
                    totals[code],
                    round(mean, 5),
                    round(sum(here) / len(here), 5),
                    round(brier(chances, here), 6),
                    round(log_loss(chances, here), 5),
                    round(gap, 5),
                    round(gap / mean, 3),
                    gap <= mean / 10,
                ]
            )
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def trials_text(side: dict, logistic, trees) -> str:
    matches = side["matches"]
    outcomes = side["outcomes"]
    scored = side["scored"]
    tables = {}
    for model in MODELS:
        tables[model] = per_match(
            matches, scored[model], outcomes, bin_of(scored[model]), BINS
        )
    resampled = paired_bootstrap(tables)
    lines = [
        "# Day 54: the ML baseline against the engine (D-062, pre-registration section 7)",
        "",
        (
            f"Every test ball the engine answers: {len(outcomes):,}, in "
            f"{len(set(matches)):,} matches; 1,000 paired resamples of whole matches "
            "(seed 42)."
        ),
        "",
        "| model | Brier score | calibration gap (ECE) |",
        "|---|---|---|",
    ]
    for model in MODELS:
        gap = ece(reliability(scored[model], outcomes))
        lines.append(f"| {model} | {brier(scored[model], outcomes):.6f} | {gap:.5f} |")
    lines.append("")
    lines.append(
        f"The logistic regression ran {int(logistic[-1].n_iter_[0])} iterations; the "
        f"trees, sized on the validation year, have {trees.n_iter_} rounds."
    )
    lines.append("")
    for model in ("logistic", "trees"):
        brier_gaps = []
        ece_gaps = []
        for (new_brier, new_gap), (old_brier, old_gap) in zip(
            resampled[model], resampled["B"], strict=True
        ):
            brier_gaps.append(new_brier - old_brier)
            ece_gaps.append(new_gap - old_gap)
        brier_range = middle_95(brier_gaps)
        ece_range = middle_95(ece_gaps)
        lines.append(f"## {model} against the engine (B)")
        lines.append("")
        lines.append(
            f"- Brier({model}) − Brier(B): 95% range {brier_range[0]:+.7f} to "
            f"{brier_range[1]:+.7f}"
        )
        lines.append(
            f"- ECE({model}) − ECE(B): 95% range {ece_range[0]:+.6f} to "
            f"{ece_range[1]:+.6f}"
        )
        if keeps(brier_range, ece_range):
            lines.append(
                "- Better by more than luck, and calibration not clearly worse: it may "
                "take the engine's place only once it gives an honest range (20 refits "
                "on resampled training matches), the rule's third part."
            )
        else:
            lines.append("- **A baseline:** it does not take the engine's place.")
        lines.append("")
    return "\n".join(lines) + "\n"


def write(name: str, text: str) -> None:
    path = RESULTS / name
    write_safely(path, text)
    add_to_ledger(
        {
            "experiment": "E-034 the ML baseline",
            "data": "real: the D-032 snapshot",
            "script": "thesis/scripts/ml_baseline_day54.py",
            "prereg_sha256": PREREG_SHA256,
        },
        path,
    )
    print(f"wrote {relative(path)}")


def say(message: str, started: float) -> None:
    print(f"[{time.time() - started:6.0f}s] {message}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], "--ranges" in sys.argv)
