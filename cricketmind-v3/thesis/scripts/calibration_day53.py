"""E-007: Day 53's calibration run, exactly as pre-registered.

Run from backend/:
    uv run python ../thesis/scripts/calibration_day53.py ../data/cricket.duckdb

Reads the database only. Writes, in thesis/results/: calibration_day53.csv (every score),
reliability_day53.csv (the plot's points), trials_day53.md (the two trials, by the rules),
engine_check_day53.csv (the engine itself against this run); and
thesis/figures/reliability_day53.svg. One ledger line for each. Refuses to run if the
pre-registration has changed since its SHA-256 was written in the journal.
"""

import csv
import hashlib
import io
import random
import sys
import time

import duckdb
from ledger import PROJECT, RESULTS, add_to_ledger, relative, write_safely

from cricketmind_v3.domain.bootstrap import middle_95
from cricketmind_v3.domain.calibration import (
    K,
    ball_chance,
    displayed_middle,
    fit,
    window_chance,
    windows,
)
from cricketmind_v3.domain.out_within import Stage, out_within_stages
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
from cricketmind_v3.domain.stages import STARTS, stage_of
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository

PREREG = PROJECT / "thesis" / "prereg" / "day53_calibration.md"
PREREG_SHA256 = "1a7070c97c281a7bd2237afeedfb409ec322060d4381ccfa94741b872defff34"
CUTOFF = "2025-10-15"  # the test year is on or after it; the train matches before it
KINDS = (
    ("men's T20", "T20", "male"),
    ("women's T20", "T20", "female"),
    ("men's ODI", "ODI", "male"),
    ("men's Test", "Test", "male"),
)
TRIAL_MODELS = ("D", "A", "B", "C")  # scored on the balls every one of them answers
CHECKS = 30  # moments where the engine itself is asked (section 6)
FIGURE = PROJECT / "thesis" / "figures" / "reliability_day53.svg"


def main(database: str) -> None:
    found = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if found != PREREG_SHA256:
        sys.exit(
            "the pre-registration has changed since it was fingerprinted: not running"
        )
    started = time.time()
    db = duckdb.connect(database, read_only=True)
    repo = DuckDBDeliveryRepository(db)
    data = {}  # kind -> what each model said about each test ball and window
    fitted_by_kind = {}
    for label, match_format, gender in KINDS:
        train = CricketSubject(match_format=match_format, gender=gender, before=CUTOFF)
        test = CricketSubject(match_format=match_format, gender=gender, since=CUTOFF)
        fitted = fit(repo.records(train), repo.pairs(train))
        fitted_by_kind[label] = fitted
        moments = repo.moments(test)
        data[label] = score_kind(fitted, moments)
        say(f"{label}: {len(moments)} test balls", started)

    rows = scores(data)
    write("calibration_day53.csv", csv_text(rows), "E-007 calibration: scores")
    write(
        "reliability_day53.csv",
        reliability_text(data),
        "E-007 calibration: plot points",
    )
    trials = trials_text(data, started)
    write("trials_day53.md", trials, "D-056, D-059: the trials")
    write_figure(data)
    checks = engine_check(db, repo, fitted_by_kind["men's T20"])
    write("engine_check_day53.csv", checks, "Day 53: the engine against the run")
    print(trials)
    say("done", started)


def score_kind(fitted, moments) -> dict:
    """Each model's chance for every test ball and window, side by side."""
    balls = {"match": [], "out": [], "chances": {}}
    spans = {"match": [], "out": [], "balls": [], "chances": {}, "shown": []}
    for model in (*TRIAL_MODELS, "E"):
        balls["chances"][model] = []
        spans["chances"][model] = []
    for moment in moments:
        balls["match"].append(moment.match)
        balls["out"].append(int(moment.out))
        for model in (*TRIAL_MODELS, "E"):
            balls["chances"][model].append(ball_chance(fitted, model, moment))
    for window in windows(moments):
        spans["match"].append(window.start.match)
        spans["out"].append(int(window.out))
        spans["balls"].append(window.balls)
        for model in (*TRIAL_MODELS, "E"):
            spans["chances"][model].append(window_chance(fitted, model, window))
        spans["shown"].append(displayed_middle(fitted, window))
    return {"balls": balls, "windows": spans}


def common(part: dict, models: tuple[str, ...]) -> list[int]:
    # the places where every one of these models gives a chance
    found = []
    for i in range(len(part["out"])):
        every = True
        for model in models:
            if part["chances"][model][i] is None:
                every = False
        if every:
            found.append(i)
    return found


def pick(values: list, places: list[int]) -> list:
    found = []
    for i in places:
        found.append(values[i])
    return found


def scores(data: dict) -> list[list]:
    """One row per kind (and all four pooled), measure and model."""
    rows = [
        [
            "kind",
            "measure",
            "model",
            "answered",
            "of",
            "mean chance",
            "share out",
            "brier",
            "log loss",
            "ece",
            "ece / mean chance",
            "calibrated (rule 3)",
        ]
    ]
    for kind in (*data, "all four"):
        for measure in ("balls", "windows"):
            parts = []
            if kind == "all four":
                for one in data.values():
                    parts.append(one[measure])
            else:
                parts.append(data[kind][measure])
            for models, names in (
                (TRIAL_MODELS, TRIAL_MODELS),
                (("B", "E"), ("B", "E")),  # T2 beside B, on the balls T2 answers
            ):
                outcomes = []
                chances = {}
                for name in models:
                    chances[name] = []
                total = 0
                for part in parts:
                    total += len(part["out"])
                    places = common(part, models)
                    outcomes.extend(pick(part["out"], places))
                    for name in models:
                        chances[name].extend(pick(part["chances"][name], places))
                for name in names:
                    label = name
                    if models == ("B", "E"):
                        label = name + ", on the balls E answers"
                    row = score_row(
                        kind, measure, label, chances[name], outcomes, total
                    )
                    if row is not None:
                        rows.append(row)
            if measure == "windows":
                outcomes = []
                shown = []
                total = 0
                for part in parts:
                    total += len(part["out"])
                    for i in range(len(part["out"])):
                        if part["shown"][i] is not None:
                            outcomes.append(part["out"][i])
                            shown.append(part["shown"][i])
                row = score_row(
                    kind, measure, "B shown (the middle)", shown, outcomes, total
                )
                if row is not None:
                    rows.append(row)
    return rows


def score_row(kind, measure, model, chances, outcomes, total) -> list | None:
    if not chances:
        return None
    mean = sum(chances) / len(chances)
    observed = sum(outcomes) / len(outcomes)
    gap = ece(reliability(chances, outcomes))
    return [
        kind,
        measure,
        model,
        len(chances),
        total,
        round(mean, 5),
        round(observed, 5),
        round(brier(chances, outcomes), 6),
        round(log_loss(chances, outcomes), 5),
        round(gap, 5),
        round(gap / mean, 3),
        gap <= mean / 10,
    ]


def trials_text(data: dict, started: float) -> str:
    """The two trials, decided by rules 1 and 2 on every test ball of the four kinds."""
    matches = []
    outcomes = []
    chances = {}
    for name in TRIAL_MODELS:
        chances[name] = []
    for one in data.values():
        part = one["balls"]
        places = common(part, TRIAL_MODELS)
        matches.extend(pick(part["match"], places))
        outcomes.extend(pick(part["out"], places))
        for name in TRIAL_MODELS:
            chances[name].extend(pick(part["chances"][name], places))
    tables = {}
    for name in ("A", "B", "C"):
        tables[name] = per_match(
            matches, chances[name], outcomes, bin_of(chances[name]), BINS
        )
    resampled = paired_bootstrap(tables)
    say("bootstrap done", started)

    lines = [
        "# Day 53: the two trials, decided by the pre-registered rules",
        "",
        (
            f"Every test ball of the four kinds that every model answers: "
            f"{len(outcomes):,}; in {len(set(matches)):,} matches, resampled 1,000 times "
            "(seed 42)."
        ),
        "",
        "| model | Brier score | calibration gap (ECE) |",
        "|---|---|---|",
    ]
    for name in TRIAL_MODELS:
        gap = ece(reliability(chances[name], outcomes))
        lines.append(f"| {name} | {brier(chances[name], outcomes):.6f} | {gap:.5f} |")
    lines.append("")
    place_kept = trial(
        lines, "1. Batting place (D-059): B against A", resampled, "B", "A"
    )
    winner = "A"
    if place_kept:
        winner = "B"
    trial(lines, f"2. Stages (D-056): C against {winner}", resampled, "C", winner)
    return "\n".join(lines) + "\n"


def trial(lines, title, resampled, new, old) -> bool:
    brier_gaps = []
    ece_gaps = []
    for (new_brier, new_gap), (old_brier, old_gap) in zip(
        resampled[new], resampled[old], strict=True
    ):
        brier_gaps.append(new_brier - old_brier)
        ece_gaps.append(new_gap - old_gap)
    brier_range = middle_95(brier_gaps)
    ece_range = middle_95(ece_gaps)
    kept = keeps(brier_range, ece_range)
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        f"- Brier({new}) − Brier({old}): 95% range {brier_range[0]:+.7f} to "
        f"{brier_range[1]:+.7f}"
    )
    lines.append(
        f"- ECE({new}) − ECE({old}): 95% range {ece_range[0]:+.6f} to {ece_range[1]:+.6f}"
    )
    if kept:
        lines.append(f"- **Kept:** {new} replaces {old}.")
    else:
        lines.append(f"- **Not kept:** {old} stays.")
    lines.append("")
    return kept


def engine_check(db, repo, fitted) -> str:
    """Section 6: the engine itself, asked as if it were 2025-10-14, against this run."""
    moments = repo.moments(
        CricketSubject(match_format="T20", gender="male", since=CUTOFF)
    )
    names = {}
    for person, name in db.execute("SELECT person_id, name FROM players").fetchall():
        names[person] = name
    shared = set()
    for (name,) in db.execute(
        "SELECT name FROM players GROUP BY name HAVING COUNT(*) > 1"
    ).fetchall():
        shared.add(name)
    rng = random.Random(42)
    engines = engines_on(db)
    rows = [["batter", "faced so far", "engine", "this run", "gap"]]
    tries = 0
    while len(rows) <= CHECKS and tries < 10 * CHECKS:
        tries += 1
        moment = moments[rng.randrange(len(moments))]
        name = names.get(moment.batter)
        mine = bulk_answer(fitted, moment.batter, moment.faced_before)
        if name is None or name in shared or mine is None:
            continue
        subject = CricketSubject(
            player=name,
            match_format="T20",
            before=CUTOFF,
            k=K,
            faced_so_far=moment.faced_before,
        )
        fact = engines.fact("p_dismissal_within_k", subject)
        if fact is None:
            continue
        rows.append(
            [
                name,
                moment.faced_before,
                round(fact.value, 5),
                round(mine, 5),
                round(fact.value - mine, 5),
            ]
        )
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def bulk_answer(fitted, batter, so_far) -> float | None:
    # the engine's staged answer, made from this run's priors (he is in his own prior)
    if batter not in fitted.usual:
        return None
    usual = fitted.usual[batter]
    stages = []
    for number in range(stage_of(so_far + 1), stage_of(so_far + K) + 1):
        prior = fitted.stage.get((usual, number))
        if prior is None:
            return None
        outs, faced = fitted.by_stage.get((batter, number), (0, 0))
        stages.append(
            Stage(first=STARTS[number - 1], outs=outs, faced=faced, prior=prior)
        )
    found = out_within_stages(stages, K, so_far)
    if found is None:
        return None
    return found.chance


def reliability_text(data: dict) -> str:
    rows = [["kind", "measure", "model", "bin", "count", "mean chance", "share out"]]
    for kind, one in data.items():
        for measure in ("balls", "windows"):
            part = one[measure]
            places = common(part, TRIAL_MODELS)
            outcomes = pick(part["out"], places)
            for name in TRIAL_MODELS:
                bins = reliability(pick(part["chances"][name], places), outcomes)
                for number, point in enumerate(bins, start=1):
                    rows.append(
                        [
                            kind,
                            measure,
                            name,
                            number,
                            point.count,
                            round(point.chance, 5),
                            round(point.observed, 5),
                        ]
                    )
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def write_figure(data: dict) -> None:
    """The reliability plot: for each kind, per ball and per window, B and C against the
    diagonal (a perfect forecast). Plain SVG, made by this script only."""
    width = 220
    gap = 30
    parts = []
    for row, measure in enumerate(("balls", "windows")):
        for column, kind in enumerate(data):
            part = data[kind][measure]
            places = common(part, TRIAL_MODELS)
            outcomes = pick(part["out"], places)
            points = {}
            top = 0.0
            for name in ("B", "C"):
                points[name] = reliability(
                    pick(part["chances"][name], places), outcomes
                )
                for point in points[name]:
                    top = max(top, point.chance, point.observed)
            top = top * 1.1
            x0 = gap + column * (width + gap)
            y0 = gap + row * (width + 2 * gap)
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{width}" height="{width}" '
                'fill="none" stroke="#999"/>'
            )
            parts.append(
                f'<line x1="{x0}" y1="{y0 + width}" x2="{x0 + width}" y2="{y0}" '
                'stroke="#bbb" stroke-dasharray="4 3"/>'
            )
            for name, colour in (("B", "#1f77b4"), ("C", "#d62728")):
                for point in points[name]:
                    x = x0 + width * point.chance / top
                    y = y0 + width - width * point.observed / top
                    parts.append(
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colour}"/>'
                    )
            parts.append(
                f'<text x="{x0}" y="{y0 - 8}" font-size="12">{kind}, per '
                f"{measure[:-1]} (axes 0 to {top:.2f})</text>"
            )
    total_width = gap + len(data) * (width + gap)
    total_height = 2 * (width + 2 * gap) + gap
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" '
        f'height="{total_height}" font-family="sans-serif">'
        '<text x="30" y="16" font-size="12">Chance given (across) against share out '
        "(up); blue B, red C; the dashed line is a perfect forecast</text>"
        + "".join(parts)
        + "</svg>\n"
    )
    write_safely(FIGURE, svg)
    add_to_ledger(
        {
            "experiment": "E-007 calibration: reliability plot",
            "data": "real",
            "script": "thesis/scripts/calibration_day53.py",
        },
        FIGURE,
    )


def csv_text(rows: list[list]) -> str:
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    return out.getvalue()


def write(name: str, text: str, experiment: str) -> None:
    path = RESULTS / name
    write_safely(path, text)
    add_to_ledger(
        {
            "experiment": experiment,
            "data": "real: the D-032 snapshot",
            "script": "thesis/scripts/calibration_day53.py",
            "prereg_sha256": PREREG_SHA256,
        },
        path,
    )
    print(f"wrote {relative(path)}")


def say(message: str, started: float) -> None:
    print(f"[{time.time() - started:6.0f}s] {message}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
