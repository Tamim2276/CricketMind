"""M2: the engine report (Day 54). Every engine number in it carries a receipt.

Run from backend/:
    uv run python ../thesis/scripts/engine_report_day54.py ../data/cricket.duckdb

Asks the real engines the questions below, writes every receipt to
thesis/results/engine_report_receipts.jsonl, checks every one of them again on the same
data (Day 48's checker), and only then writes thesis/results/engine_report.md. Reads
the database only. Published figures are cited, never computed; the calibration and the
ML baseline are read from their own result files (Days 53 and 54).
"""

import csv
import sys
import time
from pathlib import Path

import duckdb
from ledger import RESULTS, add_to_ledger, relative, write_safely

from cricketmind_v3.domain.subject import CricketSubject as S
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.pipeline.check_receipts import check_file, write_receipts

REPORT = RESULTS / "engine_report.md"
RECEIPTS = RESULTS / "engine_report_receipts.jsonl"

# E-026: retired players whose every match in a format is in the data, against their
# published records (found by web search on 2026-10-26; the sources are in numbers.md)
PUBLISHED = (
    # player, format, matches, runs, dismissals, average, strike rate
    ("AN Cook", "Test", 161, 12472, 275, 45.35, None),
    ("AN Cook", "ODI", 92, 3204, 88, 36.41, 77.13),
    ("DA Warner", "Test", 112, 8786, 197, 44.59, None),
    ("SPD Smith", "ODI", 170, 5800, 134, 43.28, 86.96),
    ("V Kohli", "Test", 123, 9230, 197, 46.85, 55.57),
)

MIRPUR = "Shere Bangla National Stadium"
KOHLI_PP_PACE = S(
    player="V Kohli", match_format="T20", phase="powerplay", bowler_type="pace", k=6
)
QUESTIONS = (
    # (where it was first reported, what, metric, subject)
    (
        "N-016",
        "Kohli against Bumrah, against everyone else's",
        "sr_vs_average",
        S(player="V Kohli", bowler="JJ Bumrah"),
    ),
    (
        "N-020",
        "Kohli's T20I strike rate",
        "strike_rate",
        S(player="V Kohli", match_format="T20I"),
    ),
    (
        "N-020",
        "Kohli in T20Is, against every other man",
        "sr_vs_average",
        S(player="V Kohli", match_format="T20I", gender="male"),
    ),
    (
        "N-020",
        "Kohli against Australia since 2020, all formats",
        "strike_rate",
        S(player="V Kohli", opposition="Australia", since="2020"),
    ),
    (
        "N-020",
        "Kohli against Australia since 2020, Tests",
        "strike_rate",
        S(player="V Kohli", opposition="Australia", since="2020", match_format="Test"),
    ),
    (
        "N-021",
        "Bumrah's Test wickets",
        "wickets",
        S(bowler="JJ Bumrah", match_format="Test"),
    ),
    (
        "N-021",
        "Bumrah's Test economy",
        "economy",
        S(bowler="JJ Bumrah", match_format="Test"),
    ),
    (
        "N-021",
        "Bumrah's Test strike rate",
        "bowling_strike_rate",
        S(bowler="JJ Bumrah", match_format="Test"),
    ),
    (
        "N-021",
        "Anderson's Test wickets of Kohli",
        "wickets",
        S(bowler="JM Anderson", player="V Kohli", match_format="Test"),
    ),
    (
        "N-021",
        "Kohli's Test average against Anderson",
        "batting_average",
        S(player="V Kohli", bowler="JM Anderson", match_format="Test"),
    ),
    (
        "N-017",
        "Mirpur, men's T20, first-innings average",
        "venue_first_innings_avg",
        S(venue=MIRPUR, match_format="T20", gender="male"),
    ),
    (
        "N-017",
        "Mirpur, men's T20, chasing sides' win share",
        "venue_chasing_win_share",
        S(venue=MIRPUR, match_format="T20", gender="male"),
    ),
    (
        "N-017",
        "Mirpur, men's T20, spin's share of the wickets",
        "venue_wicket_share",
        S(venue=MIRPUR, match_format="T20", gender="male", bowler_type="spin"),
    ),
    (
        "N-018",
        "Bumrah's workload ratio, 31 December 2024",
        "workload_ratio_flag",
        S(bowler="JJ Bumrah", date="2024-12-31"),
    ),
    (
        "N-018",
        "Bumrah's workload ratio, 4 January 2025",
        "workload_ratio_flag",
        S(bowler="JJ Bumrah", date="2025-01-04"),
    ),
    (
        "N-019",
        "India's men: reviews overturned",
        "review_overturn_rate",
        S(team="India", gender="male"),
    ),
    (
        "N-019",
        "Bangladesh's men: reviews overturned",
        "review_overturn_rate",
        S(team="Bangladesh", gender="male"),
    ),
    (
        "N-019",
        "England's men: reviews overturned",
        "review_overturn_rate",
        S(team="England", gender="male"),
    ),
    (
        "N-019",
        "India's women: reviews overturned",
        "review_overturn_rate",
        S(team="India", gender="female"),
    ),
    (
        "N-022",
        "Pakistan beat India, 24 October 2021, by wickets",
        "match_won_by_wickets",
        S(team="Pakistan", opposition="India", date="2021-10-24", gender="male"),
    ),
    (
        "N-022",
        "Bangladesh beat India, 3 November 2019, by wickets",
        "match_won_by_wickets",
        S(team="Bangladesh", opposition="India", date="2019-11-03", gender="male"),
    ),
    (
        "N-022",
        "Bangladesh's T20I win share against India",
        "team_win_share",
        S(team="Bangladesh", opposition="India", match_format="T20I", gender="male"),
    ),
    (
        "N-022",
        "Bangladesh's ODI win share at Mirpur",
        "team_win_share",
        S(team="Bangladesh", venue=MIRPUR, match_format="ODI", gender="male"),
    ),
    (
        "N-026",
        "Kohli out within 6 balls, T20 powerplay against pace",
        "p_dismissal_within_k",
        KOHLI_PP_PACE,
    ),
    (
        "N-026",
        "Tamim Iqbal out within 6 balls, T20 powerplay against spin",
        "p_dismissal_within_k",
        S(
            player="Tamim Iqbal",
            match_format="T20",
            phase="powerplay",
            bowler_type="spin",
            k=6,
        ),
    ),
    (
        "N-026",
        "Bumrah batting, out within 6 balls, T20",
        "p_dismissal_within_k",
        S(player="JJ Bumrah", match_format="T20", k=6),
    ),
    (
        "N-026",
        "Kohli out within 30 balls, Tests",
        "p_dismissal_within_k",
        S(player="V Kohli", match_format="Test", k=30),
    ),
    (
        "N-031",
        "Kohli: Bumrah instead of Shami, next 6 balls, T20",
        "matchup_difference",
        S(
            player="V Kohli",
            bowler="JJ Bumrah",
            other_bowler="Mohammed Shami",
            k=6,
            match_format="T20",
        ),
    ),
    (
        "N-031",
        "Smith: Anderson instead of Root, next 30 balls, Tests",
        "matchup_difference",
        S(
            player="SPD Smith",
            bowler="JM Anderson",
            other_bowler="JE Root",
            k=30,
            match_format="Test",
        ),
    ),
)
FLOOR = 1000  # L-034: every man with at least this many balls faced in T20Is


def main(database: str) -> None:
    started = time.time()
    db = duckdb.connect(database, read_only=True)
    engines = engines_on(db)
    receipts = []
    lines = [
        "# The engine report (M2, Day 54)",
        "",
        "Every engine number below was asked of the real engines on the D-032 snapshot, and",
        "carries a receipt (its line in `engine_report_receipts.jsonl`); every receipt was",
        "checked again on the same data before this report was written.",
        "",
    ]

    lines += published_section(db, engines, receipts)
    say("published records", started)
    lines += questions_section(engines, receipts)
    say("the engines' answers", started)
    lines += near_zero_section(db, engines, receipts)
    say("ranges near zero", started)
    lines += results_section()

    write_receipts(RECEIPTS, receipts)
    said = []
    confirmed, failed = check_file(db, RECEIPTS, say=said.append)
    say(f"receipts: {confirmed} confirmed, {failed} not", started)
    lines += [
        "## The receipts",
        "",
        (
            f"{len(receipts)} receipts written; checked again on the same data: "
            f"**{confirmed} confirmed, {failed} not**."
        ),
        "",
    ]
    write_safely(REPORT, "\n".join(lines) + "\n")
    for path, what in (
        (RECEIPTS, "M2: the report's receipts"),
        (REPORT, "M2: the engine report"),
    ):
        add_to_ledger(
            {
                "experiment": what,
                "data": "real: the D-032 snapshot",
                "script": "thesis/scripts/engine_report_day54.py",
                "receipts_confirmed": confirmed,
                "receipts_not_confirmed": failed,
            },
            path,
        )
        print(f"wrote {relative(path)}")
    if failed:
        sys.exit("some receipts were not confirmed: see the report")


def ask(engines, receipts, metric, subject):
    """The engine's fact and its receipt's line number, or (None, None)."""
    fact = engines.fact(metric, subject)
    if fact is None:
        return None, None
    receipts.append(fact.receipt)
    return fact, len(receipts)


def published_section(db, engines, receipts) -> list[str]:
    lines = [
        "## 1. M1 against published records (E-026)",
        "",
        (
            "Retired players whose every match in a format is in the data (the matches column "
            "is a count of the data, not an engine claim)."
        ),
        "",
        (
            "| player | format | matches (data / published) | runs | dismissals | average "
            "| strike rate | receipts |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for player, match_format, matches, runs, dismissals, average, rate in PUBLISHED:
        subject = S(player=player, match_format=match_format)
        counted = db.execute(
            "SELECT COUNT(DISTINCT m.match_id) FROM people p JOIN players x USING "
            "(person_id) JOIN matches m USING (match_id) WHERE x.name = ? AND "
            "m.match_type = ?",
            [player, match_format],
        ).fetchone()[0]
        found_runs, runs_line = ask(engines, receipts, "runs", subject)
        found_average, average_line = ask(engines, receipts, "batting_average", subject)
        found_rate, rate_line = ask(engines, receipts, "strike_rate", subject)
        outs = round(found_runs.value / found_average.value)
        shown_rate = f"{found_rate.value:.2f}"
        if rate is not None:
            shown_rate += f" / {rate:.2f}"
        lines.append(
            f"| {player} | {match_format} | {counted} / {matches} | "
            f"{found_runs.value:,.0f} / {runs:,} | {outs} / {dismissals} | "
            f"{found_average.value:.2f} / {average:.2f} | {shown_rate} | "
            f"{runs_line}, {average_line}, {rate_line} |"
        )
    lines.append("")
    return lines


def questions_section(engines, receipts) -> list[str]:
    lines = [
        "## 2. The engines' answers to the questions of Days 39 to 52",
        "",
        (
            "Asked again with today's engines; where an engine changed since (D-058 to D-066), "
            "the number may differ from its first report."
        ),
        "",
        "| first reported | question | engine | answer | n | range | receipt |",
        "|---|---|---|---|---|---|---|",
    ]
    for first, what, metric, subject in QUESTIONS:
        fact, line = ask(engines, receipts, metric, subject)
        if fact is None:
            lines.append(f"| {first} | {what} | {metric} | no answer | | | |")
            continue
        interval = ""
        if fact.interval is not None:
            interval = f"{fact.interval[0]:.4g} to {fact.interval[1]:.4g}"
        lines.append(
            f"| {first} | {what} | {fact.receipt.engine} | {fact.value:.4g} | "
            f"{fact.n if fact.n is not None else ''} | {interval} | {line} |"
        )
    lines.append("")
    return lines


def near_zero_section(db, engines, receipts) -> list[str]:
    """L-034: of the real ranges here, how many end within one replay step of zero? For
    a strike-rate difference, a step is one run in his balls: 100 / n."""
    shared = set()
    for (name,) in db.execute(
        "SELECT name FROM players GROUP BY name HAVING COUNT(*) > 1"
    ).fetchall():
        shared.add(name)
    names = []
    for (name,) in db.execute(
        "SELECT x.name FROM deliveries_full d JOIN matches m USING (match_id) "
        "JOIN players x ON x.person_id = d.batter_id WHERE m.match_type = 'T20' AND "
        "m.team_type = 'international' AND m.gender = 'male' AND d.wides = 0 AND NOT "
        "d.super_over GROUP BY x.name HAVING COUNT(*) >= ? ORDER BY x.name",
        [FLOOR],
    ).fetchall():
        if name not in shared:
            names.append(name)
    ranges = 0
    near = []
    across = 0
    for name in names:
        subject = S(player=name, match_format="T20I", gender="male")
        fact, line = ask(engines, receipts, "sr_vs_average", subject)
        if fact is None or fact.interval is None:
            continue
        ranges += 1
        low, high = fact.interval
        step = 100 / fact.n
        if low <= 0 <= high:
            across += 1
        if min(abs(low), abs(high)) <= step:
            near.append(
                f"{name} ({low:+.2f} to {high:+.2f}, step {step:.3f}; receipt {line})"
            )
    lines = [
        "## 3. Ranges that end within one replay step of zero (L-034)",
        "",
        (
            f"Every man with at least {FLOOR:,} balls faced in T20Is whose name nobody else "
            f"shares: his strike rate against every other man's, {ranges} ranges. "
            f"{across} include zero (no detectable difference). **{len(near)} end within one "
            "step of zero**, where another seed could move the verdict:"
        ),
        "",
    ]
    if near:
        for one in near:
            lines.append(f"- {one}")
    else:
        lines.append("- none")
    lines.append("")
    return lines


def results_section() -> list[str]:
    lines = ["## 4. Calibration and the ML baseline (Days 53 and 54)", ""]
    for name, what in (
        ("calibration_day53.csv", "Day 53's calibration"),
        ("ml_baseline.csv", "Day 54's ML baseline"),
    ):
        path = RESULTS / name
        if not path.exists():
            lines.append(f"- {what}: not run yet.")
            continue
        lines.append(f"{what}, per ball, all four kinds (`{name}`, its ledger line):")
        lines.append("")
        lines.append("| model | Brier score | calibration gap |")
        lines.append("|---|---|---|")
        for row in read(path):
            if row["kind"] == "all four" and row.get("measure", "balls") == "balls":
                lines.append(f"| {row['model']} | {row['brier']} | {row['ece']} |")
        lines.append("")
    return lines


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def say(message: str, started: float) -> None:
    print(f"[{time.time() - started:6.0f}s] {message}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
