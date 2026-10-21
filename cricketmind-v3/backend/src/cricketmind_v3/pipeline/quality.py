"""Checking the data against itself (Day 32).

Each check is a rule and a query that lists the rows breaking it. Some rules must hold
everywhere (a problem if any row breaks them); others only count something real but
unusual (a rain-shortened powerplay), which the paper reports.

Run from backend/:
    uv run python -m cricketmind_v3.pipeline.quality ../data/cricket.duckdb
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True, kw_only=True)
class Check:
    """One rule the data should keep, and the query for the rows that break it."""

    name: str
    rule: str  # in plain words
    must_be_zero: bool  # False: a real-world count to report, not an error
    sql: str


@dataclass(frozen=True, kw_only=True)
class Outcome:
    check: Check
    count: int
    examples: list  # up to five rows that break the rule


# a team's score in an innings: its balls plus penalty runs awarded outside them
SCORES = """
    SELECT i.match_id, i.innings, i.team,
           COALESCE(SUM(d.runs_total), 0) + i.penalty_pre + i.penalty_post AS runs
    FROM innings i
    LEFT JOIN deliveries d ON d.match_id = i.match_id AND d.innings = i.innings
    GROUP BY i.match_id, i.innings, i.team, i.penalty_pre, i.penalty_post
"""

LIMITED = "('T20', 'IT20', 'ODI', 'ODM')"

CHECKS = (
    Check(
        name="ball totals",
        rule="a ball's total is the batter's runs plus the extras",
        must_be_zero=True,
        sql="""
            SELECT match_id, innings, over, ball_index, runs_batter, runs_extras, runs_total
            FROM deliveries WHERE runs_total <> runs_batter + runs_extras
        """,
    ),
    Check(
        name="extras add up",
        rule="the extras are the wides, no-balls, byes, leg-byes and penalties",
        must_be_zero=True,
        sql="""
            SELECT match_id, innings, over, ball_index, runs_extras
            FROM deliveries
            WHERE runs_extras <> wides + noballs + byes + legbyes + penalty
        """,
    ),
    Check(
        name="targets",
        rule="a chase's target is the first innings' score plus one (no rain rule)",
        must_be_zero=True,
        # a target for fewer overs than the match had was reset for rain, even when
        # the result names no rain rule ("no result"): found on Day 32
        sql=f"""
            WITH scores AS ({SCORES})
            SELECT i.match_id, i.target_runs, s.runs + 1 AS first_innings_plus_one
            FROM innings i
            JOIN scores s ON s.match_id = i.match_id AND s.innings = 1
            JOIN matches m ON m.match_id = i.match_id
            WHERE i.innings = 2 AND NOT i.super_over AND i.target_runs IS NOT NULL
              AND m.method IS NULL
              AND (m.overs IS NULL OR i.target_overs IS NULL OR i.target_overs >= m.overs)
              AND i.target_runs <> s.runs + 1
        """,
    ),
    Check(
        name="winning margins",
        rule="'won by N runs' is the first innings' score minus the second's",
        must_be_zero=True,
        sql=f"""
            WITH scores AS ({SCORES})
            SELECT m.match_id, m.won_by_runs, a.runs - b.runs AS difference
            FROM matches m
            JOIN scores a ON a.match_id = m.match_id AND a.innings = 1
            JOIN scores b ON b.match_id = m.match_id AND b.innings = 2
            WHERE m.match_type IN {LIMITED} AND m.method IS NULL
              AND m.won_by_runs IS NOT NULL AND m.winner = a.team
              AND m.won_by_runs <> a.runs - b.runs
        """,
    ),
    Check(
        name="ball counts",
        rule="every loaded match has as many balls stored as were counted",
        must_be_zero=True,
        sql="""
            SELECT l.match_id, l.deliveries, COALESCE(d.balls, 0) AS stored
            FROM loaded l
            LEFT JOIN (SELECT match_id, COUNT(*) AS balls FROM deliveries GROUP BY match_id) d
                ON d.match_id = l.match_id
            WHERE l.deliveries <> COALESCE(d.balls, 0)
        """,
    ),
    Check(
        name="match rows",
        rule="every loaded match has a row in the matches table",
        must_be_zero=True,
        sql="""
            SELECT match_id FROM loaded
            WHERE match_id NOT IN (SELECT match_id FROM matches)
        """,
    ),
    Check(
        name="empty names",
        rule="every ball names a batter, a bowler and a non-striker",
        must_be_zero=True,
        sql="""
            SELECT match_id, innings, over, ball_index FROM deliveries
            WHERE TRIM(batter) = '' OR TRIM(bowler) = '' OR TRIM(non_striker) = ''
        """,
    ),
    Check(
        name="names without an id",
        rule="every batter and bowler has a register id in that match",
        must_be_zero=True,
        sql="""
            SELECT DISTINCT d.match_id, d.name FROM (
                SELECT match_id, batter AS name FROM deliveries
                UNION SELECT match_id, bowler AS name FROM deliveries
            ) d
            LEFT JOIN people p ON p.match_id = d.match_id AND p.name = d.name
            WHERE p.person_id IS NULL
        """,
    ),
    Check(
        name="unreadable matches",
        rule="every match file could be read",
        must_be_zero=True,
        sql="SELECT match_id, error FROM problems",
    ),
    Check(
        name="overs with too many legal balls",
        rule="(a count) overs with more legal balls than the format allows",
        must_be_zero=False,
        sql="""
            SELECT d.match_id, d.innings, d.over, COUNT(*) AS legal_balls
            FROM deliveries d JOIN matches m ON m.match_id = d.match_id
            WHERE d.wides = 0 AND d.noballs = 0
            GROUP BY d.match_id, d.innings, d.over, m.balls_per_over
            HAVING COUNT(*) > m.balls_per_over
        """,
    ),
    Check(
        name="powerplays unlike the rule",
        rule="(a count) innings whose own powerplay ends in another over than the "
        "rule says: shortened matches",
        must_be_zero=False,
        sql=f"""
            SELECT i.match_id, i.innings, i.powerplay_to, m.match_type
            FROM innings i JOIN matches m ON m.match_id = i.match_id
            WHERE i.powerplay_to IS NOT NULL AND NOT i.super_over
              AND m.balls_per_over = 6 AND m.match_type IN {LIMITED}
              AND FLOOR(i.powerplay_to) <> CASE
                  WHEN m.match_type IN ('T20', 'IT20') THEN 5 ELSE 9 END
        """,
    ),
    Check(
        name="balls with two wickets",
        rule="(a count) balls on which more than one wicket fell",
        must_be_zero=False,
        sql="SELECT match_id, innings, over, ball_index FROM deliveries WHERE wickets > 1",
    ),
)


def run(db) -> list[Outcome]:
    """Every check, with how many rows break it and a few examples."""
    outcomes = []
    for check in CHECKS:
        count = db.execute(f"SELECT COUNT(*) FROM ({check.sql})").fetchone()[0]
        examples = db.execute(f"{check.sql} LIMIT 5").fetchall()
        outcomes.append(Outcome(check=check, count=count, examples=examples))
    return outcomes


def report(outcomes: list[Outcome]) -> str:
    """The outcomes as a Markdown table, problems first in words."""
    lines = ["| Check | Rule | Rows | Verdict |", "|---|---|---|---|"]
    for outcome in outcomes:
        if not outcome.check.must_be_zero:
            verdict = "counted"
        elif outcome.count == 0:
            verdict = "ok"
        else:
            verdict = "PROBLEM"
        lines.append(
            f"| {outcome.check.name} | {outcome.check.rule} | {outcome.count} | {verdict} |"
        )
    for outcome in outcomes:
        if outcome.count:
            lines.append("")
            lines.append(f"Examples, {outcome.check.name}: {outcome.examples}")
    return "\n".join(lines)


def main(arguments: list[str]) -> None:
    if len(arguments) != 1:
        print("usage: python -m cricketmind_v3.pipeline.quality <database>")
        sys.exit(2)
    with duckdb.connect(str(Path(arguments[0])), read_only=True) as db:
        print(report(run(db)))


if __name__ == "__main__":
    main(sys.argv[1:])
