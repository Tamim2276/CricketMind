"""The ML baseline's inputs (Day 54, D-062): every ball, with only what was known before it.

One row for every ball faced, and every ball a batter was out on, in the four kinds of
cricket of Day 53. A batter's and a bowler's records come from earlier matches only: for
a train ball, his matches before that one; for a test ball, his whole train record, as N9
has it. The match state (score, wickets, balls left) is the state before the ball.
Millions of rows, so DuckDB builds them and hands them over as numpy arrays.
"""

import numpy as np

from cricketmind_v3.domain.balls import NOT_DISMISSALS
from cricketmind_v3.infrastructure.duckdb_repo import IS_OUT, order_case

# Day 53's four kinds of cricket: a code for SQL (no apostrophes), a name, and the rule
KINDS = (
    ("men_t20", "men's T20", "m.match_type IN ('T20', 'IT20') AND m.gender = 'male'"),
    (
        "women_t20",
        "women's T20",
        "m.match_type IN ('T20', 'IT20') AND m.gender = 'female'",
    ),
    ("men_odi", "men's ODI", "m.match_type = 'ODI' AND m.gender = 'male'"),
    ("men_test", "men's Test", "m.match_type = 'Test' AND m.gender = 'male'"),
)


def kind_case() -> str:
    parts = []
    for code, _, condition in KINDS:
        parts.append(f"WHEN {condition} THEN '{code}'")
    return "CASE " + " ".join(parts) + " END"


def features_sql() -> str:
    """The query: one row per ball, its inputs, and whether he was out on it."""
    marks = ", ".join(["?"] * len(NOT_DISMISSALS))
    is_out = IS_OUT.format(marks=marks)
    wicket = f"(d.wickets > 0 AND COALESCE(d.wicket_kind, '') NOT IN ({marks}))"
    return f"""
WITH ball AS (
    SELECT d.match_id, m.date, d.innings, d.over, d.ball_index, {kind_case()} AS kind,
           COALESCE(d.batter_id, d.batter) AS batter,
           COALESCE(d.bowler_id, d.bowler) AS bowler,
           s.faced_before, {order_case("b.slot")} AS batting_order,
           COALESCE(d.phase, 'none') AS phase,
           COALESCE(d.bowler_type, 'unknown') AS bowler_type,
           CAST(COALESCE({is_out}, false) AS INTEGER) AS out,
           CAST(d.wides = 0 AS INTEGER) AS faced,
           d.runs_total, CAST({wicket} AS INTEGER) AS wicket,
           CAST(d.wides = 0 AND d.noballs = 0 AS INTEGER) AS legal,
           m.overs, m.balls_per_over, i.target_runs, i.target_overs,
           m.date >= CAST(? AS DATE) AS test
    FROM deliveries_full d
    JOIN matches m ON m.match_id = d.match_id
    JOIN ball_stages s ON s.match_id = d.match_id AND s.innings = d.innings
         AND s.over = d.over AND s.ball_index = d.ball_index
    JOIN batting_slots b ON b.match_id = d.match_id AND b.innings = d.innings
         AND b.batter = d.batter
    LEFT JOIN innings i ON i.match_id = d.match_id AND i.innings = d.innings
    WHERE NOT d.super_over AND {kind_case()} IS NOT NULL
),
state AS (
    SELECT *,
           COALESCE(SUM(wicket) OVER innings_so_far, 0) AS wickets_down,
           COALESCE(SUM(runs_total) OVER innings_so_far, 0) AS runs_so_far,
           COALESCE(SUM(legal) OVER innings_so_far, 0) AS legal_so_far
    FROM ball
    WINDOW innings_so_far AS (
        PARTITION BY match_id, innings ORDER BY over, ball_index
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
batter_match AS (
    SELECT kind, batter, match_id, MIN(date) AS date, BOOL_OR(test) AS test,
           SUM(out) AS outs, SUM(faced) AS balls
    FROM state WHERE out = 1 OR faced = 1 GROUP BY kind, batter, match_id
),
batter_before AS (
    SELECT kind, batter, match_id, test,
           COALESCE(SUM(CASE WHEN NOT test THEN outs ELSE 0 END) OVER earlier, 0)
               AS outs_before,
           COALESCE(SUM(CASE WHEN NOT test THEN balls ELSE 0 END) OVER earlier, 0)
               AS balls_before
    FROM batter_match
    WINDOW earlier AS (
        PARTITION BY kind, batter ORDER BY date, match_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
),
bowler_match AS (
    SELECT kind, bowler, match_id, MIN(date) AS date, BOOL_OR(test) AS test,
           SUM(out) AS outs, SUM(faced) AS balls
    FROM state WHERE out = 1 OR faced = 1 GROUP BY kind, bowler, match_id
),
bowler_before AS (
    SELECT kind, bowler, match_id,
           COALESCE(SUM(CASE WHEN NOT test THEN outs ELSE 0 END) OVER earlier, 0)
               AS outs_before,
           COALESCE(SUM(CASE WHEN NOT test THEN balls ELSE 0 END) OVER earlier, 0)
               AS balls_before
    FROM bowler_match
    WINDOW earlier AS (
        PARTITION BY kind, bowler ORDER BY date, match_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT st.kind, st.match_id, st.date, st.test, st.batter, st.bowler, st.batting_order,
       st.phase, st.bowler_type, st.innings, st.faced_before, st.wickets_down,
       st.runs_so_far, st.legal_so_far, st.overs, st.balls_per_over,
       st.target_runs, st.target_overs,
       bb.outs_before AS batter_outs, bb.balls_before AS batter_balls,
       wb.outs_before AS bowler_outs, wb.balls_before AS bowler_balls,
       st.out
FROM state st
JOIN batter_before bb ON bb.kind = st.kind AND bb.batter = st.batter
     AND bb.match_id = st.match_id
JOIN bowler_before wb ON wb.kind = st.kind AND wb.bowler = st.bowler
     AND wb.match_id = st.match_id
WHERE st.out = 1 OR st.faced = 1
ORDER BY st.match_id, st.innings, st.over, st.ball_index
"""


def features(db, cutoff: str) -> dict:
    """Every ball's inputs, as numpy arrays by column name. A test ball's records are the
    whole train record, because the earlier matches summed are only train matches."""
    values = [*NOT_DISMISSALS, *NOT_DISMISSALS, cutoff]
    return db.execute(features_sql(), values).fetchnumpy()


# the model's inputs, in order: numbers first, then one column per category
NUMBERS = (
    "batter_outs",
    "batter_balls",
    "batter_rate",  # his record pulled toward his part of the order, as N9 does
    "batter_log_balls",
    "faced_before",
    "bowler_outs",
    "bowler_balls",
    "bowler_rate",
    "bowler_log_balls",
    "innings",
    "wickets_down",
    "runs_so_far",
    "legal_so_far",
    "chasing",
    "runs_needed",
    "balls_left",
    "limited_overs",
)
CATEGORIES = (
    ("kind", ("men_t20", "women_t20", "men_odi", "men_test")),
    ("batting_order", ("top", "middle", "tail")),
    ("bowler_type", ("pace", "spin", "unknown")),
    ("phase", ("powerplay", "middle", "death", "none")),
)


def column_names() -> list[str]:
    names = list(NUMBERS)
    for column, values in CATEGORIES:
        for value in values:
            names.append(f"{column}={value}")
    return names


def matrix(found: dict, batter_priors: dict, bowler_priors: dict) -> np.ndarray:
    """The rows as numbers a model can read (float32). batter_priors holds a prior for
    each (kind, part of the order), bowler_priors one for each kind: each a pair of
    (pretend outs, pretend balls survived)."""
    rows = len(found["out"])
    batter_a = np.zeros(rows)
    batter_b = np.zeros(rows)
    for (kind, order), (outs, survived) in batter_priors.items():
        here = (found["kind"] == kind) & (found["batting_order"] == order)
        batter_a[here] = outs
        batter_b[here] = survived
    bowler_a = np.zeros(rows)
    bowler_b = np.zeros(rows)
    for kind, (outs, survived) in bowler_priors.items():
        here = found["kind"] == kind
        bowler_a[here] = outs
        bowler_b[here] = survived

    batter_outs = _numbers(found["batter_outs"])
    batter_balls = _numbers(found["batter_balls"])
    bowler_outs = _numbers(found["bowler_outs"])
    bowler_balls = _numbers(found["bowler_balls"])
    runs = _numbers(found["runs_so_far"])
    legal = _numbers(found["legal_so_far"])
    target = _numbers(found["target_runs"])
    chasing = ~np.isnan(target)
    # the balls the innings can last: the chase's overs if it has them, else the match's
    overs = _numbers(found["overs"])
    target_overs = _numbers(found["target_overs"])
    overs = np.where(chasing & ~np.isnan(target_overs), target_overs, overs)
    limited = ~np.isnan(overs)
    total = np.where(limited, overs, 0.0) * _numbers(found["balls_per_over"])
    columns = [
        batter_outs,
        batter_balls,
        (batter_a + batter_outs) / (batter_a + batter_b + batter_balls),
        np.log1p(batter_balls),
        _numbers(found["faced_before"]),
        bowler_outs,
        bowler_balls,
        (bowler_a + bowler_outs) / (bowler_a + bowler_b + bowler_balls),
        np.log1p(bowler_balls),
        _numbers(found["innings"]),
        _numbers(found["wickets_down"]),
        runs,
        legal,
        chasing.astype(float),
        np.where(chasing, target - runs, 0.0),
        np.where(limited, np.maximum(total - legal, 0.0), 0.0),
        limited.astype(float),
    ]
    for column, values in CATEGORIES:
        for value in values:
            columns.append((found[column] == value).astype(float))
    return np.column_stack(columns).astype(np.float32)


def _numbers(values) -> np.ndarray:
    # a column as floats, with SQL's NULL as nan
    return np.ma.filled(np.ma.asarray(values).astype(float), np.nan)
