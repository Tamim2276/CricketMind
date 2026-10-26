"""Tables made from the deliveries, ready for the engines of Phase 4 (Day 36).

Run from backend/ after the ingest:
    uv run python -m cricketmind_v3.pipeline.derived ../data/cricket.duckdb
"""

import sys

import duckdb

# kinds of "wicket" that are not dismissals: the batter can come back
NOT_OUT = "('retired hurt', 'retired not out')"

# how long the batter had been in at each ball: the balls he had faced in the innings
# before it (N9, Day 50). A wide is not a ball faced; a no-ball is. Worked out on the
# whole innings, so a filter (the middle overs) does not start him again
BALL_STAGES = """
CREATE OR REPLACE TABLE ball_stages AS
SELECT match_id, innings, over, ball_index,
       COALESCE(SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END) OVER (
           PARTITION BY match_id, innings, batter
           ORDER BY over, ball_index
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS faced_before
FROM deliveries
WHERE NOT super_over
"""

# where each batter batted in each innings: the order they came in, from the first ball
# they were seen on at either end. A ball is one number that sorts as the balls do:
# over x 1000 + ball x 2, plus 1 at the non-striker's end, so on the first ball the
# striker comes first. Normal is for batters who bat where he bats (N9, Day 50, L-045)
BATTING_SLOTS = """
CREATE OR REPLACE TABLE batting_slots AS
WITH seen AS (
    SELECT match_id, innings, batter AS name, MIN(over * 1000 + ball_index * 2) AS ball
    FROM deliveries WHERE NOT super_over
    GROUP BY match_id, innings, batter
    UNION ALL
    SELECT match_id, innings, non_striker AS name,
           MIN(over * 1000 + ball_index * 2 + 1) AS ball
    FROM deliveries WHERE NOT super_over
    GROUP BY match_id, innings, non_striker
),
first_seen AS (
    SELECT match_id, innings, name, MIN(ball) AS ball
    FROM seen GROUP BY match_id, innings, name
)
SELECT match_id, innings, name AS batter,
       ROW_NUMBER() OVER (PARTITION BY match_id, innings ORDER BY ball, name) AS slot
FROM first_seen
"""

BUILD = (
    # One name per ground. The part before the first comma names the ground, and
    # "Lord's" and "Lord's, London" are one ground. But when that name comes with
    # several different places after the comma, they are different grounds: England
    # has a "County Ground" in Bristol, Taunton, Hove and more (found on Day 36, when
    # the first rule merged nine grounds into one).
    """
    CREATE OR REPLACE TABLE match_venues AS
    WITH parts AS (
        SELECT match_id, city,
               TRIM(SPLIT_PART(venue, ',', 1)) AS ground,
               NULLIF(TRIM(SUBSTRING(venue, LENGTH(SPLIT_PART(venue, ',', 1)) + 2)), '')
                   AS place
        FROM matches
    ),
    places AS (
        SELECT ground, COUNT(DISTINCT place) AS how_many FROM parts GROUP BY ground
    )
    SELECT p.match_id,
           CASE WHEN s.how_many <= 1 THEN p.ground
                ELSE p.ground || ', ' || COALESCE(p.place, p.city, '?')
           END AS venue_key
    FROM parts p JOIN places s ON s.ground = p.ground
    """,
    # every ground once (M2)
    """
    CREATE OR REPLACE TABLE venues AS
    SELECT v.venue_key,
           MODE(m.city) AS city,
           COUNT(*) AS matches,
           MIN(m.date) AS first_match,
           MAX(m.date) AS last_match,
           COUNT(DISTINCT m.venue) AS names_used
    FROM match_venues v JOIN matches m ON m.match_id = v.match_id
    GROUP BY v.venue_key
    """,
    # every innings: its score, wickets and legal balls, and whether the side won (M2)
    f"""
    CREATE OR REPLACE TABLE innings_scores AS
    SELECT m.match_id, m.date, m.match_type, m.gender, m.team_type, m.competition,
           v.venue_key, i.innings, i.team,
           COALESCE(SUM(d.runs_total), 0) + i.penalty_pre + i.penalty_post AS runs,
           COALESCE(SUM(CASE WHEN d.wickets > 0 AND d.wicket_kind NOT IN {NOT_OUT}
                             THEN d.wickets ELSE 0 END), 0) AS wickets,
           COALESCE(SUM(CASE WHEN d.wides = 0 AND d.noballs = 0 THEN 1 ELSE 0 END), 0)
               AS legal_balls,
           m.winner = i.team AS won,
           i.target_runs IS NOT NULL AS chasing
    FROM innings i
    JOIN matches m ON m.match_id = i.match_id
    JOIN match_venues v ON v.match_id = i.match_id
    LEFT JOIN deliveries d ON d.match_id = i.match_id AND d.innings = i.innings
    WHERE NOT i.super_over
    GROUP BY m.match_id, m.date, m.match_type, m.gender, m.team_type, m.competition,
             v.venue_key, i.innings, i.team, i.penalty_pre, i.penalty_post, m.winner,
             i.target_runs
    """,
    # how many balls each bowler bowled on each match day (M3: workload)
    """
    CREATE OR REPLACE TABLE bowler_days AS
    SELECT p.person_id AS bowler_id, d.bowler, d.date, d.match_type, d.gender,
           SUM(CASE WHEN d.wides = 0 AND d.noballs = 0 THEN 1 ELSE 0 END) AS legal_balls,
           COUNT(*) AS deliveries
    FROM deliveries d
    LEFT JOIN people p ON p.match_id = d.match_id AND p.name = d.bowler
    GROUP BY p.person_id, d.bowler, d.date, d.match_type, d.gender
    """,
    # every DRS review: who asked, and was the decision changed (M4)
    """
    CREATE OR REPLACE TABLE reviews AS
    SELECT match_id, date, match_type, gender, competition, innings, over, ball_index,
           review_by, batting_team, review_batter, review_decision, review_umpires_call,
           review_by = batting_team AS by_batting_side,
           review_decision = 'upheld' AS overturned,
           wickets > 0 AS out_on_this_ball,
           wicket_kind
    FROM deliveries
    WHERE review_by IS NOT NULL
    """,
    BALL_STAGES,
    BATTING_SLOTS,
)

TABLES = (
    "match_venues",
    "venues",
    "innings_scores",
    "bowler_days",
    "reviews",
    "ball_stages",
    "batting_slots",
)


def build_derived(db) -> dict[str, int]:
    """Make the derived tables; returns their row counts."""
    for sql in BUILD:
        db.execute(sql)
    counts = {}
    for table in TABLES:
        counts[table] = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts


def main(arguments: list[str]) -> None:
    if len(arguments) != 1:
        print("usage: python -m cricketmind_v3.pipeline.derived <database>")
        sys.exit(2)
    with duckdb.connect(arguments[0]) as db:
        for table, rows in build_derived(db).items():
            print(f"{table}: {rows} rows")


if __name__ == "__main__":
    main(sys.argv[1:])
