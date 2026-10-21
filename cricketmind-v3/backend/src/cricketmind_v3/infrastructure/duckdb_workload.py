"""The DuckDB adapter for M3: where a bowler's balls came in each match (Day 41)."""

from datetime import date

from cricketmind_v3.domain.workload import Spell
from cricketmind_v3.infrastructure.duckdb_repo import find_person

# every ball of his matches numbered in the order bowled, then his own picked out
SPELLS = """
WITH his_matches AS (
    SELECT DISTINCT match_id FROM people WHERE person_id = ?
),
placed AS (
    SELECT d.match_id, d.bowler,
           ROW_NUMBER() OVER (PARTITION BY d.match_id
                              ORDER BY d.innings, d.over, d.ball_index) - 1 AS position,
           COUNT(*) OVER (PARTITION BY d.match_id) AS match_balls
    FROM deliveries d
    WHERE d.match_id IN (SELECT match_id FROM his_matches)
)
SELECT m.date, m.days, p.match_balls, LIST(p.position ORDER BY p.position)
FROM placed p
JOIN people pe ON pe.match_id = p.match_id AND pe.name = p.bowler
JOIN matches m ON m.match_id = p.match_id
WHERE pe.person_id = ?
GROUP BY p.match_id, m.date, m.days, p.match_balls
ORDER BY m.date, p.match_id
"""


class DuckDBWorkloadRepository:
    """Spells from the deliveries table, matched on the register id."""

    def __init__(self, db) -> None:
        self.db = db

    def spells(self, bowler: str) -> list[Spell]:
        person_id = find_person(self.db, bowler)
        if person_id is None:
            return []
        spells = []
        for first_day, days, match_balls, positions in self.db.execute(
            SPELLS, [person_id, person_id]
        ).fetchall():
            spells.append(
                Spell(
                    first_day=first_day,
                    days=days,
                    match_balls=match_balls,
                    positions=tuple(positions),
                )
            )
        return spells

    def data_until(self) -> date:
        return self.db.execute("SELECT MAX(date) FROM matches").fetchone()[0]
