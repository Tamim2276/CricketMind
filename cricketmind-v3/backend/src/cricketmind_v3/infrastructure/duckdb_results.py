"""The DuckDB adapter for M5: how matches ended (Day 45)."""

import dataclasses

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.results import Result
from cricketmind_v3.domain.subject import Subject
from cricketmind_v3.infrastructure.filters import MATCH_FILTERS, match_condition


class DuckDBResultRepository:
    """Results from the matches table (Day 30)."""

    def __init__(self, db) -> None:
        self.db = db

    def results(self, subject: Subject) -> list[Result]:
        where = ["true"]
        values = []
        for field in dataclasses.fields(subject):
            value = getattr(subject, field.name)
            if is_empty(value):
                continue
            if field.name in ("team", "opposition"):
                where.append("(m.team1 = ? OR m.team2 = ?)")  # it played in the match
                values.extend([value, value])
            elif field.name == "date":
                where.append("m.date = ?")
                values.append(value)
            elif field.name in MATCH_FILTERS:
                condition, more = match_condition(field.name, value)
                where.append(condition)
                values.extend(more)
            else:
                raise FilterError(f"cannot pick matches by {field.name}")
        rows = self.db.execute(
            "SELECT m.team1, m.team2, m.winner, m.won_by_runs, m.won_by_wickets, "
            "m.won_by_innings FROM matches m "
            "JOIN match_venues v ON v.match_id = m.match_id "
            f"WHERE {' AND '.join(where)} ORDER BY m.date, m.match_id",
            values,
        ).fetchall()
        found = []
        for team1, team2, winner, by_runs, by_wickets, by_innings in rows:
            found.append(
                Result(
                    teams=(team1, team2),
                    winner=winner,
                    by_runs=by_runs,
                    by_wickets=by_wickets,
                    by_innings=by_innings,
                )
            )
        return found
