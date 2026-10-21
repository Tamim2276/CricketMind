"""The DuckDB adapter: SQL picks the balls a subject is about (Day 38)."""

import dataclasses

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.subject import Subject

# subject field -> the column that picks balls by it (Day 43 adds the rest)
COLUMNS = {
    "phase": "phase",
    "bowler_type": "bowler_type",
}

# subject fields that name a person: matched on the register id, never on the name,
# because one person can play under two names
PEOPLE = {
    "player": "batter_id",
    "bowler": "bowler_id",
}


def ball_columns() -> list[str]:
    """The view's columns that make a Ball: named as its fields, so the two stay in step."""
    names = []
    for field in dataclasses.fields(Ball):
        names.append(field.name)
    return names


class DuckDBDeliveryRepository:
    """Balls from the deliveries_full view (Day 35), on an open DuckDB connection."""

    def __init__(self, db) -> None:
        self.db = db

    def deliveries(self, subject: Subject) -> list[Ball]:
        where = ["NOT super_over"]  # a tie-breaker is not part of the match
        values = []
        for field in dataclasses.fields(subject):
            value = getattr(subject, field.name)
            if is_empty(value):
                continue
            if field.name in PEOPLE:
                person_id = self.person_id(value)
                if person_id is None:
                    return []  # nobody of that name has played
                where.append(PEOPLE[field.name] + " = ?")
                values.append(person_id)
            elif field.name in COLUMNS:
                where.append(COLUMNS[field.name] + " = ?")
                values.append(value)
            else:
                raise FilterError(f"cannot pick balls by {field.name} yet")

        # column names come from the lists above; every value goes in as a ?
        columns = ball_columns()
        sql = (
            f"SELECT {', '.join(columns)} FROM deliveries_full "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY date, match_id, innings, over, ball_index"  # match order, always
        )
        balls = []
        for row in self.db.execute(sql, values).fetchall():
            fields = {}
            for name, value in zip(columns, row, strict=True):
                fields[name] = value
            balls.append(Ball(**fields))
        return balls

    def person_id(self, name: str) -> str | None:
        """The register id of the player called this; None if nobody is."""
        rows = self.db.execute(
            "SELECT person_id FROM players WHERE name = ?", [name]
        ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise FilterError(f"{len(rows)} players are called {name}: say which one")
        return rows[0][0]
