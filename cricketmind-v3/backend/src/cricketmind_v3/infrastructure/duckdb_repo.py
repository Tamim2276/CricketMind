"""The DuckDB adapter: SQL picks the balls a subject is about (Day 38)."""

import dataclasses

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.balls import SOMEONE_ELSE, THE_BATTER, Ball
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


def find_person(db, name: str) -> str | None:
    """The register id of the player called this; None if nobody is.

    Raises FilterError when several players share the name (Day 38: five Imran Khans).
    """
    rows = db.execute("SELECT person_id FROM players WHERE name = ?", [name]).fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        raise FilterError(f"{len(rows)} players are called {name}: say which one")
    return rows[0][0]


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
        where, values = self._where(subject, player_is="=")
        if where is None:
            return []  # nobody of that name has played

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

    def others(self, subject: Subject) -> dict[Ball, int]:
        """Every other batter's balls under the same filters, counted by DuckDB (Day 39).

        The names give way to THE_BATTER and SOMEONE_ELSE, so balls of different
        batters that are alike are counted together: a few hundred rows, not millions.
        """
        where, values = self._where(subject, player_is="<>")
        if where is None:
            return {}
        sql = (
            "SELECT match_type, runs_batter, wides, "
            "CASE WHEN player_out IS NULL THEN NULL "
            "WHEN player_out = batter THEN ? ELSE ? END AS who_out, "
            "wicket_kind, COUNT(*) "
            f"FROM deliveries_full WHERE {' AND '.join(where)} "
            "GROUP BY match_type, runs_batter, wides, who_out, wicket_kind"
        )
        counts = {}
        rows = self.db.execute(sql, [THE_BATTER, SOMEONE_ELSE, *values]).fetchall()
        for match_type, runs_batter, wides, player_out, wicket_kind, times in rows:
            ball = Ball(
                match_type=match_type,
                batter=THE_BATTER,
                runs_batter=runs_batter,
                wides=wides,
                player_out=player_out,
                wicket_kind=wicket_kind,
            )
            counts[ball] = times
        return counts

    def _where(self, subject: Subject, player_is: str) -> tuple[list | None, list]:
        """The WHERE conditions and their values; None when the player is nobody.

        player_is is "=" for the player's own balls, "<>" for everyone else's.
        """
        where = ["NOT super_over"]  # a tie-breaker is not part of the match
        values = []
        for field in dataclasses.fields(subject):
            value = getattr(subject, field.name)
            if is_empty(value):
                continue
            if field.name in PEOPLE:
                person_id = find_person(self.db, value)
                if person_id is None:
                    return None, []
                test = player_is if field.name == "player" else "="
                where.append(f"{PEOPLE[field.name]} {test} ?")
                values.append(person_id)
            elif field.name in COLUMNS:
                where.append(COLUMNS[field.name] + " = ?")
                values.append(value)
            else:
                raise FilterError(f"cannot pick balls by {field.name} yet")
        return where, values
