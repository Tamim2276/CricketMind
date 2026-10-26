"""The DuckDB adapter: SQL picks the balls a subject is about (Days 38, 43, 50, 51, 53)."""

import dataclasses

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.balls import NOT_DISMISSALS, SOMEONE_ELSE, THE_BATTER, Ball
from cricketmind_v3.domain.calibration import Moment
from cricketmind_v3.domain.matchups import Pair
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.stages import ORDERS, STARTS, Record
from cricketmind_v3.domain.subject import Subject
from cricketmind_v3.infrastructure.filters import MATCH_FILTERS, match_condition

# subject field -> the column of the ball that picks it
COLUMNS = {
    "phase": "d.phase",
    "bowler_type": "d.bowler_type",
    "batting_hand": "d.batting_hand",
    "innings": "d.innings",
}

# subject fields that name a person: matched on the register id, never on the name,
# because one person can play under two names
PEOPLE = {
    "player": "d.batter_id",
    "bowler": "d.bowler_id",
}

# he was the one out, named as he was in that match, and someone else was on strike
OUT_AT_OTHER_END = (
    "(EXISTS (SELECT 1 FROM people o WHERE o.match_id = d.match_id "
    "AND o.name = d.player_out AND o.person_id = ?) AND d.batter_id <> ?)"
)

# every ball, with its match and its ground, so a match filter can pick it
BALLS = (
    "deliveries_full d "
    "JOIN matches m ON m.match_id = d.match_id "
    "JOIN match_venues v ON v.match_id = d.match_id"
)


# each ball with where its batter batted (Day 50)
ORDERED = BALLS + (
    " JOIN batting_slots b ON b.match_id = d.match_id AND b.innings = d.innings"
    " AND b.batter = d.batter"
)

# and with how long he had been in
STAGED = ORDERED + (
    " JOIN ball_stages s ON s.match_id = d.match_id AND s.innings = d.innings"
    " AND s.over = d.over AND s.ball_index = d.ball_index"
)

# the counting rules of balls.py, for DuckDB to count by: out when the one out was on
# strike and did not retire (one ? per kind of retiring); faced unless a wide
IS_OUT = "(d.player_out = d.batter AND COALESCE(d.wicket_kind, '') NOT IN ({marks}))"
OUT = "SUM(CASE WHEN " + IS_OUT + " THEN 1 ELSE 0 END)"
FACED = "SUM(CASE WHEN d.wides = 0 THEN 1 ELSE 0 END)"


def stage_case(faced_before: str) -> str:
    """The SQL for a ball's stage (stages.STARTS), from the balls faced before it: the
    ball is his ball number faced_before + 1."""
    parts = []
    for number in range(1, len(STARTS)):
        parts.append(f"WHEN {faced_before} + 1 < {STARTS[number]} THEN {number}")
    return "CASE " + " ".join(parts) + f" ELSE {len(STARTS)} END"


def order_case(slot: str) -> str:
    """The SQL for the part of the order a place is in (stages.ORDERS)."""
    parts = []
    for number in range(1, len(ORDERS)):
        name, _ = ORDERS[number - 1]
        _, next_first = ORDERS[number]
        parts.append(f"WHEN {slot} < {next_first} THEN '{name}'")
    last, _ = ORDERS[-1]
    return "CASE " + " ".join(parts) + f" ELSE '{last}' END"


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
        columns = []
        for name in ball_columns():
            columns.append("d." + name)
        sql = (
            f"SELECT {', '.join(columns)} FROM {BALLS} "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY d.date, d.match_id, d.innings, d.over, d.ball_index"  # match order
        )
        balls = []
        for row in self.db.execute(sql, values).fetchall():
            fields = {}
            for name, value in zip(ball_columns(), row, strict=True):
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
            "SELECT d.match_type, d.gender, d.runs_batter, d.wides, d.noballs, "
            "CASE WHEN d.player_out IS NULL THEN NULL "
            "WHEN d.player_out = d.batter THEN ? ELSE ? END AS who_out, "
            "d.wicket_kind, COUNT(*) "
            f"FROM {BALLS} WHERE {' AND '.join(where)} "
            "GROUP BY d.match_type, d.gender, d.runs_batter, d.wides, d.noballs, "
            "who_out, d.wicket_kind "
            "ORDER BY 1, 2, 3, 4, 5, 6, 7"  # the same order every time (Day 54)
        )
        counts = {}
        rows = self.db.execute(sql, [THE_BATTER, SOMEONE_ELSE, *values]).fetchall()
        for row in rows:
            match_type, gender, runs, wides, noballs, player_out, kind, times = row
            ball = Ball(
                match_type=match_type,
                gender=gender,
                batter=THE_BATTER,
                runs_batter=runs,
                wides=wides,
                noballs=noballs,
                player_out=player_out,
                wicket_kind=kind,
            )
            counts[ball] = times
        return counts

    def outs_at_other_end(self, subject: Subject) -> int:
        """Balls on which the player, not on strike, was out (Day 44): matched on his
        register id through the names of that match. Retiring is not out."""
        where, values = self._where(subject, player_is="at the other end")
        if where is None:
            return 0
        marks = ", ".join(["?"] * len(NOT_DISMISSALS))
        sql = (
            f"SELECT COUNT(*) FROM {BALLS} "
            f"WHERE {' AND '.join(where)} AND d.wicket_kind NOT IN ({marks})"
        )
        return self.db.execute(sql, [*values, *NOT_DISMISSALS]).fetchone()[0]

    def records(self, subject: Subject) -> list[Record]:
        """The player's times out and balls faced, by stage of his innings and the part
        of the order he batted in (Day 50)."""
        return self._records(subject, player_is="=")

    def others_records(self, subject: Subject) -> list[Record]:
        """Every other batter's, one record per batter, stage and part of the order."""
        return self._records(subject, player_is="<>")

    def pairs(self, subject: Subject) -> list[Pair]:
        """Every batter against every bowler under the subject's filters (Day 51)."""
        where, values = self._where(subject, player_is="=")
        if where is None:
            return []
        marks = ", ".join(["?"] * len(NOT_DISMISSALS))
        sql = (
            "SELECT COALESCE(d.batter_id, d.batter) AS who, "
            "COALESCE(d.bowler_id, d.bowler) AS whom, "
            f"{order_case('b.slot')} AS batting_order, "
            f"{OUT.format(marks=marks)}, {FACED} "
            f"FROM {ORDERED} WHERE {' AND '.join(where)} "
            "GROUP BY who, whom, batting_order "
            # the same order every time: summed in another order, the fitted prior moved
            # in its eighth digit and two receipts failed (found on Day 54)
            "ORDER BY who, whom, batting_order"
        )
        found = []
        rows = self.db.execute(sql, [*NOT_DISMISSALS, *values]).fetchall()
        for who, whom, order, outs, faced in rows:
            found.append(
                Pair(batter=who, bowler=whom, order=order, outs=outs, faced=faced)
            )
        return found

    def person(self, name: str) -> str | None:
        """The register id of the player called this (Day 51)."""
        return find_person(self.db, name)

    def moments(self, subject: Subject) -> list[Moment]:
        """Every ball faced, and every ball a batter was out on, under the subject's
        filters, in match order: what the calibration scores (Day 53)."""
        where, values = self._where(subject, player_is="=")
        if where is None:
            return []
        marks = ", ".join(["?"] * len(NOT_DISMISSALS))
        is_out = IS_OUT.format(marks=marks)
        sql = (
            "SELECT d.match_id, d.innings, COALESCE(d.batter_id, d.batter), "
            "COALESCE(d.bowler_id, d.bowler), s.faced_before, "
            # no one out is NULL in SQL, not false
            f"{order_case('b.slot')}, COALESCE({is_out}, false), d.wides = 0 "
            f"FROM {STAGED} WHERE {' AND '.join(where)} "
            f"AND (d.wides = 0 OR {is_out}) "
            "ORDER BY d.date, d.match_id, d.innings, d.over, d.ball_index"
        )
        found = []
        rows = self.db.execute(sql, [*NOT_DISMISSALS, *values, *NOT_DISMISSALS])
        for (
            match,
            innings,
            batter,
            bowler,
            before,
            order,
            out,
            faced,
        ) in rows.fetchall():
            found.append(
                Moment(
                    match=match,
                    innings=innings,
                    batter=batter,
                    bowler=bowler,
                    faced_before=before,
                    order=order,
                    out=out,
                    faced=faced,
                )
            )
        return found

    def _records(self, subject: Subject, player_is: str) -> list[Record]:
        # millions of balls, so DuckDB counts, by the rules of balls.py; a test holds
        # the two together. A batter with no register id goes by his name
        where, values = self._where(subject, player_is=player_is)
        if where is None:
            return []
        marks = ", ".join(["?"] * len(NOT_DISMISSALS))
        sql = (
            "SELECT COALESCE(d.batter_id, d.batter) AS who, "
            f"{stage_case('s.faced_before')} AS stage, "
            f"{order_case('b.slot')} AS batting_order, "
            f"{OUT.format(marks=marks)}, {FACED} "
            f"FROM {STAGED} WHERE {' AND '.join(where)} "
            "GROUP BY who, stage, batting_order ORDER BY who, stage, batting_order"
        )
        found = []
        rows = self.db.execute(sql, [*NOT_DISMISSALS, *values]).fetchall()
        for who, stage, order, outs, faced in rows:
            found.append(
                Record(batter=who, stage=stage, order=order, outs=outs, faced=faced)
            )
        return found

    def _where(self, subject: Subject, player_is: str) -> tuple[list | None, list]:
        """The WHERE conditions and their values; None when a person named is nobody.

        player_is is "=" for the player's own balls, "<>" for everyone else's, and
        "at the other end" for balls he was out on while the other batter faced.
        """
        where = ["NOT d.super_over"]  # a tie-breaker is not part of the match
        values = []
        # whose side "opposition" is seen from: the batter's, or else the bowler's
        batting = not is_empty(getattr(subject, "player", None))
        for field in dataclasses.fields(subject):
            name = field.name
            value = getattr(subject, name)
            if is_empty(value):
                continue
            if name in PEOPLE:
                person_id = find_person(self.db, value)
                if person_id is None:
                    return None, []
                if name == "player" and player_is == "at the other end":
                    where.append(OUT_AT_OTHER_END)
                    values.extend([person_id, person_id])
                    continue
                test = player_is if name == "player" else "="
                where.append(f"{PEOPLE[name]} {test} ?")
                values.append(person_id)
            elif name in COLUMNS:
                where.append(COLUMNS[name] + " = ?")
                values.append(value)
            elif name == "opposition" and batting:
                # the batter's opponents bowled: the other side in his match
                where.append("(d.batting_team <> ? AND (m.team1 = ? OR m.team2 = ?))")
                values.extend([value, value, value])
            elif name == "opposition":
                where.append("d.batting_team = ?")  # the bowler's opponents batted
                values.append(value)
            elif name in MATCH_FILTERS:
                condition, more = match_condition(name, value)
                where.append(condition)
                values.extend(more)
            else:
                raise FilterError(f"cannot pick balls by {name} yet")
        return where, values
