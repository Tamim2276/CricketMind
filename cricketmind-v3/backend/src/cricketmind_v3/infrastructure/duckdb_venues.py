"""The DuckDB adapter for M2: a ground's innings and wickets (Days 40, 43)."""

import dataclasses

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.subject import Subject
from cricketmind_v3.domain.venues import InningsScore, Wicket
from cricketmind_v3.infrastructure.filters import MATCH_FILTERS, match_condition


class DuckDBVenueRepository:
    """Innings from innings_scores and wickets from deliveries_full (Days 35-36)."""

    def __init__(self, db) -> None:
        self.db = db

    def innings_at(self, subject: Subject) -> list[InningsScore]:
        where, values = matches_picked(subject)
        rows = self.db.execute(
            "SELECT s.match_type, m.overs * m.balls_per_over, s.innings, s.runs, "
            "s.wickets, s.legal_balls, i.declared, s.won "
            "FROM innings_scores s "
            "JOIN matches m ON m.match_id = s.match_id "
            "JOIN match_venues v ON v.match_id = s.match_id "
            "JOIN innings i ON i.match_id = s.match_id AND i.innings = s.innings "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY s.date, s.match_id, s.innings",
            values,
        ).fetchall()
        scores = []
        for row in rows:
            kind, scheduled, innings, runs, wickets, legal, declared, won = row
            scores.append(
                InningsScore(
                    match_type=kind,
                    scheduled_balls=scheduled,
                    innings=innings,
                    runs=runs,
                    wickets=wickets,
                    legal_balls=legal,
                    declared=declared,
                    won=won,
                )
            )
        return scores

    def wickets_at(self, subject: Subject) -> dict[Wicket, int]:
        where, values = matches_picked(subject)
        rows = self.db.execute(
            "SELECT d.match_type, m.overs * m.balls_per_over AS scheduled, "
            "d.bowler_type, d.wicket_kind, COUNT(*) "
            "FROM deliveries_full d "
            "JOIN matches m ON m.match_id = d.match_id "
            "JOIN match_venues v ON v.match_id = d.match_id "
            f"WHERE {' AND '.join(where)} "
            "AND d.wicket_kind IS NOT NULL AND NOT d.super_over "
            "GROUP BY d.match_type, scheduled, d.bowler_type, d.wicket_kind",
            values,
        ).fetchall()
        counts = {}
        for kind, scheduled, bowler_type, wicket_kind, times in rows:
            wicket = Wicket(
                match_type=kind,
                scheduled_balls=scheduled,
                bowler_type=bowler_type,
                kind=wicket_kind,
            )
            counts[wicket] = times
        return counts


def matches_picked(subject: Subject) -> tuple[list[str], list]:
    """The WHERE conditions for a ground's matches; a field M2 cannot use is refused."""
    where = ["true"]
    values = []
    for field in dataclasses.fields(subject):
        value = getattr(subject, field.name)
        if is_empty(value):
            continue
        if field.name not in MATCH_FILTERS:
            raise FilterError(f"cannot pick a ground's matches by {field.name}")
        condition, more = match_condition(field.name, value)
        where.append(condition)
        values.extend(more)
    return where, values
