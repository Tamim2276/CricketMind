"""The DuckDB adapter for M4: a team's reviews, counted by how they ended (Days 42-43)."""

import dataclasses

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.subject import Subject
from cricketmind_v3.infrastructure.filters import MATCH_FILTERS, match_condition

# who asked for the review: the side batting (to save a wicket) or fielding (to take one)
SIDES = {"batting": True, "fielding": False}


class DuckDBReviewRepository:
    """Counts from the reviews table (Day 36)."""

    def __init__(self, db) -> None:
        self.db = db

    def decisions(self, subject: Subject) -> dict[str, int]:
        where = ["true"]
        values = []
        for field in dataclasses.fields(subject):
            value = getattr(subject, field.name)
            if is_empty(value):
                continue
            if field.name == "team":
                where.append("r.review_by = ?")
                values.append(value)
            elif field.name == "side":
                if value not in SIDES:
                    raise FilterError(
                        f"a review's side is batting or fielding, not {value}"
                    )
                where.append("r.by_batting_side = ?")
                values.append(SIDES[value])
            elif field.name in MATCH_FILTERS:
                condition, more = match_condition(field.name, value)
                where.append(condition)
                values.extend(more)
            else:
                raise FilterError(f"cannot pick reviews by {field.name}")
        rows = self.db.execute(
            "SELECT r.review_decision, COUNT(*) FROM reviews r "
            "JOIN matches m ON m.match_id = r.match_id "
            "JOIN match_venues v ON v.match_id = r.match_id "
            f"WHERE {' AND '.join(where)} GROUP BY r.review_decision",
            values,
        ).fetchall()
        counts = {}
        for decision, times in rows:
            counts[decision] = times
        return counts
