"""The DuckDB adapter for M4: a team's reviews, counted by how they ended (Day 42)."""


class DuckDBReviewRepository:
    """Counts from the reviews table (Day 36)."""

    def __init__(self, db) -> None:
        self.db = db

    def decisions(self, team: str, gender: str) -> dict[str, int]:
        rows = self.db.execute(
            "SELECT review_decision, COUNT(*) FROM reviews "
            "WHERE review_by = ? AND gender = ? GROUP BY review_decision",
            [team, gender],
        ).fetchall()
        counts = {}
        for decision, times in rows:
            counts[decision] = times
        return counts
