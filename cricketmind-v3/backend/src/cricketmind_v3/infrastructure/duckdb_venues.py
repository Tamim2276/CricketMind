"""The DuckDB adapter for M2: a ground's innings and wickets (Day 40)."""

from cricketmind_v3.domain.venues import InningsScore, Wicket


class DuckDBVenueRepository:
    """Innings from innings_scores and wickets from deliveries_full (Days 35-36)."""

    def __init__(self, db) -> None:
        self.db = db

    def innings_at(
        self, venue: str, match_type: str, gender: str
    ) -> list[InningsScore]:
        rows = self.db.execute(
            """
            SELECT s.match_type, m.overs * m.balls_per_over, s.innings, s.runs,
                   s.wickets, s.legal_balls, i.declared, s.won
            FROM innings_scores s
            JOIN matches m ON m.match_id = s.match_id
            JOIN innings i ON i.match_id = s.match_id AND i.innings = s.innings
            WHERE s.venue_key = ? AND s.match_type = ? AND s.gender = ?
            ORDER BY s.date, s.match_id, s.innings
            """,
            [venue, match_type, gender],
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

    def wickets_at(self, venue: str, match_type: str, gender: str) -> dict[Wicket, int]:
        rows = self.db.execute(
            """
            SELECT d.match_type, m.overs * m.balls_per_over AS scheduled,
                   d.bowler_type, d.wicket_kind, COUNT(*)
            FROM deliveries_full d
            JOIN match_venues v ON v.match_id = d.match_id
            JOIN matches m ON m.match_id = d.match_id
            WHERE v.venue_key = ? AND d.match_type = ? AND d.gender = ?
              AND d.wicket_kind IS NOT NULL AND NOT d.super_over
            GROUP BY d.match_type, scheduled, d.bowler_type, d.wicket_kind
            """,
            [venue, match_type, gender],
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
