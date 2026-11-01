"""The team register N3 looks teams up in (Day 58, part 2c)."""

from cricketmind_v3.domain.teams import Team

# every team that took the field, with its matches and years
TEAMS = """
SELECT p.team,
       COUNT(DISTINCT p.match_id),
       MIN(YEAR(CAST(m.date AS DATE))),
       MAX(YEAR(CAST(m.date AS DATE)))
FROM people p JOIN matches m ON m.match_id = p.match_id
WHERE p.team IS NOT NULL
GROUP BY p.team
ORDER BY p.team
"""


def load_teams(db) -> list[Team]:
    """Every team, as the data names it."""
    teams = []
    for name, matches, first_year, last_year in db.execute(TEAMS).fetchall():
        teams.append(
            Team(name=name, matches=matches, first_year=first_year, last_year=last_year)
        )
    return teams
