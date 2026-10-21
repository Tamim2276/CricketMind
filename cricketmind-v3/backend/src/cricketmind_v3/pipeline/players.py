"""The players table, built from the register ids in every match (Day 33)."""

from cricketmind_v3.domain.names import Player

# one row per person who played: the name used most, and how many matches
BUILD_PLAYERS = """
CREATE OR REPLACE TABLE players AS
WITH named AS (
    SELECT person_id, name, COUNT(DISTINCT match_id) AS matches
    FROM people WHERE team IS NOT NULL
    GROUP BY person_id, name
),
ranked AS (
    SELECT person_id, name,
           ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY matches DESC, name) AS place
    FROM named
),
played AS (
    SELECT person_id, COUNT(DISTINCT match_id) AS matches
    FROM people WHERE team IS NOT NULL
    GROUP BY person_id
)
SELECT r.person_id, r.name, p.matches
FROM ranked r JOIN played p ON p.person_id = r.person_id
WHERE r.place = 1
"""

# every name a player was listed under
BUILD_PLAYER_NAMES = """
CREATE OR REPLACE TABLE player_names AS
SELECT DISTINCT person_id, name FROM people WHERE team IS NOT NULL
"""


def build_players(db) -> int:
    """Make the players and player_names tables; returns how many players."""
    db.execute(BUILD_PLAYERS)
    db.execute(BUILD_PLAYER_NAMES)
    return db.execute("SELECT COUNT(*) FROM players").fetchone()[0]


def load_players(db) -> list[Player]:
    """Every player, with their other names as aliases."""
    others = {}
    rows = db.execute(
        "SELECT n.person_id, n.name FROM player_names n "
        "JOIN players p ON p.person_id = n.person_id WHERE n.name <> p.name "
        "ORDER BY n.person_id, n.name"
    ).fetchall()
    for person_id, name in rows:
        _add_alias(others, person_id, name)

    # once the attributes are in (Day 34), full names become names to search by too
    if _has_table(db, "attributes"):
        rows = db.execute(
            "SELECT a.person_id, a.known_as, a.full_name, p.name FROM attributes a "
            "JOIN players p ON p.person_id = a.person_id ORDER BY a.person_id"
        ).fetchall()
        for person_id, known_as, full_name, name in rows:
            for alias in (known_as, full_name):
                if alias and alias != name:
                    _add_alias(others, person_id, alias)

    players = []
    rows = db.execute(
        "SELECT person_id, name, matches FROM players ORDER BY person_id"
    ).fetchall()
    for person_id, name, matches in rows:
        players.append(
            Player(
                person_id=person_id,
                name=name,
                aliases=tuple(others.get(person_id, [])),
                matches=matches,
            )
        )
    return players


def _add_alias(others: dict[str, list[str]], person_id: str, alias: str) -> None:
    if person_id not in others:
        others[person_id] = []
    if alias not in others[person_id]:
        others[person_id].append(alias)


def _has_table(db, name: str) -> bool:
    found = db.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()[0]
    return found > 0
