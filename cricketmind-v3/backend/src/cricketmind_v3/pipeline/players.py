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

    profiles = _profiles(db)
    players = []
    rows = db.execute(
        "SELECT person_id, name, matches FROM players ORDER BY person_id"
    ).fetchall()
    for person_id, name, matches in rows:
        profile = profiles.get(person_id, {})
        players.append(
            Player(
                person_id=person_id,
                name=name,
                aliases=tuple(others.get(person_id, [])),
                matches=matches,
                teams=profile.get("teams", ()),
                formats=profile.get("formats", ()),
                first_year=profile.get("first_year"),
                last_year=profile.get("last_year"),
                gender=profile.get("gender"),
            )
        )
    return players


# what tells two players of one name apart (Day 58): never a statistic
PROFILES = """
SELECT p.person_id, p.team, m.match_type, m.team_type, m.gender,
       MIN(YEAR(CAST(m.date AS DATE))) AS first_year,
       MAX(YEAR(CAST(m.date AS DATE))) AS last_year,
       COUNT(DISTINCT p.match_id) AS matches
FROM people p JOIN matches m ON m.match_id = p.match_id
WHERE p.team IS NOT NULL
GROUP BY p.person_id, p.team, m.match_type, m.team_type, m.gender
ORDER BY p.person_id, matches DESC, p.team, m.match_type
"""


def _profiles(db) -> dict[str, dict]:
    """Each player's teams (most played first), formats, years and gender."""
    teams = {}
    formats = {}
    years = {}
    genders = {}
    for (
        person_id,
        team,
        match_type,
        team_type,
        gender,
        first,
        last,
        matches,
    ) in db.execute(PROFILES).fetchall():
        _count(teams.setdefault(person_id, {}), team, matches)
        _count(genders.setdefault(person_id, {}), gender, matches)
        kinds = formats.setdefault(person_id, [])
        names = [match_type]
        if match_type == "T20" and team_type == "international":
            names.append("T20I")  # D-047: a T20 between international sides
        for name in names:
            if name not in kinds:
                kinds.append(name)
        low, high = years.get(person_id, (first, last))
        years[person_id] = (min(low, first), max(high, last))
    profiles = {}
    for person_id, counted in teams.items():
        profiles[person_id] = {
            "teams": _most_first(counted),
            "formats": tuple(sorted(formats[person_id])),
            "first_year": years[person_id][0],
            "last_year": years[person_id][1],
            "gender": _most_first(genders[person_id])[0],
        }
    return profiles


def _count(counts: dict[str, int], key: str, matches: int) -> None:
    counts[key] = counts.get(key, 0) + matches


def _most_first(counts: dict[str, int]) -> tuple[str, ...]:
    ranked = []
    for key, matches in counts.items():
        ranked.append((-matches, key))
    ranked.sort()
    names = []
    for _, key in ranked:
        names.append(key)
    return tuple(names)


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
