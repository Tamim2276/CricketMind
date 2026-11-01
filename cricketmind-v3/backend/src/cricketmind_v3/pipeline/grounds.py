"""One ground, one name, in the data (Day 58, part 2b; L-028), and the ground
register N3 looks grounds up in."""

from cricketmind_v3.domain.grounds import Ground, same_grounds

# every ground name with the cities its matches give, and how many matches
NAMES_AND_CITIES = """
SELECT v.venue_key,
       LIST(DISTINCT m.city ORDER BY m.city) FILTER (WHERE m.city IS NOT NULL),
       COUNT(*)
FROM match_venues v JOIN matches m ON m.match_id = v.match_id
GROUP BY v.venue_key
ORDER BY v.venue_key
"""


def merge_grounds(db) -> int:
    """Give every match of a ground its main name; returns how many names were joined."""
    rows = []
    for key, cities, matches in db.execute(NAMES_AND_CITIES).fetchall():
        rows.append((key, tuple(cities or ()), matches))
    joined = 0
    for key, main in same_grounds(rows).items():
        if key != main:
            db.execute(
                "UPDATE match_venues SET venue_key = ? WHERE venue_key = ?", [main, key]
            )
            joined += 1
    return joined


def load_grounds(db) -> list[Ground]:
    """Every ground: its main name, the names its matches used, its places, its years."""
    names = {}
    places = {}
    for key, venue, city in db.execute(
        "SELECT DISTINCT v.venue_key, m.venue, m.city FROM match_venues v "
        "JOIN matches m ON m.match_id = v.match_id ORDER BY v.venue_key, m.venue, m.city"
    ).fetchall():
        if venue != key and venue not in names.setdefault(key, []):
            names[key].append(venue)
        if city is not None and city not in places.setdefault(key, []):
            places[key].append(city)
    grounds = []
    for key, matches, first_match, last_match in db.execute(
        "SELECT venue_key, matches, first_match, last_match FROM venues ORDER BY venue_key"
    ).fetchall():
        grounds.append(
            Ground(
                name=key,
                other_names=tuple(names.get(key, [])),
                places=tuple(places.get(key, [])),
                matches=matches,
                first_year=first_match.year,
                last_year=last_match.year,
            )
        )
    return grounds
