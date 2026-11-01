"""One ground, one name (Day 58, part 2b; L-028), and which ground a question means.

Cricsheet files one ground under several names: "M Chinnaswamy Stadium, Bangalore",
"..., Bengaluru" and "M.Chinnaswamy Stadium"; "Kensington Oval, Bridgetown" with the
island as its city, and "Kensington Oval, Barbados" with the town. Two names are one
ground when the ground's name is the same, letters and digits only, and they share a
place: a word after the comma, or a city its matches give, renamed cities mapped to their
new name. So Lord's (London) and Lords (Bermuda) stay apart, and so do the County Grounds
of Taunton and Bristol. A renamed or misspelt ground no rule can see is in a short list,
each one checked by hand.
"""

from dataclasses import dataclass

# cities known by an older name in some matches
RENAMED_PLACES = {
    "bangalore": "bengaluru",
    "bombay": "mumbai",
    "madras": "chennai",
    "calcutta": "kolkata",
    "poona": "pune",
    "baroda": "vadodara",
    "trivandrum": "thiruvananthapuram",
    "mysore": "mysuru",
    "gurgaon": "gurugram",
    "portelizabeth": "gqeberha",
    "chittagong": "chattogram",
}

# one ground under another name, checked by hand (Day 58): renamed, misspelt, or given
# two places. Each other name -> the name it joins
SAME_GROUND = {
    "Feroz Shah Kotla": "Arun Jaitley Stadium",  # renamed in 2019
    "Punjab Cricket Association Stadium, Mohali": (
        "Punjab Cricket Association IS Bindra Stadium, Mohali"  # renamed
    ),
    "Darren Sammy National Cricket Stadium, St Lucia": (
        "Daren Sammy National Cricket Stadium, Gros Islet, St Lucia"  # spelt both ways
    ),
    "Sher-e-Bangla National Cricket Stadium": "Shere Bangla National Stadium",  # Mirpur
    "Bharat Ratna Shri Atal Bihari Vajpai Ekana Cricket Stadium": (
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium"
    ),
    "Holkar Stadium": "Holkar Cricket Stadium",  # Indore
    "Sardar Patel (Gujarat) Stadium": "Sardar Patel Stadium",
    "Dr DY Patil Sports Academy, Navi Mumbai": "Dr DY Patil Sports Academy, Mumbai",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": (
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"
    ),
}
# not joined, on purpose: the Narendra Modi Stadium was built where the Sardar Patel
# Stadium stood, a new ground; "Ground 'A'", "Palam II" and "... Stadium B" are other
# grounds beside the main one


def squash(text: str) -> str:
    """Letters and digits only, lower case: "M.Chinnaswamy" and "M Chinnaswamy" agree."""
    kept = ""
    for char in text.lower():
        if char.isalnum():
            kept += char
    return kept


def places_of(key: str, cities: tuple[str, ...]) -> set[str]:
    """The places a ground's name and its matches give: Bridgetown, Barbados."""
    found = set()
    parts = key.split(",")
    for part in parts[1:]:
        _add_place(found, part)
    for city in cities:
        _add_place(found, city)
    return found


def _add_place(found: set[str], text: str | None) -> None:
    if text is None:
        return
    place = squash(text)
    if place:
        found.add(RENAMED_PLACES.get(place, place))


def same_grounds(rows: list[tuple[str, tuple[str, ...], int]]) -> dict[str, str]:
    """Each ground name's main name. rows: (name, the cities its matches give, how many
    matches). The main name of a ground is the name with the most matches."""
    main = {}
    for key, _, _ in rows:
        main[key] = key
    by_ground = {}
    for key, cities, _ in rows:
        name = squash(key.split(",")[0])
        if name not in by_ground:
            by_ground[name] = []
        by_ground[name].append((key, places_of(key, cities)))

    # one ground name sharing a place: one ground
    for members in by_ground.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if members[i][1] & members[j][1]:
                    _join(main, members[i][0], members[j][0])

    # the ones checked by hand
    for other, joins in SAME_GROUND.items():
        if other in main and joins in main:
            _join(main, other, joins)

    matches = {}
    for key, _, count in rows:
        matches[key] = count
    groups = {}
    for key in main:
        root = _root(main, key)
        if root not in groups:
            groups[root] = []
        groups[root].append(key)
    named = {}
    for keys in groups.values():
        best = keys[0]
        for key in keys:
            if (matches[key], key) > (matches[best], best):
                best = key
        for key in keys:
            named[key] = best
    return named


def _root(main: dict[str, str], key: str) -> str:
    while main[key] != key:
        key = main[key]
    return key


def _join(main: dict[str, str], one: str, other: str) -> None:
    first = _root(main, one)
    second = _root(main, other)
    if first != second:
        main[second] = first


# which ground a question means


@dataclass(frozen=True, kw_only=True)
class Ground:
    """One ground: its main name, and every way the data names it."""

    name: str  # "M Chinnaswamy Stadium, Bangalore": the engines' key
    other_names: tuple[str, ...] = ()  # "M.Chinnaswamy Stadium", as the matches had it
    places: tuple[str, ...] = ()  # "Bengaluru": the towns and cities its matches give
    matches: int = 0
    first_year: int | None = None
    last_year: int | None = None


def find_grounds(written: str, grounds: list[Ground]) -> list[Ground]:
    """Which grounds the words could be, most-played first: by the ground's name
    ("Chinnaswamy", "Lord's"), or else by its place ("Mirpur", "Mumbai")."""
    wanted = _words(written)
    if not wanted:
        return []
    by_name = []
    by_place = []
    for ground in grounds:
        # before the comma, its name; after it, places: "Dr DY Patil Sports Academy,
        # Mumbai" is not named Mumbai, so "Mumbai" finds Wankhede too
        names = []
        places = list(ground.places)
        for full in (ground.name, *ground.other_names):
            parts = full.split(",")
            names.append(parts[0])
            places.extend(parts[1:])
        if _any_has(names, wanted):
            by_name.append(ground)
        elif _any_has(places, wanted):
            by_place.append(ground)
    for found in (by_name, by_place):
        if found:
            return sorted(found, key=_most_played_first)
    return []


def describe_ground(ground: Ground) -> str:
    """Which ground it is, for a question back: never a statistic."""
    parts = [ground.name]
    if ground.places:
        parts.append(", ".join(ground.places[:2]))
    if ground.first_year is not None:
        parts.append(f"{ground.first_year}-{ground.last_year}")
    return "; ".join(parts)


def _words(text: str) -> list[str]:
    # "Lord's" -> lords; "M.Chinnaswamy" -> mchinnaswamy
    found = []
    for word in text.replace("'", "").replace("’", "").replace(".", "").split():
        squashed = squash(word)
        if squashed:
            found.append(squashed)
    return found


def _any_has(names, wanted: list[str]) -> bool:
    # the words, in a row, somewhere in one of the names
    for name in names:
        said = _words(name)
        for start in range(len(said) - len(wanted) + 1):
            if said[start : start + len(wanted)] == wanted:
                return True
    return False


def _most_played_first(ground: Ground) -> tuple[int, str]:
    return (-ground.matches, ground.name)
