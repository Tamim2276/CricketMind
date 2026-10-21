"""Who a name in a question means (Day 33): "Kohli" -> "V Kohli"."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Player:
    """One person, as Cricsheet's register knows them."""

    person_id: str  # "ba607b88"
    name: str  # "V Kohli": the name used most in the data
    aliases: tuple[str, ...] = ()  # other names: "Virat Kohli" (Day 34)
    matches: int = 0  # how many matches they played: the tie-breaker


def simplify(text: str) -> str:
    """Lower case, no dots or apostrophes, hyphens as spaces: "M.S. Dhoni" -> "ms dhoni".

    Cricsheet tells two people of the same name apart with "(1)": that goes too.
    """
    text = text.lower().replace(".", "").replace("'", "").replace("’", "")
    text = text.replace("-", " ")
    kept = ""
    inside = False
    for char in text:
        if char == "(":
            inside = True
        elif char == ")":
            inside = False
        elif not inside:
            kept += char
    return " ".join(kept.split())


def find_players(query: str, players: list[Player]) -> list[Player]:
    """Who the name could be, most-played first, using the closest kind of match found.

    1. the whole name: "V Kohli", or an alias like "Virat Kohli"
    2. a first name's initial and the surname: "Rohit Sharma" -> "RG Sharma"
    3. one part of the name, surname or first name, most-played first: "Kohli" ->
       "V Kohli"; "Mendis" -> both Mendises; "Babar" -> "Babar Azam" before
       "Zulfiqar Babar" (many players are known by their first name: found on Day 33)
    """
    wanted = simplify(query)
    if not wanted:
        return []
    whole = []
    initial = []
    part = []
    for player in players:
        names = [simplify(player.name)]
        for alias in player.aliases:
            names.append(simplify(alias))
        if wanted in names:
            whole.append(player)
        elif _initial_and_surname(wanted, names[0]):
            initial.append(player)
        elif _ends_with(names, wanted) or _starts_with_first_name(names, wanted):
            part.append(player)
    for found in (whole, initial, part):
        if found:
            return sorted(found, key=_most_played_first)
    return []


def _initial_and_surname(wanted: str, name: str) -> bool:
    # "rohit sharma" against "rg sharma": the same surname, initials starting with "r"
    asked = wanted.split()
    known = name.split()
    if len(asked) < 2 or len(known) < 2:
        return False
    return asked[-1] == known[-1] and known[0][0] == asked[0][0]


def _ends_with(names: list[str], wanted: str) -> bool:
    # "kohli" ends "v kohli"; "sciver brunt" ends "nr sciver brunt"
    for name in names:
        if name.endswith(" " + wanted):
            return True
    return False


def _starts_with_first_name(names: list[str], wanted: str) -> bool:
    # "shakib" starts "shakib al hasan"; a first word of one or two letters is
    # initials ("ms dhoni"), not a first name
    for name in names:
        words = name.split()
        if len(words) >= 2 and len(words[0]) >= 3 and name.startswith(wanted + " "):
            return True
    return False


def _most_played_first(player: Player) -> tuple[int, str]:
    return (-player.matches, player.name)


def name_list(players: list[Player]) -> dict[str, tuple[str, ...]]:
    """Every way to say each player, for C6's gazetteer: "V Kohli" -> ("Kohli", ...)."""
    names = {}
    for player in players:
        ways = []
        last = player.name.split()[-1]
        if last != player.name:
            ways.append(last)  # the surname alone
        for alias in player.aliases:
            if alias not in ways:
                ways.append(alias)
        names[player.name] = tuple(ways)
    return names
