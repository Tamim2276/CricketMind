"""Who a name in a question means (Day 33): "Kohli" -> "V Kohli". And, when several
players have it, what tells them apart (Day 58): never a statistic, only who they are."""

import difflib
from dataclasses import dataclass

# the top player is kept only with this many times the next one's matches: "Kohli" is
# V Kohli (837 matches) before A Kohli (41); "Smith" (574 against 319) is asked back.
# A design parameter, like D-010's (Day 58)
MOST_PLAYED_BY = 5
CLOSE_SPELLING = 0.8  # how alike two names must be to be offered: "kholi", "kohli"


@dataclass(frozen=True, kw_only=True)
class Player:
    """One person, as Cricsheet's register knows them."""

    person_id: str  # "ba607b88"
    name: str  # "V Kohli": the name used most in the data
    aliases: tuple[str, ...] = ()  # other names: "Virat Kohli" (Day 34)
    matches: int = 0  # how many matches they played: the tie-breaker
    # what tells two players of one name apart (Day 58)
    teams: tuple[
        str, ...
    ] = ()  # "India", "Royal Challengers Bengaluru": most played first
    formats: tuple[str, ...] = ()  # "Test", "ODI", "T20I", "T20" (D-047)
    first_year: int | None = None
    last_year: int | None = None
    gender: str | None = None  # "male" or "female"


@dataclass(frozen=True, kw_only=True)
class Clues:
    """What a question says that can tell players of one name apart."""

    teams: tuple[str, ...] = ()  # the sides it names, his or the other: "Australia"
    match_format: str | None = None  # "Test"
    gender: str | None = None
    from_year: int | None = None  # since, or a season
    to_year: int | None = None  # before, or a season


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

    1. the whole name: "V Kohli", or an alias like "Virat Kohli"; and, together with
       it, given names he is known by and the surname: "Kamindu Mendis" is "Pasqual
       Handi Kamindu Dilanka Mendis" (Day 58: his K is his third initial). Together,
       because two players registered as "Rohit Sharma" (5 and 7 matches) must not
       hide RG Sharma, "Rohit Gurunath Sharma" (785)
    2. a first name's initial and the surname: "Rohit Sharma" -> "RG Sharma"
    3. one part of the name, surname or first name, most-played first: "Kohli" ->
       "V Kohli"; "Mendis" -> both Mendises; "Babar" -> "Babar Azam" before
       "Zulfiqar Babar" (many players are known by their first name: found on Day 33)
    """
    wanted = simplify(query)
    if not wanted:
        return []
    whole = []
    given = []
    initial = []
    part = []
    for player in players:
        names = [simplify(player.name)]
        for alias in player.aliases:
            names.append(simplify(alias))
        if wanted in names:
            whole.append(player)
        elif _given_names_and_surname(wanted, names):
            given.append(player)
        elif _initial_and_surname(wanted, names[0]):
            initial.append(player)
        elif _ends_with(names, wanted) or _starts_with_first_name(names, wanted):
            part.append(player)
    for found in (whole + given, initial, part):
        if found:
            return sorted(found, key=_most_played_first)
    return []


def _given_names_and_surname(wanted: str, names: list[str]) -> bool:
    # "kamindu mendis" in "pasqual handi kamindu dilanka mendis": the same surname, and
    # every other word asked is one of his given names
    asked = wanted.split()
    if len(asked) < 2:
        return False
    for name in names:
        known = name.split()
        if len(known) < 2 or known[-1] != asked[-1]:
            continue
        every = True
        for word in asked[:-1]:
            if word not in known[:-1]:
                every = False
        if every:
            return True
    return False


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
    """Every way to say each player, for C6's gazetteer: "V Kohli" -> ("Kohli", ...).

    A name several players share is listed under each one's register id, with the name
    among its ways: the id is what N3 puts in the subject for them (Day 58).
    """
    shared = shared_names(players)
    names = {}
    for player in players:
        ways = []
        key = player.name
        if player.name in shared:
            key = player.person_id
            ways.append(player.name)
        last = player.name.split()[-1]
        if last != player.name:
            ways.append(last)  # the surname alone
        for alias in player.aliases:
            if alias not in ways:
                ways.append(alias)
        names[key] = tuple(ways)
    return names


def shared_names(players: list[Player]) -> set[str]:
    """The names more than one player has, word for word: "Imran Khan"."""
    seen = set()
    shared = set()
    for player in players:
        if player.name in seen:
            shared.add(player.name)
        seen.add(player.name)
    return shared


def register_name(player: Player, shared: set[str]) -> str:
    """How the subject names him: his name, or his register id if others have it."""
    if player.name in shared:
        return player.person_id
    return player.name


# telling players of one name apart (Day 58)


def close_spellings(written: str, players: list[Player]) -> list[Player]:
    """Players whose name, or surname, is spelt nearly like this: "Kholi" finds the
    Kohlis. Only ever offered, never taken: a near name can be another player."""
    wanted = simplify(written)
    spellings = {}
    for player in players:
        for name in (player.name, *player.aliases):
            whole = simplify(name)
            _note(spellings, whole, player)
            parts = whole.split()
            if len(parts) >= 2:
                _note(spellings, parts[-1], player)  # the surname alone
    found = []
    for spelling in difflib.get_close_matches(
        wanted, list(spellings), n=5, cutoff=CLOSE_SPELLING
    ):
        if spelling == wanted:
            continue  # the same spelling was a find, not a near miss
        for player in spellings[spelling]:
            if player not in found:
                found.append(player)
    return sorted(found, key=_most_played_first)


def _note(spellings: dict[str, list[Player]], spelling: str, player: Player) -> None:
    if spelling not in spellings:
        spellings[spelling] = []
    if player not in spellings[spelling]:
        spellings[spelling].append(player)


def prefer_by_clues(candidates: list[Player], clues: Clues) -> list[Player]:
    """The candidates the question's clues fit: his team, the format, men or women, the
    years. A clue that fits none of them is set aside, not used to rule everyone out."""
    kept = list(candidates)
    for fits in (_fits_gender, _fits_format, _fits_team, _fits_years):
        narrowed = []
        for player in kept:
            if fits(player, clues):
                narrowed.append(player)
        if narrowed:
            kept = narrowed
    return kept


def clearly_most_played(candidates: list[Player]) -> bool:
    """One player has played far more than the next (MOST_PLAYED_BY times)."""
    if len(candidates) < 2:
        return len(candidates) == 1
    ranked = sorted(candidates, key=_most_played_first)
    return ranked[0].matches >= MOST_PLAYED_BY * ranked[1].matches


def describe(player: Player) -> str:
    """Who he is, for a question back: never a statistic."""
    parts = [player.name]
    if player.teams:
        parts.append(", ".join(player.teams[:3]))
    if player.formats:
        parts.append(", ".join(player.formats))
    if player.first_year is not None:
        parts.append(f"{player.first_year}-{player.last_year}")
    return "; ".join(parts)


def _fits_gender(player: Player, clues: Clues) -> bool:
    return clues.gender is None or player.gender == clues.gender


def _fits_format(player: Player, clues: Clues) -> bool:
    return clues.match_format is None or clues.match_format in player.formats


def _fits_team(player: Player, clues: Clues) -> bool:
    if not clues.teams:
        return True
    for named in clues.teams:
        for team in player.teams:
            if simplify(team) == simplify(named):
                return True
    return False


def clue_fits(clue: str, player: Player) -> bool:
    """A clue's words fit who he is: a side he played for ("Australian" fits
    Australia), or a year of his career."""
    said = simplify(clue)
    for team in player.teams:
        if (" " + simplify(team)) in (" " + said):  # at the start of a word
            return True
    if player.first_year is None:
        return False
    for word in said.split():
        is_year = word.isdigit() and len(word) == 4
        if is_year and player.first_year <= int(word) <= player.last_year:
            return True
    return False


def _fits_years(player: Player, clues: Clues) -> bool:
    if player.first_year is None:
        return clues.from_year is None and clues.to_year is None
    if clues.from_year is not None and player.last_year < clues.from_year:
        return False  # he had stopped before it
    return clues.to_year is None or player.first_year <= clues.to_year
