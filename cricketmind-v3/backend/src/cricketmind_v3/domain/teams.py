"""Which team a question means (Day 58, part 2c).

A team is found by its name in the data ("India"), by what people call it, checked by
hand ("Windies", "RCB"), or by words of its name ("Sunrisers": two teams, asked). A team
that played under another name keeps both names apart: "Delhi Capitals" counts only the
matches it played as Delhi Capitals, and the reading says so (D-078).
"""

from dataclasses import dataclass

from cricketmind_v3.domain.grounds import squash


@dataclass(frozen=True, kw_only=True)
class Team:
    """One team, as the data names it."""

    name: str  # "Delhi Capitals"
    matches: int = 0
    first_year: int | None = None
    last_year: int | None = None


# what people call a team -> its name in the data, each checked by hand
NICKNAMES = {
    "windies": "West Indies",
    "proteas": "South Africa",
    "blackcaps": "New Zealand",
    "kiwis": "New Zealand",
    "aussies": "Australia",
    "uae": "United Arab Emirates",
    "rcb": "Royal Challengers Bengaluru",
    "csk": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad",
    "dc": "Delhi Capitals",
    "pbks": "Punjab Kings",
    "rr": "Rajasthan Royals",
    "lsg": "Lucknow Super Giants",
    "gt": "Gujarat Titans",
}

# teams that played under another name: the older name -> the newer one. Kept apart in
# the counts; the reading names the other (D-078)
RENAMED_TEAMS = {
    "Delhi Daredevils": "Delhi Capitals",  # 2019
    "Kings XI Punjab": "Punjab Kings",  # 2021
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",  # 2024
    "Rising Pune Supergiants": "Rising Pune Supergiant",  # 2017
}


def find_teams(written: str, teams: list[Team]) -> list[Team]:
    """Which teams the words could be, most-played first: a nickname; else every team
    whose name has the words in a row, the whole name among them. Together, because a
    team named exactly "Sunrisers" (76 matches) must not hide Sunrisers Hyderabad (211):
    the reader keeps the clearly most-played ("India", 1,353), or asks."""
    wanted = squash(written)
    if not wanted:
        return []
    nickname = NICKNAMES.get(wanted)
    if nickname is not None:
        for team in teams:
            if team.name == nickname:
                return [team]
    words = _words(written)
    found = []
    for team in teams:
        said = _words(team.name)
        for start in range(len(said) - len(words) + 1):
            if said[start : start + len(words)] == words:
                found.append(team)
                break
    return sorted(found, key=_most_played_first)


def other_name(team: Team) -> str | None:
    """The name it also played under, if it was renamed: Delhi Daredevils."""
    for old, new in RENAMED_TEAMS.items():
        if team.name == new:
            return old
        if team.name == old:
            return new
    return None


def describe_team(team: Team) -> str:
    """Which team it is, for a question back: never a statistic."""
    if team.first_year is None:
        return team.name
    return f"{team.name}; {team.first_year}-{team.last_year}"


def _words(text: str) -> list[str]:
    found = []
    for word in text.split():
        squashed = squash(word)
        if squashed:
            found.append(squashed)
    return found


def _most_played_first(team: Team) -> tuple[int, str]:
    return (-team.matches, team.name)
