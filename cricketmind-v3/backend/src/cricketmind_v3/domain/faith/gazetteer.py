"""The names C6 knows: players, the fifteen shots, phases and bowler types."""

from dataclasses import dataclass

from cricketmind_v3.domain.faith.text import tokens


@dataclass(frozen=True, kw_only=True)
class Entry:
    """What a name stands for, and which subject fields it may fill."""

    value: str  # "V Kohli", "Cover Drive", "powerplay"
    fields: tuple[str, ...]  # ("shot", "other_shot")


PERSON = ("player", "bowler", "other_bowler")  # a name may bat or bowl
SHOT = ("shot", "other_shot")

# the fifteen CricShot10k shots, and other ways people say them
SHOTS = {
    "Cover Drive": ("cover drives", "drive through cover", "drive through covers"),
    "Defensive": ("defensive shot", "forward defence", "forward defense"),
    "Down The Wicket": ("down the track",),
    "Flick": ("flicks",),
    "Hook": ("hooks",),
    "Late Cut": ("late cuts",),
    "Lofted Legside": ("lofted leg side",),
    "Lofted Offside": ("lofted off side",),
    "Pull": ("pulls",),
    "Reverse Sweep": ("reverse sweeps",),
    "Scoop": ("scoops", "ramp"),
    "Square Cut": ("square cuts",),
    "Straight Drive": ("straight drives",),
    "Sweep": ("sweeps",),
    "Upper Cut": ("uppercut", "upper cuts"),
}

PHASES = {
    "powerplay": ("power play",),
    "middle": ("middle overs",),
    "death": ("death overs", "at the death"),
}

BOWLER_TYPES = {
    "pace": (
        "pacers",
        "pace bowling",
        "pace bowlers",
        "fast bowling",
        "fast bowlers",
        "seam",
        "seamers",
        "quicks",
    ),
    "spin": ("spinners", "spin bowling", "slow bowling", "slow bowlers"),
}


def cricket_gazetteer(players: dict[str, tuple[str, ...]]) -> dict[str, list[Entry]]:
    """Every name C6 knows: phrase -> what it may stand for.

    players maps a register name to other ways of saying it, as in
    {"V Kohli": ("Kohli", "Virat Kohli")}. The full register comes on Day 33.
    """
    names = {}
    for player, others in players.items():
        _add_all(names, player, others, PERSON)
    for shot, others in SHOTS.items():
        _add_all(names, shot, others, SHOT)
    for phase, others in PHASES.items():
        _add_all(names, phase, others, ("phase",))
    for kind, others in BOWLER_TYPES.items():
        _add_all(names, kind, others, ("bowler_type",))
    return names


def _add_all(
    names: dict[str, list[Entry]],
    value: str,
    others: tuple[str, ...],
    fields: tuple[str, ...],
) -> None:
    entry = Entry(value=value, fields=fields)
    _add(names, value, entry)
    for other in others:
        _add(names, other, entry)


def _add(names: dict[str, list[Entry]], phrase: str, entry: Entry) -> None:
    key = " ".join(tokens(phrase))  # "Power-Play" and "power play" are one name
    if key not in names:
        names[key] = []
    if entry not in names[key]:
        names[key].append(entry)  # a surname can belong to two players
