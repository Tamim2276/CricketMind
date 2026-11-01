"""The question's context (N7b): who the people are and where the ground is (Day 60).

Words from the registers, never a statistic: the numbers come only from the engines.
"""

from dataclasses import dataclass

from cricketmind_v3.domain.grounds import Ground, describe_ground
from cricketmind_v3.domain.names import Player, describe
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.domain.teams import Team, describe_team, other_name

PEOPLE = ("player", "bowler", "other_bowler")
SIDES = ("team", "opposition")


@dataclass(frozen=True, kw_only=True)
class QuestionContext:
    subject: CricketSubject  # what it describes
    people: tuple[str, ...] = ()  # "V Kohli; India, ...; Test, ODI; 2008-2025"
    ground: str | None = None  # "Wankhede Stadium, Mumbai; Mumbai; 2008-2025"
    teams: tuple[str, ...] = ()  # "Delhi Capitals; 2019-2025; also Delhi Daredevils"
    not_found: tuple[str, ...] = ()  # a name none of the registers given holds


def question_context(
    subject: CricketSubject,
    players: tuple[Player, ...],
    grounds: tuple[Ground, ...],
    teams: tuple[Team, ...],
) -> QuestionContext:
    """Every person, ground and team the subject names, as the registers describe them."""
    people = []
    sides = []
    ground = None
    not_found = []
    for name in PEOPLE:
        value = getattr(subject, name)
        if is_empty(value):
            continue
        player = _player(value, players)
        if player is None:
            not_found.append(value)
        else:
            people.append(describe(player))
    if not is_empty(subject.venue):
        ground = _ground(subject.venue, grounds)
        if ground is None:
            not_found.append(subject.venue)
    for name in SIDES:
        value = getattr(subject, name)
        if is_empty(value):
            continue
        team = _team(value, teams)
        if team is None:
            not_found.append(value)
        else:
            sides.append(team)
    return QuestionContext(
        subject=subject,
        people=tuple(people),
        ground=ground,
        teams=tuple(sides),
        not_found=tuple(not_found),
    )


def _player(value: str, players: tuple[Player, ...]) -> Player | None:
    # a register id first (a shared name, Day 58), then a name only one player has
    named = []
    for player in players:
        if player.person_id == value:
            return player
        if player.name == value:
            named.append(player)
    if len(named) == 1:
        return named[0]
    return None


def _ground(value: str, grounds: tuple[Ground, ...]) -> str | None:
    for ground in grounds:
        if ground.name == value:
            return describe_ground(ground)
    return None


def _team(value: str, teams: tuple[Team, ...]) -> str | None:
    for team in teams:
        if team.name == value:
            text = describe_team(team)
            also = other_name(team)
            if also is not None:
                text += "; also played as " + also  # its matches are not counted here
            return text
    return None
