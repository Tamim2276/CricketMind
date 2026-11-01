"""Evidence (N7a): the balls a subject is about, filed under that subject (Day 60).

Subject-keyed: Kohli's balls are filed under Kohli's subject, so a number counted from
them can never be passed off as Rohit's.
"""

import dataclasses
from dataclasses import dataclass, field

from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.subject import CricketSubject


@dataclass(frozen=True, kw_only=True)
class Evidence:
    subject: CricketSubject  # the key: exactly what these balls fit
    count: int = 0  # how many balls, wides included
    # each different ball, and how often it happened (tally, Day 39)
    balls: dict[Ball, int] = field(default_factory=dict)
    none_because: str | None = None  # why there are no balls, when there are none


def sides(subject: CricketSubject) -> list[CricketSubject]:
    """One subject for each side of a comparison, each without k: k is how many balls
    ahead a prediction looks, not a filter on the past."""
    first = dataclasses.replace(subject, k=None, other_shot=None, other_bowler=None)
    found = [first]
    if not is_empty(subject.other_shot):
        found.append(dataclasses.replace(first, shot=subject.other_shot))
    if not is_empty(subject.other_bowler):
        found.append(dataclasses.replace(first, bowler=subject.other_bowler))
    return found
