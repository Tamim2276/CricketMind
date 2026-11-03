"""Evidence (N7a): how many balls a subject is about, filed under that subject (Day 60).

Subject-keyed: Kohli's count is filed under Kohli's subject, so it can never be passed off
as Rohit's. The engines fetch their own balls; the router reads this count (Day 64).
"""

import dataclasses
from dataclasses import dataclass

from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.subject import CricketSubject


@dataclass(frozen=True, kw_only=True)
class Evidence:
    subject: CricketSubject  # the key: exactly what these balls fit
    count: int = 0  # how many balls, wides included
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
