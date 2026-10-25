"""N9 T1: the chance a batter is out within his next k balls, with its range (Day 49)."""

import dataclasses

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain import out_within
from cricketmind_v3.domain.balls import tally
from cricketmind_v3.domain.facts import Fact, Replays
from cricketmind_v3.domain.measures import outs_and_faced
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def p_dismissal_within_k(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject.

    The batter's own balls under the subject's filters give his outs and balls faced;
    k, how many balls ahead, says how far to play them forward.
    """
    claim_type = lookup("p_dismissal_within_k")
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None
    if not isinstance(subject.k, int) or subject.k < 1:
        return None  # "within 0 balls" is not a question
    past = dataclasses.replace(subject, k=None)  # k is about the future, not a filter
    try:
        balls = repo.deliveries(past)
    except FilterError:
        return None  # a shot, before the bridge (Day 128): no answer
    outs, faced = outs_and_faced(tally(balls))
    found = out_within.out_within(outs, faced, subject.k)
    if found is None:
        return None
    return Fact(
        value=found.chance,
        n=faced,
        interval=(found.low, found.high),
        replays=Replays(survival=found.survival, overs=found.overs),
    )
