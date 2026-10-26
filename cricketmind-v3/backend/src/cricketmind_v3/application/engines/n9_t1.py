"""N9 T1: the chance a batter is out within his next k balls, with its range (Days 49-53)."""

import dataclasses

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain import out_within
from cricketmind_v3.domain.balls import tally
from cricketmind_v3.domain.facts import Fact, Replays
from cricketmind_v3.domain.measures import outs_and_faced
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.shrinkage import population_prior
from cricketmind_v3.domain.stages import by_batter, in_order, usual_order
from cricketmind_v3.domain.subject import Subject


def p_dismissal_within_k(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject.

    The batter's own balls under the subject's filters give his record; every other
    batter's, under the same filters and batting where he bats, give what is normal,
    and he is pulled toward it (Day 50). k, how many balls ahead, says how far to play
    them forward. One risk for his whole innings: by stage was tried, not kept (D-066).
    """
    claim_type = lookup("p_dismissal_within_k")
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None
    if not isinstance(subject.k, int) or subject.k < 1:
        return None  # "within 0 balls" is not a question
    past = dataclasses.replace(subject, k=None)  # k is about the future, not a filter
    try:
        balls = repo.deliveries(past)
        genders = set()
        for ball in balls:
            genders.add(ball.gender)
        if len(genders) != 1:
            return None  # no balls: nothing to go on
        # normal is for his kind of cricket, his format and men's or women's, and for
        # batters who bat where he usually bats: a tailender is not a top-order batter
        like_him = dataclasses.replace(past, gender=genders.pop())
        order = usual_order(repo.records(past))
        if order is None:
            return None
        theirs = in_order(repo.others_records(like_him), order)
    except FilterError:
        return None  # a shot, before the bridge (Day 129): no answer
    prior = population_prior(by_batter(theirs))
    if prior is None:
        return None  # too few others, or no sign they differ
    outs, faced = outs_and_faced(tally(balls))
    found = out_within.out_within(outs, faced, subject.k, prior)
    if found is None:
        return None
    return Fact(
        value=found.chance,
        n=faced,
        interval=(found.low, found.high),
        replays=Replays(survival=found.survival, overs=found.overs),
    )
