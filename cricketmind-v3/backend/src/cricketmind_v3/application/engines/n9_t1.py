"""N9 T1: the chance a batter is out within his next k balls, with its range (Days 49-50)."""

import dataclasses

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain import out_within
from cricketmind_v3.domain.balls import Ball, tally
from cricketmind_v3.domain.facts import Fact, Replays
from cricketmind_v3.domain.measures import outs_and_faced
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.shrinkage import population_prior
from cricketmind_v3.domain.stages import (
    STARTS,
    Record,
    by_batter,
    in_order,
    in_stage,
    stage_of,
    usual_order,
)
from cricketmind_v3.domain.subject import Subject


def p_dismissal_within_k(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject.

    The batter's own balls under the subject's filters give his record; every other
    batter's, under the same filters and batting where he bats, give what is normal,
    and he is pulled toward it (Day 50). k, how many balls ahead, says how far to play
    them forward.
    """
    claim_type = lookup("p_dismissal_within_k")
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None
    if not isinstance(subject.k, int) or subject.k < 1:
        return None  # "within 0 balls" is not a question
    so_far = subject.faced_so_far
    if so_far is not None and (not isinstance(so_far, int) or so_far < 0):
        return None
    # k and faced_so_far are about now and the future, not filters on the past
    past = dataclasses.replace(subject, k=None, faced_so_far=None)
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
        his = repo.records(past)
        order = usual_order(his)
        if order is None:
            return None
        theirs = in_order(repo.others_records(like_him), order)
        if so_far is None:
            return _one_risk(theirs, balls, subject.k)
        return _by_stage(his, theirs, so_far, subject.k)
    except FilterError:
        return None  # a shot, before the bridge (Day 129): no answer


def _one_risk(theirs: list[Record], balls: list[Ball], k: int) -> Fact | None:
    # one risk for the whole innings, pulled toward the other batters' (Day 50)
    prior = population_prior(by_batter(theirs))
    if prior is None:
        return None  # too few others, or no sign they differ
    outs, faced = outs_and_faced(tally(balls))
    return _fact(out_within.out_within(outs, faced, k, prior), faced)


def _by_stage(
    his: list[Record], theirs: list[Record], so_far: int, k: int
) -> Fact | None:
    # how long he has been in (on trial, D-056): each stage his next k balls fall in,
    # his record there pulled toward the other batters' in that stage
    stages = []
    faced_here = 0
    for number in range(stage_of(so_far + 1), stage_of(so_far + k) + 1):
        prior = population_prior(in_stage(theirs, number))
        if prior is None:
            return None
        outs = 0
        faced = 0
        for record_outs, record_faced in in_stage(his, number):
            outs += record_outs
            faced += record_faced
        faced_here += faced
        stages.append(
            out_within.Stage(
                first=STARTS[number - 1], outs=outs, faced=faced, prior=prior
            )
        )
    return _fact(out_within.out_within_stages(stages, k, so_far), faced_here)


def _fact(found: out_within.OutWithin | None, faced: int) -> Fact | None:
    if found is None:
        return None
    return Fact(
        value=found.chance,
        n=faced,
        interval=(found.low, found.high),
        replays=Replays(survival=found.survival, overs=found.overs),
    )
