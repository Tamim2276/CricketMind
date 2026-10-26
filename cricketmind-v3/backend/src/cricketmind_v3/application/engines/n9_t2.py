"""N9 T2: what if this bowler bowled to him instead of that one? (Days 51-52)"""

import dataclasses

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain import matchups
from cricketmind_v3.domain.facts import Fact, Replays
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.stages import usual_order
from cricketmind_v3.domain.subject import Subject


def matchup_difference(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The chance he is out within his next k balls against the bowler, minus against
    the other bowler, with its range; None when it has no answer.

    Everyone's balls under the subject's filters, in his kind of cricket and his part of
    the order, give each batter's and each bowler's record and their meetings.
    """
    claim_type = lookup("matchup_difference")
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None
    if not isinstance(subject.k, int) or subject.k < 1:
        return None
    if subject.bowler == subject.other_bowler:
        return None  # a bowler instead of himself is not a question
    # his own balls, against any bowler: his kind of cricket and where he bats
    his = dataclasses.replace(subject, k=None, bowler=None, other_bowler=None)
    try:
        genders = set()
        for ball in repo.deliveries(his):
            genders.add(ball.gender)
        if len(genders) != 1:
            return None
        order = usual_order(repo.records(his))
        if order is None:
            return None
        # every batter and bowler under the same filters, men's or women's like him
        everyone = dataclasses.replace(his, player=None, gender=genders.pop())
        pairs = matchups.in_order(repo.pairs(everyone), order)
        batter = repo.person(subject.player)
        this_bowler = repo.person(subject.bowler)
        that_bowler = repo.person(subject.other_bowler)
    except FilterError:
        return None  # a name several players share
    if batter is None or this_bowler is None or that_bowler is None:
        return None
    found = matchups.matchup(pairs, batter, this_bowler, that_bowler, subject.k)
    if found is None:
        return None
    that = Replays(survival=found.that.survival, overs=found.that.overs)
    return Fact(
        value=found.difference,
        n=found.n,
        interval=(found.low, found.high),
        replays=Replays(
            survival=found.this.survival, overs=found.this.overs, versus=that
        ),
    )
