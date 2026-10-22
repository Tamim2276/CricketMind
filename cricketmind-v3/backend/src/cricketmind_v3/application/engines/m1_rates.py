"""N8 M1: any measure of a batter or a bowler, under any filters (Days 38-44)."""

import dataclasses

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain import bootstrap, measures
from cricketmind_v3.domain.balls import Ball, tally
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def strike_rate(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    balls = _balls(repo, "strike_rate", subject)
    if balls is None:
        return None
    return measures.strike_rate(tally(balls))


def dismissal_rate(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    balls = _balls(repo, "dismissal_rate", subject)
    if balls is None:
        return None
    return measures.dismissal_rate(tally(balls))


# More measures (Day 44): the same balls, counted another way


def runs(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    return _measure(repo, "runs", subject, measures.runs)


def batting_average(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """Runs per time out, his run outs at the non-striker's end included."""
    balls = _balls(repo, "batting_average", subject)
    if balls is None:
        return None
    try:
        other_end = repo.outs_at_other_end(subject)
    except FilterError:
        return None
    return measures.batting_average(tally(balls), other_end)


def boundary_share(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    return _measure(repo, "boundary_share", subject, measures.boundary_share)


def dot_ball_share(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    return _measure(repo, "dot_ball_share", subject, measures.dot_ball_share)


def wickets(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """A bowler's wickets; the subject names him as its bowler."""
    return _measure(repo, "wickets", subject, measures.wickets)


def economy(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    return _measure(repo, "economy", subject, measures.economy)


def bowling_strike_rate(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    return _measure(repo, "bowling_strike_rate", subject, measures.bowling_strike_rate)


def _measure(repo, metric: str, subject: Subject, measure) -> Fact | None:
    # the balls the menu row allows, counted by the measure
    balls = _balls(repo, metric, subject)
    if balls is None:
        return None
    return measure(tally(balls))


def sr_difference(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The shot's strike rate minus the other shot's (Day 39)."""
    if not _fits(lookup("sr_difference"), subject):
        return None
    shot = dataclasses.replace(subject, other_shot=None)
    other_shot = dataclasses.replace(subject, shot=subject.other_shot, other_shot=None)
    try:
        first = repo.deliveries(shot)
        second = repo.deliveries(other_shot)
    except FilterError:
        return None
    return bootstrap.difference(tally(first), tally(second), measures.strike_rate)


def sr_vs_average(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The player's strike rate minus every other batter's, under the same filters."""
    balls = _balls(repo, "sr_vs_average", subject)
    if balls is None:
        return None
    try:
        everyone_else = repo.others(subject)
    except FilterError:
        return None
    return bootstrap.versus_average(tally(balls), everyone_else, measures.strike_rate)


def _balls(
    repo: DeliveryRepository, metric: str, subject: Subject
) -> list[Ball] | None:
    if not _fits(lookup(metric), subject):
        return None
    try:
        return repo.deliveries(subject)
    except FilterError:
        return None  # a filter it cannot apply: no answer, never a different question


def _fits(claim_type, subject: Subject) -> bool:
    # the menu says what the measure needs (no player, no rate: not everyone's balls)
    # and what it may be narrowed by; anything else is a different question (Day 43)
    return not missing_keys(claim_type, subject) and not extra_keys(claim_type, subject)
