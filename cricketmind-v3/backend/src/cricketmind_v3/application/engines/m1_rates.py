"""N8 M1: a player's rates with the balls behind them, and differences with their range."""

import dataclasses

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain import bootstrap, measures
from cricketmind_v3.domain.balls import Ball, tally
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import lookup, missing_keys
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


def sr_difference(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The shot's strike rate minus the other shot's (Day 39)."""
    if missing_keys(lookup("sr_difference"), subject):
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
    # the menu says what the measure needs: no player, no rate (not everyone's balls)
    if missing_keys(lookup(metric), subject):
        return None
    try:
        return repo.deliveries(subject)
    except FilterError:
        return None  # a filter it cannot apply: no answer, never a different question
