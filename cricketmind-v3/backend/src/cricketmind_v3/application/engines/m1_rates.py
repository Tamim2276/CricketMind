"""N8 M1: a player's strike rate and dismissal rate, with the balls behind them."""

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain import measures
from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def strike_rate(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    balls = _balls(repo, "strike_rate", subject)
    if balls is None:
        return None
    return measures.strike_rate(balls)


def dismissal_rate(repo: DeliveryRepository, subject: Subject) -> Fact | None:
    balls = _balls(repo, "dismissal_rate", subject)
    if balls is None:
        return None
    return measures.dismissal_rate(balls)


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
