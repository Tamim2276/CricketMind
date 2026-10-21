from typing import Protocol

from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import Subject


class FilterError(ValueError):
    """A subject asks for something the repository cannot pick balls by."""


class DeliveryRepository(Protocol):
    """Anything that can give the balls a subject is about: the database, or a fake."""

    def deliveries(self, subject: Subject) -> list[Ball]:
        """Every ball that fits all of the subject's filled fields, in match order.

        Raises FilterError rather than leave out a filter it cannot apply.
        """


class FakeDeliveryRepository:
    """Balls written in by hand, for tests: the same balls whatever is asked."""

    def __init__(self, balls: list[Ball]) -> None:
        self.balls = balls
        self.asked: list[Subject] = []  # what the engine asked for, to check in tests

    def deliveries(self, subject: Subject) -> list[Ball]:
        self.asked.append(subject)
        return list(self.balls)


class FactSource(Protocol):
    """Anything that can compute a number for a subject: the engines, or a fake."""

    def fact(
        self, metric: str, subject: Subject, clip_id: str | None = None
    ) -> Fact | None:
        """The engine's own answer, or None when it has none for this subject."""


class FakeFactSource:
    """Facts written in by hand, for tests and the synthetic harness."""

    def __init__(self) -> None:
        self.facts: dict[tuple[str, Subject, str | None], Fact] = {}

    def add(
        self, metric: str, subject: Subject, fact: Fact, clip_id: str | None = None
    ) -> None:
        self.facts[(metric, subject, clip_id)] = fact

    def fact(
        self, metric: str, subject: Subject, clip_id: str | None = None
    ) -> Fact | None:
        return self.facts.get((metric, subject, clip_id))

    def numbers(self) -> tuple[float, ...]:
        """Every number this source can give: v2's sanctioned fact surface."""
        found = []
        for fact in self.facts.values():
            found.append(fact.value)
        return tuple(found)
