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

    def others(self, subject: Subject) -> dict[Ball, int]:
        """Every other batter's balls under the same filters, as a tally (Day 39).

        Millions of balls, so they come already counted, with THE_BATTER and
        SOMEONE_ELSE in place of the names.
        """


class FakeDeliveryRepository:
    """Balls written in by hand for tests, filed under the subject they answer."""

    def __init__(self) -> None:
        self.balls: dict[Subject, list[Ball]] = {}
        self.everyone_else: dict[Subject, dict[Ball, int]] = {}
        self.asked: list[Subject] = []  # what the engine asked for, to check in tests

    def add(self, subject: Subject, balls: list[Ball]) -> None:
        self.balls[subject] = balls

    def add_others(self, subject: Subject, counts: dict[Ball, int]) -> None:
        self.everyone_else[subject] = counts

    def deliveries(self, subject: Subject) -> list[Ball]:
        self.asked.append(subject)
        return list(self.balls.get(subject, []))

    def others(self, subject: Subject) -> dict[Ball, int]:
        self.asked.append(subject)
        return dict(self.everyone_else.get(subject, {}))


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
