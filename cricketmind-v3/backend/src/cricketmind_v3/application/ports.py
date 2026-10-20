from typing import Protocol

from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import Subject


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
