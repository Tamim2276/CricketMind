from datetime import date
from typing import Protocol

from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import Subject
from cricketmind_v3.domain.venues import InningsScore, Wicket
from cricketmind_v3.domain.workload import Spell


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


class VenueRepository(Protocol):
    """Anything that can tell what happened at a ground: the database, or a fake."""

    def innings_at(
        self, venue: str, match_type: str, gender: str
    ) -> list[InningsScore]:
        """Every innings played at the ground in that format, in match order."""

    def wickets_at(self, venue: str, match_type: str, gender: str) -> dict[Wicket, int]:
        """Every wicket that fell at the ground in that format, counted by kind."""


class FakeVenueRepository:
    """Innings and wickets written in by hand, for tests, by ground, format and
    gender."""

    def __init__(self) -> None:
        self.innings: dict[tuple[str, str, str], list[InningsScore]] = {}
        self.wickets: dict[tuple[str, str, str], dict[Wicket, int]] = {}

    def add_innings(
        self, venue: str, match_type: str, gender: str, scores: list[InningsScore]
    ) -> None:
        self.innings[(venue, match_type, gender)] = scores

    def add_wickets(
        self, venue: str, match_type: str, gender: str, counts: dict[Wicket, int]
    ) -> None:
        self.wickets[(venue, match_type, gender)] = counts

    def innings_at(
        self, venue: str, match_type: str, gender: str
    ) -> list[InningsScore]:
        return list(self.innings.get((venue, match_type, gender), []))

    def wickets_at(self, venue: str, match_type: str, gender: str) -> dict[Wicket, int]:
        return dict(self.wickets.get((venue, match_type, gender), {}))


class WorkloadRepository(Protocol):
    """Anything that can tell when a bowler bowled: the database, or a fake (Day 41)."""

    def spells(self, bowler: str) -> list[Spell]:
        """His balls in every match he played, in date order.

        Raises FilterError when several players share the name.
        """

    def data_until(self) -> date:
        """The last match day in the data: a week after it would look empty."""


class FakeWorkloadRepository:
    """Spells written in by hand, for tests."""

    def __init__(self, until: date) -> None:
        self.until = until
        self.bowlers: dict[str, list[Spell]] = {}

    def add(self, bowler: str, spells: list[Spell]) -> None:
        self.bowlers[bowler] = spells

    def spells(self, bowler: str) -> list[Spell]:
        return list(self.bowlers.get(bowler, []))

    def data_until(self) -> date:
        return self.until


class ReviewRepository(Protocol):
    """Anything that can count a team's DRS reviews: the database, or a fake (Day 42)."""

    def decisions(self, team: str, gender: str) -> dict[str, int]:
        """How the team's reviews ended: "upheld" -> 81, "struck down" -> 219."""


class FakeReviewRepository:
    """Review counts written in by hand, for tests."""

    def __init__(self) -> None:
        self.teams: dict[tuple[str, str], dict[str, int]] = {}

    def add(self, team: str, gender: str, decisions: dict[str, int]) -> None:
        self.teams[(team, gender)] = decisions

    def decisions(self, team: str, gender: str) -> dict[str, int]:
        return dict(self.teams.get((team, gender), {}))
