from datetime import date
from typing import Protocol

from pydantic import BaseModel, ValidationError

from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.calibration import Moment
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.matchups import Pair
from cricketmind_v3.domain.results import Result
from cricketmind_v3.domain.stages import Record
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

    def outs_at_other_end(self, subject: Subject) -> int:
        """How often the subject's player was out as the non-striker (a run out), on
        balls that fit the other filters (Day 44). They count in his average."""

    def records(self, subject: Subject) -> list[Record]:
        """The player's times out and balls faced under the subject's filters, one
        record for each stage of his innings and part of the order (Day 50)."""

    def others_records(self, subject: Subject) -> list[Record]:
        """Every other batter's, one record per batter, stage and part of the order:
        what is normal, the average he is pulled toward (Day 50)."""

    def pairs(self, subject: Subject) -> list[Pair]:
        """Every batter against every bowler under the subject's filters, one pair per
        batter, bowler and part of the order (Day 51)."""

    def person(self, name: str) -> str | None:
        """The register id of the player called this; None if nobody is (Day 51).

        Raises FilterError when several players share the name.
        """

    def moments(self, subject: Subject) -> list[Moment]:
        """Every ball faced, and every ball a batter was out on, under the subject's
        filters, in match order (Day 53)."""


class FakeDeliveryRepository:
    """Balls written in by hand for tests, filed under the subject they answer."""

    def __init__(self) -> None:
        self.balls: dict[Subject, list[Ball]] = {}
        self.everyone_else: dict[Subject, dict[Ball, int]] = {}
        self.other_end: dict[Subject, int] = {}
        self.his_records: dict[Subject, list[Record]] = {}
        self.their_records: dict[Subject, list[Record]] = {}
        self.every_pair: dict[Subject, list[Pair]] = {}
        self.every_moment: dict[Subject, list[Moment]] = {}
        self.asked: list[Subject] = []  # what the engine asked for, to check in tests

    def add(self, subject: Subject, balls: list[Ball]) -> None:
        self.balls[subject] = balls

    def add_others(self, subject: Subject, counts: dict[Ball, int]) -> None:
        self.everyone_else[subject] = counts

    def add_outs_at_other_end(self, subject: Subject, outs: int) -> None:
        self.other_end[subject] = outs

    def add_records(self, subject: Subject, records: list[Record]) -> None:
        self.his_records[subject] = records

    def add_others_records(self, subject: Subject, records: list[Record]) -> None:
        self.their_records[subject] = records

    def add_pairs(self, subject: Subject, pairs: list[Pair]) -> None:
        self.every_pair[subject] = pairs

    def add_moments(self, subject: Subject, moments: list[Moment]) -> None:
        self.every_moment[subject] = moments

    def deliveries(self, subject: Subject) -> list[Ball]:
        self.asked.append(subject)
        return list(self.balls.get(subject, []))

    def others(self, subject: Subject) -> dict[Ball, int]:
        self.asked.append(subject)
        return dict(self.everyone_else.get(subject, {}))

    def outs_at_other_end(self, subject: Subject) -> int:
        self.asked.append(subject)
        return self.other_end.get(subject, 0)

    def records(self, subject: Subject) -> list[Record]:
        self.asked.append(subject)
        return list(self.his_records.get(subject, []))

    def others_records(self, subject: Subject) -> list[Record]:
        self.asked.append(subject)
        return list(self.their_records.get(subject, []))

    def pairs(self, subject: Subject) -> list[Pair]:
        self.asked.append(subject)
        return list(self.every_pair.get(subject, []))

    def person(self, name: str) -> str | None:
        return name  # the fake knows players by their names

    def moments(self, subject: Subject) -> list[Moment]:
        self.asked.append(subject)
        return list(self.every_moment.get(subject, []))


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

    def innings_at(self, subject: Subject) -> list[InningsScore]:
        """Every innings of the matches the subject picks (a ground, a format, men's or
        women's, a season...), in match order."""

    def wickets_at(self, subject: Subject) -> dict[Wicket, int]:
        """Every wicket that fell in those matches, counted by kind."""


class FakeVenueRepository:
    """Innings and wickets written in by hand for tests, filed under a subject."""

    def __init__(self) -> None:
        self.innings: dict[Subject, list[InningsScore]] = {}
        self.wickets: dict[Subject, dict[Wicket, int]] = {}

    def add_innings(self, subject: Subject, scores: list[InningsScore]) -> None:
        self.innings[subject] = scores

    def add_wickets(self, subject: Subject, counts: dict[Wicket, int]) -> None:
        self.wickets[subject] = counts

    def innings_at(self, subject: Subject) -> list[InningsScore]:
        return list(self.innings.get(subject, []))

    def wickets_at(self, subject: Subject) -> dict[Wicket, int]:
        return dict(self.wickets.get(subject, {}))


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

    def decisions(self, subject: Subject) -> dict[str, int]:
        """How the subject's team's reviews ended: "upheld" -> 81, "struck down" -> 219."""


class FakeReviewRepository:
    """Review counts written in by hand for tests, filed under a subject."""

    def __init__(self) -> None:
        self.teams: dict[Subject, dict[str, int]] = {}

    def add(self, subject: Subject, decisions: dict[str, int]) -> None:
        self.teams[subject] = decisions

    def decisions(self, subject: Subject) -> dict[str, int]:
        return dict(self.teams.get(subject, {}))


class ResultRepository(Protocol):
    """Anything that can tell how matches ended: the database, or a fake (Day 45)."""

    def results(self, subject: Subject) -> list[Result]:
        """How every match the subject picks ended: the team's, against the opposition,
        on the date, in the format... in match order."""


class FakeResultRepository:
    """Results written in by hand for tests, filed under a subject."""

    def __init__(self) -> None:
        self.matches: dict[Subject, list[Result]] = {}

    def add(self, subject: Subject, results: list[Result]) -> None:
        self.matches[subject] = results

    def results(self, subject: Subject) -> list[Result]:
        return list(self.matches.get(subject, []))


# the language model (Day 57, D-024)


class LLMError(ValueError):
    """The language model gave no answer, or one that does not fit its shape."""


class LLM(Protocol):
    """A language model that answers in a fixed shape: the shape's JSON schema goes in
    with the question, and the reply comes back checked against it."""

    def answer[Shape: BaseModel](
        self, system: str, prompt: str, shape: type[Shape]
    ) -> Shape:
        """Raises LLMError when there is no reply, or it does not fit the shape."""


def read_reply[Shape: BaseModel](reply: str, shape: type[Shape]) -> Shape:
    """A model's reply, checked against its shape: broken JSON, a missing field or a
    wrong type is an LLMError, never a half-filled answer."""
    try:
        return shape.model_validate_json(reply)
    except ValidationError as error:
        raise LLMError(f"the reply does not fit {shape.__name__}: {error}") from error


class FakeLLM:
    """Replies written by hand and given out in order, for tests: each is the text a
    model would send, or an LLMError to raise."""

    def __init__(self, replies: list[str | LLMError]) -> None:
        self.replies = list(replies)
        self.asked: list[tuple[str, str, str]] = []  # system, prompt, the shape's name

    def answer[Shape: BaseModel](
        self, system: str, prompt: str, shape: type[Shape]
    ) -> Shape:
        self.asked.append((system, prompt, shape.__name__))
        if not self.replies:
            raise LLMError("the fake has no reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, LLMError):
            raise reply
        return read_reply(reply, shape)  # checked like a real model's
