"""Day 41: the M3 engine, on the fake repository (no database)."""

import dataclasses
from datetime import date

from cricketmind_v3.application.engines.m3_workload import workload_ratio_flag
from cricketmind_v3.application.ports import FakeWorkloadRepository, FilterError
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from tests.unit.test_workload import calendar

ASKED = CricketSubject(bowler="A Bowler", date="2025-03-01")


def repo(until=date(2025, 10, 1)) -> FakeWorkloadRepository:
    fake = FakeWorkloadRepository(until)
    fake.add("A Bowler", calendar())
    return fake


class SharedName(FakeWorkloadRepository):
    def spells(self, bowler):
        raise FilterError("2 players are called A Bowler: say which one")


def test_the_engine_reads_the_calendar():
    assert workload_ratio_flag(repo(), ASKED) == Fact(value=2.0, n=240)


def test_a_day_after_the_data_ends_gives_no_answer():
    # the data stops on 28 February: the week up to 1 March would look short
    assert workload_ratio_flag(repo(until=date(2025, 2, 28)), ASKED) is None


def test_the_question_must_be_a_bowler_on_a_date():
    assert workload_ratio_flag(repo(), CricketSubject(bowler="A Bowler")) is None
    not_a_date = dataclasses.replace(ASKED, date="last Tuesday")
    assert workload_ratio_flag(repo(), not_a_date) is None
    with_a_phase = dataclasses.replace(ASKED, phase="death")
    assert workload_ratio_flag(repo(), with_a_phase) is None  # not this measure


def test_a_shared_name_gives_no_answer():
    assert workload_ratio_flag(SharedName(date(2025, 10, 1)), ASKED) is None
