"""Days 38-39: the M1 engine, on the fake repository (no database)."""

import dataclasses

from cricketmind_v3.application.engines.m1_rates import (
    dismissal_rate,
    sr_difference,
    sr_vs_average,
    strike_rate,
)
from cricketmind_v3.application.ports import FakeDeliveryRepository, FilterError
from cricketmind_v3.domain.balls import THE_BATTER, Ball
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import V2_COVER_DRIVE, V2_PULL

KOHLI_POWERPLAY = CricketSubject(player="V Kohli", phase="powerplay")

# the v2 running example (SYNTHETIC): cover drive against pull
COVER_DRIVE = CricketSubject(
    player="V Kohli", shot="Cover Drive", phase="powerplay", bowler_type="pace"
)
PULL = dataclasses.replace(COVER_DRIVE, shot="Pull")
COVER_DRIVE_VS_PULL = dataclasses.replace(COVER_DRIVE, other_shot="Pull")


def ball(runs, out=None, kind=None, batter="V Kohli") -> Ball:
    return Ball(
        match_type="T20",
        gender="male",
        batter=batter,
        runs_batter=runs,
        wides=0,
        player_out=out,
        wicket_kind=kind,
    )


def five_balls() -> list[Ball]:
    # 11 runs off 5 balls, out lbw on the last
    return [ball(4), ball(0), ball(1), ball(6), ball(0, out="V Kohli", kind="lbw")]


class RefusingRepository:
    """A repository that cannot apply the subject's filters."""

    def deliveries(self, subject):
        raise FilterError("cannot pick balls by shot yet")

    def others(self, subject):
        raise FilterError("cannot pick balls by shot yet")


# Day 38: rates


def test_the_engine_asks_for_the_subject_and_counts():
    repo = FakeDeliveryRepository()
    repo.add(KOHLI_POWERPLAY, five_balls())
    assert strike_rate(repo, KOHLI_POWERPLAY) == Fact(value=220.0, n=5)
    assert dismissal_rate(repo, KOHLI_POWERPLAY) == Fact(value=0.2, n=5)
    assert repo.asked == [KOHLI_POWERPLAY, KOHLI_POWERPLAY]  # passed on unchanged


def test_no_player_no_rate():
    # "the strike rate in the powerplay" of everyone is not this measure
    repo = FakeDeliveryRepository()
    assert strike_rate(repo, CricketSubject(phase="powerplay")) is None
    assert dismissal_rate(repo, CricketSubject(player="")) is None
    assert repo.asked == []  # the database was never asked


def test_a_filter_it_cannot_apply_gives_no_answer():
    subject = CricketSubject(player="V Kohli", shot="Cover Drive")
    assert strike_rate(RefusingRepository(), subject) is None
    assert dismissal_rate(RefusingRepository(), subject) is None


def test_no_balls_no_answer():
    assert strike_rate(FakeDeliveryRepository(), KOHLI_POWERPLAY) is None


# Day 39: one shot against another


def test_the_engine_asks_for_each_shot_on_its_own():
    repo = FakeDeliveryRepository()
    repo.add(COVER_DRIVE, V2_COVER_DRIVE)
    repo.add(PULL, V2_PULL)
    found = sr_difference(repo, COVER_DRIVE_VS_PULL)
    assert found.value == 30.0  # 142.0 - 112.0 (N-001)
    assert found.n == 50
    assert repo.asked == [COVER_DRIVE, PULL]


def test_a_difference_needs_both_shots():
    repo = FakeDeliveryRepository()
    repo.add(COVER_DRIVE, V2_COVER_DRIVE)
    assert sr_difference(repo, COVER_DRIVE) is None  # no other shot named
    assert sr_difference(repo, COVER_DRIVE_VS_PULL) is None  # no pull balls
    assert sr_difference(RefusingRepository(), COVER_DRIVE_VS_PULL) is None


# Day 39: a player against the average batter


def test_a_player_against_everyone_else():
    repo = FakeDeliveryRepository()
    repo.add(KOHLI_POWERPLAY, five_balls())  # strike rate 220.0 on 5 balls
    everyone_else = {
        ball(1, batter=THE_BATTER): 60,
        ball(0, batter=THE_BATTER): 40,
    }  # 60 runs off 100 balls: 60.0, on 20 times the balls
    repo.add_others(KOHLI_POWERPLAY, everyone_else)
    found = sr_vs_average(repo, KOHLI_POWERPLAY)
    assert found.value == 160.0
    assert found.n == 5  # the player's side is the thinner one
    low, high = found.interval
    assert low <= 160.0 <= high


def test_no_average_without_enough_balls():
    repo = FakeDeliveryRepository()
    repo.add(KOHLI_POWERPLAY, five_balls())
    repo.add_others(KOHLI_POWERPLAY, {ball(1, batter=THE_BATTER): 99})  # under 20x
    assert sr_vs_average(repo, KOHLI_POWERPLAY) is None
    assert sr_vs_average(RefusingRepository(), KOHLI_POWERPLAY) is None
