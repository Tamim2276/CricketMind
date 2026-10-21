"""Day 38: the M1 engine, on the fake repository (no database)."""

from cricketmind_v3.application.engines.m1_rates import dismissal_rate, strike_rate
from cricketmind_v3.application.ports import FakeDeliveryRepository, FilterError
from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject

KOHLI_POWERPLAY = CricketSubject(player="V Kohli", phase="powerplay")


def ball(runs, out=None, kind=None) -> Ball:
    return Ball(
        batter="V Kohli", runs_batter=runs, wides=0, player_out=out, wicket_kind=kind
    )


def five_balls() -> list[Ball]:
    # 11 runs off 5 balls, out lbw on the last
    return [ball(4), ball(0), ball(1), ball(6), ball(0, out="V Kohli", kind="lbw")]


class RefusingRepository:
    """A repository that cannot apply the subject's filters."""

    def deliveries(self, subject):
        raise FilterError("cannot pick balls by shot yet")


def test_the_engine_asks_for_the_subject_and_counts():
    repo = FakeDeliveryRepository(five_balls())
    assert strike_rate(repo, KOHLI_POWERPLAY) == Fact(value=220.0, n=5)
    assert dismissal_rate(repo, KOHLI_POWERPLAY) == Fact(value=0.2, n=5)
    assert repo.asked == [KOHLI_POWERPLAY, KOHLI_POWERPLAY]  # passed on unchanged


def test_no_player_no_rate():
    # "the strike rate in the powerplay" of everyone is not this measure
    repo = FakeDeliveryRepository(five_balls())
    assert strike_rate(repo, CricketSubject(phase="powerplay")) is None
    assert dismissal_rate(repo, CricketSubject(player="")) is None
    assert repo.asked == []  # the database was never asked


def test_a_filter_it_cannot_apply_gives_no_answer():
    subject = CricketSubject(player="V Kohli", shot="Cover Drive")
    assert strike_rate(RefusingRepository(), subject) is None
    assert dismissal_rate(RefusingRepository(), subject) is None


def test_no_balls_no_answer():
    assert strike_rate(FakeDeliveryRepository([]), KOHLI_POWERPLAY) is None
