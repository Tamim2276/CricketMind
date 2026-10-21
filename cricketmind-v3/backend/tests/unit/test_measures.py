"""Day 38: the counting rules, and strike rate and dismissal rate from hand-made balls."""

from cricketmind_v3.domain.balls import Ball, batter_out, faced
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.measures import dismissal_rate, strike_rate


def ball(runs=0, wides=0, out=None, kind=None, batter="V Kohli") -> Ball:
    return Ball(
        batter=batter, runs_batter=runs, wides=wides, player_out=out, wicket_kind=kind
    )


def dots(count: int) -> list[Ball]:
    balls = []
    for _ in range(count):
        balls.append(ball())
    return balls


# The rules


def test_a_wide_is_not_a_ball_faced():
    assert not faced(ball(wides=1))
    assert faced(ball(runs=4))  # a no-ball the batter hit is a ball faced


def test_the_striker_out_is_out():
    assert batter_out(ball(out="V Kohli", kind="caught"))
    assert batter_out(ball(out="V Kohli", kind="retired out"))


def test_a_non_striker_run_out_is_not_the_batters():
    assert not batter_out(ball(out="AB de Villiers", kind="run out"))


def test_retiring_hurt_is_not_out():
    assert not batter_out(ball(out="V Kohli", kind="retired hurt"))
    assert not batter_out(ball(out="V Kohli", kind="retired not out"))


# Why 2 balls tell you nothing: one more dot ball


def test_two_balls_move_a_lot():
    two = [ball(runs=6), ball()]
    assert strike_rate(two) == Fact(value=300.0, n=2)
    assert strike_rate([*two, ball()]) == Fact(value=200.0, n=3)  # down by 100


def test_five_hundred_balls_barely_move():
    many = dots(500 - 175)
    for _ in range(175):
        many.append(ball(runs=4))  # 700 runs in 500 balls
    assert strike_rate(many) == Fact(value=140.0, n=500)
    after = strike_rate([*many, ball()])
    assert round(after.value, 1) == 139.7  # down by 0.3
    assert after.n == 501


# Strike rate


def test_wides_add_no_ball_faced():
    balls = [ball(runs=4), ball(wides=1), ball(wides=5), ball(runs=2)]
    assert strike_rate(balls) == Fact(value=300.0, n=2)


def test_no_ball_faced_gives_no_answer():
    assert strike_rate([]) is None
    assert strike_rate([ball(wides=1)]) is None


# Dismissal rate


def test_four_outs_in_fifty_balls():
    balls = dots(46)
    for _ in range(4):
        balls.append(ball(out="V Kohli", kind="bowled"))
    assert dismissal_rate(balls) == Fact(value=0.08, n=50)


def test_only_the_batters_own_outs_count():
    balls = [
        ball(out="V Kohli", kind="caught"),
        ball(out="AB de Villiers", kind="run out"),
        ball(out="V Kohli", kind="retired hurt"),
        ball(),
    ]
    assert dismissal_rate(balls) == Fact(value=0.25, n=4)


def test_stumped_off_a_wide_is_out_but_not_a_ball_faced():
    balls = [ball(), ball(wides=1, out="V Kohli", kind="stumped")]
    assert dismissal_rate(balls) == Fact(value=1.0, n=1)


def test_no_dismissal_rate_without_a_ball_faced():
    assert dismissal_rate([ball(wides=1, out="V Kohli", kind="stumped")]) is None
