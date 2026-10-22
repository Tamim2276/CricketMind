"""Day 44: the batting and bowling measures, on hand-made balls."""

from cricketmind_v3.domain import measures
from cricketmind_v3.domain.balls import Ball, tally
from cricketmind_v3.domain.facts import Fact


def ball(runs=0, wides=0, noballs=0, out=None, kind=None, batter="V Kohli") -> Ball:
    return Ball(
        match_type="T20",
        gender="male",
        batter=batter,
        runs_batter=runs,
        wides=wides,
        noballs=noballs,
        player_out=out,
        wicket_kind=kind,
    )


def counted(*balls) -> dict[Ball, int]:
    return tally(list(balls))


# Batting


def test_runs_off_the_bat():
    # a wide adds nothing of his, and is no ball faced
    found = measures.runs(counted(ball(4), ball(6), ball(0), ball(1), ball(wides=1)))
    assert found == Fact(value=11.0, n=4)


def test_an_average_counts_a_run_out_at_the_other_end():
    balls = counted(ball(30), ball(0, out="V Kohli", kind="caught"))
    assert measures.batting_average(balls, out_at_other_end=0) == Fact(value=30.0, n=2)
    assert measures.batting_average(balls, out_at_other_end=1) == Fact(value=15.0, n=2)


def test_never_out_has_no_average():
    retired = ball(0, out="V Kohli", kind="retired hurt")
    assert measures.batting_average(counted(ball(50), retired), 0) is None


def test_boundaries_and_dots_are_shares_of_balls_faced():
    balls = counted(
        ball(4),
        ball(6),
        ball(0),  # a dot, even if byes were run
        ball(4, noballs=1),  # a no-ball hit for four: faced, and a boundary
        ball(wides=5),  # a wide that ran to the rope: not his, not faced
    )
    assert measures.boundary_share(balls) == Fact(value=0.75, n=4)
    assert measures.dot_ball_share(balls) == Fact(value=0.25, n=4)


# Bowling


def test_only_the_bowlers_wickets_count():
    balls = counted(
        ball(0, out="V Kohli", kind="caught"),
        ball(0, out="AB de Villiers", kind="run out"),
        ball(0, out="V Kohli", kind="bowled"),
        ball(0),
    )
    assert measures.wickets(balls) == Fact(value=2.0, n=4)


def test_economy_charges_the_bat_wides_and_no_balls():
    balls = counted(
        ball(1),
        ball(4),
        ball(0),
        ball(0),
        ball(0),
        ball(0),
        ball(wides=1),  # one more run, no legal ball
        ball(4, noballs=1),  # five more runs, no legal ball
    )
    assert measures.economy(balls) == Fact(value=11.0, n=6)  # 11 runs in 6 legal balls


def test_a_bowling_strike_rate_needs_a_wicket():
    dots = {ball(0): 22, ball(0, out="V Kohli", kind="lbw"): 2}
    assert measures.bowling_strike_rate(dots) == Fact(value=12.0, n=24)
    assert measures.bowling_strike_rate({ball(0): 24}) is None
