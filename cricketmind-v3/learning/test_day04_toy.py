from toy.domain import Ball, strike_rate
from toy.fakes import FakeBallRepository
from toy.use_cases import GetStrikeRate

KOHLI_BALLS = [
    Ball("V Kohli", 4),
    Ball("V Kohli", 1),
    Ball("V Kohli", 0, is_wide=True),
    Ball("V Kohli", 6),
]


def test_strike_rate_counts_runs_per_ball():
    assert strike_rate([Ball("V Kohli", 4), Ball("V Kohli", 0)]) == 200.0


def test_wide_is_not_a_ball_faced():
    assert strike_rate(KOHLI_BALLS) == 366.7


def test_no_balls_means_no_strike_rate():
    assert strike_rate([]) is None


def test_use_case_asks_the_repository():
    repo = FakeBallRepository(KOHLI_BALLS)
    use_case = GetStrikeRate(repo)
    assert use_case.run("V Kohli") == 366.7
    assert repo.asked == ["V Kohli"]


def test_unknown_batter_gets_none():
    use_case = GetStrikeRate(FakeBallRepository(KOHLI_BALLS))
    assert use_case.run("RG Sharma") is None
