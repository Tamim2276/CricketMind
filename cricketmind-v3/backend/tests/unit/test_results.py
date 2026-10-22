"""Day 45: wins and margins, on hand-made results."""

from cricketmind_v3.application.engines.m5_results import (
    match_won_by_runs,
    team_win_share,
)
from cricketmind_v3.application.ports import FakeResultRepository
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.results import Result, win_share, won_by_runs, won_by_wickets
from cricketmind_v3.domain.subject import CricketSubject


def result(winner, by_runs=None, by_wickets=None, by_innings=False) -> Result:
    return Result(
        teams=("Bangladesh", "India"),
        winner=winner,
        by_runs=by_runs,
        by_wickets=by_wickets,
        by_innings=by_innings,
    )


def test_a_win_share_leaves_out_ties_draws_and_no_results():
    results = [
        result("Bangladesh", by_runs=10),
        result("India", by_wickets=5),
        result("India", by_runs=2),
        result(None),  # a tie, a draw or no result
    ]
    assert win_share(results, "Bangladesh") == Fact(value=1 / 3, n=3)
    assert win_share(results, "India") == Fact(value=2 / 3, n=3)
    assert win_share(results, "Pakistan") is None  # not in these matches


def test_a_margin_names_its_winner():
    one = [result("Bangladesh", by_runs=10)]
    assert won_by_runs(one, "Bangladesh") == Fact(value=10.0, n=1)
    assert won_by_runs(one, "India") is None  # India did not win it
    assert won_by_wickets(one, "Bangladesh") is None  # not won by wickets


def test_an_innings_win_is_not_a_win_by_runs():
    assert won_by_runs([result("India", by_runs=20, by_innings=True)], "India") is None


def test_two_matches_on_one_day_give_no_answer():
    two = [result("India", by_runs=1), result("India", by_runs=9)]
    assert won_by_runs(two, "India") is None


def test_the_engine_needs_the_whole_match():
    repo = FakeResultRepository()
    match = CricketSubject(
        team="Bangladesh", opposition="India", date="2024-10-12", gender="male"
    )
    repo.add(match, [result("Bangladesh", by_runs=10)])
    assert match_won_by_runs(repo, match) == Fact(value=10.0, n=1)
    no_date = CricketSubject(team="Bangladesh", opposition="India", gender="male")
    assert match_won_by_runs(repo, no_date) is None
    every = CricketSubject(team="Bangladesh", gender="male")
    repo.add(every, [result("Bangladesh", by_runs=10), result("India", by_wickets=3)])
    assert team_win_share(repo, every) == Fact(value=0.5, n=2)
