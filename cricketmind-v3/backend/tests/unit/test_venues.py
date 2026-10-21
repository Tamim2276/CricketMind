"""Day 40: what M2 counts about a ground, on hand-made innings and wickets."""

from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.venues import (
    InningsScore,
    Wicket,
    chasing_win_share,
    complete,
    first_innings_average,
    standard,
    wicket_share,
)


def score(
    runs=150,
    innings=1,
    wickets=4,
    legal=120,
    won=None,
    match_type="T20",
    scheduled=120,
    declared=False,
) -> InningsScore:
    return InningsScore(
        match_type=match_type,
        scheduled_balls=scheduled,
        innings=innings,
        runs=runs,
        wickets=wickets,
        legal_balls=legal,
        declared=declared,
        won=won,
    )


def wicket(kind="caught", bowler_type="pace", match_type="T20", scheduled=120):
    return Wicket(
        match_type=match_type,
        scheduled_balls=scheduled,
        bowler_type=bowler_type,
        kind=kind,
    )


# Which matches and innings count


def test_a_standard_match_has_the_formats_usual_length():
    assert standard("T20", 120)
    assert standard("ODI", 300)
    assert not standard("T20", 100)  # The Hundred
    assert not standard("T20", 300)  # a "T20" recorded with 50 overs
    assert standard("Test", None)


def test_a_full_innings_ran_its_course():
    assert complete(score(legal=120, wickets=4))  # every ball batted
    assert complete(score(legal=90, wickets=10))  # all out early
    assert not complete(score(legal=60, wickets=5))  # cut short


def test_a_test_innings_is_full_when_all_out_or_declared():
    test = {"match_type": "Test", "scheduled": None, "legal": 600}
    assert complete(score(wickets=10, **test))
    assert complete(score(wickets=6, declared=True, **test))
    assert not complete(score(wickets=3, **test))  # the match ran out of time


# First-innings average


def test_only_full_first_innings_count():
    scores = [
        score(runs=150),  # full 20 overs
        score(runs=180, wickets=10, legal=100),  # all out
        score(runs=90, wickets=3, legal=60),  # rain: left out
        score(runs=151, innings=2, won=True),  # a second innings: left out
    ]
    assert first_innings_average(scores) == Fact(value=165.0, n=2)


def test_the_hundred_is_left_out():
    hundred = score(runs=140, legal=100, scheduled=100)
    assert first_innings_average([hundred]) is None
    assert first_innings_average([hundred, score(runs=160)]) == Fact(value=160.0, n=1)


# Chasing


def test_the_share_of_results_won_chasing():
    scores = [
        score(innings=2, won=True),
        score(innings=2, won=True),
        score(innings=2, won=False),
        score(innings=2, won=None),  # a tie
        score(innings=1, won=False),  # the side that batted first
    ]
    assert chasing_win_share(scores) == Fact(value=2 / 3, n=3)


def test_no_chasing_in_a_test():
    test = score(innings=2, won=True, match_type="Test", scheduled=None)
    assert chasing_win_share([test]) is None


# Pace and spin


def test_spins_share_of_the_bowlers_wickets():
    wickets = {
        wicket("caught", "pace"): 6,
        wicket("bowled", "spin"): 3,
        wicket("lbw", "spin"): 1,
        wicket("run out", "pace"): 5,  # not the bowler's
        wicket("retired hurt", "spin"): 1,  # not a dismissal
    }
    assert wicket_share(wickets, "spin") == Fact(value=0.4, n=10)
    assert wicket_share(wickets, "pace") == Fact(value=0.6, n=10)


def test_too_many_unknown_bowlers_gives_no_answer():
    # 8 known of 11: under 80%
    wickets = {wicket("caught", "pace"): 4, wicket("caught", "spin"): 4}
    wickets[wicket("caught", None)] = 3
    assert wicket_share(wickets, "spin") is None
    wickets[wicket("caught", None)] = 2  # 8 known of 10: enough
    assert wicket_share(wickets, "spin") == Fact(value=0.5, n=8)


def test_a_type_the_data_does_not_know_gives_no_answer():
    assert wicket_share({wicket("caught", "pace"): 40}, "medium") is None


def test_wickets_in_the_hundred_are_left_out():
    wickets = {
        wicket("caught", "spin", scheduled=100): 50,
        wicket("caught", "pace"): 10,
    }
    assert wicket_share(wickets, "spin") == Fact(value=0.0, n=10)
