"""Day 42: the overturn rate, and the M4 engine on the fake."""

from cricketmind_v3.application.engines.m4_reviews import review_overturn_rate
from cricketmind_v3.application.ports import FakeReviewRepository
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.reviews import overturn_rate
from cricketmind_v3.domain.subject import CricketSubject

INDIA = CricketSubject(team="India", gender="male")


def test_the_share_of_reviews_that_changed_the_decision():
    assert overturn_rate({"upheld": 81, "struck down": 219}) == Fact(value=0.27, n=300)


def test_a_technical_failure_decided_nothing():
    counts = {"upheld": 1, "struck down": 3, "technical failure": 2}
    assert overturn_rate(counts) == Fact(value=0.25, n=4)
    assert overturn_rate({"technical failure": 2}) is None


def test_no_reviews_no_answer():
    assert overturn_rate({}) is None


def test_the_engine_asks_for_the_team_men_or_women():
    repo = FakeReviewRepository()
    repo.add("India", "male", {"upheld": 3, "struck down": 9})
    repo.add("India", "female", {"upheld": 6, "struck down": 6})
    assert review_overturn_rate(repo, INDIA) == Fact(value=0.25, n=12)
    women = CricketSubject(team="India", gender="female")
    assert review_overturn_rate(repo, women) == Fact(value=0.5, n=12)


def test_by_dismissal_type_there_is_no_answer():
    # the data keeps no appeal kind, so "India's lbw reviews" cannot be counted fairly
    repo = FakeReviewRepository()
    repo.add("India", "male", {"upheld": 3, "struck down": 9})
    lbw = CricketSubject(team="India", gender="male", dismissal_type="lbw")
    assert review_overturn_rate(repo, lbw) is None
    assert review_overturn_rate(repo, CricketSubject(team="India")) is None
