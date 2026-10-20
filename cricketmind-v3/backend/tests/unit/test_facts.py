import pytest

from cricketmind_v3.domain.facts import Fact, FactError


def test_a_count_and_a_prediction_are_facts():
    assert Fact(value=142.0, n=50).n == 50
    assert Fact(value=0.405, interval=(0.155, 0.697)).interval == (0.155, 0.697)


def test_an_engine_answer_outside_its_own_range_is_refused():
    with pytest.raises(FactError):
        Fact(value=0.9, interval=(0.155, 0.697))
