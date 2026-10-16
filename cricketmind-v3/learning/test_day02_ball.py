import dataclasses

import pytest
from day02_ball import Ball


def test_equal_balls_compare_equal():
    first = Ball("V Kohli", 4)
    second = Ball("V Kohli", 4)
    assert first == second


def test_different_balls_are_not_equal():
    assert Ball("V Kohli", 4) != Ball("V Kohli", 6)


def test_wicket_defaults_to_false():
    assert Ball("V Kohli", 0).is_wicket is False


def test_frozen_ball_cannot_change():
    ball = Ball("V Kohli", 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ball.runs = 6
