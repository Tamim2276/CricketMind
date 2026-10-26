"""Day 43: what a format's name means, and dates in a filter."""

import pytest

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.formats import format_named
from cricketmind_v3.infrastructure.filters import first_day, match_condition


def test_a_t20i_is_an_international_t20():
    t20i = format_named("T20I")
    assert (t20i.match_types, t20i.team_type) == (("T20",), "international")


def test_t20_is_every_twenty_over_match():
    t20 = format_named("T20")
    assert (t20.match_types, t20.team_type) == (("T20", "IT20"), None)


def test_an_unknown_format_is_none():
    assert format_named("Rugby") is None
    with pytest.raises(FilterError):
        match_condition("match_format", "Rugby")


def test_a_format_becomes_sql_with_question_marks():
    sql, values = match_condition("match_format", "T20I")
    assert sql == "(m.match_type IN (?) AND m.team_type = ?)"
    assert values == ["T20", "international"]


def test_before_is_up_to_the_day_before():
    sql, values = match_condition("before", "2025-10-15")
    assert sql == "m.date < ?"
    assert str(values[0]) == "2025-10-15"


def test_since_a_year_or_a_day():
    assert str(first_day("2020")) == "2020-01-01"
    assert str(first_day("2020-06-01")) == "2020-06-01"
    with pytest.raises(FilterError):
        first_day("last year")
