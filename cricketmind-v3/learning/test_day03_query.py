import pytest
from day03_query import QueryIn
from pydantic import ValidationError


def test_good_query_is_accepted():
    query = QueryIn(text="Kohli vs pace?", player="V Kohli")
    assert query.player == "V Kohli"


def test_player_is_optional():
    query = QueryIn(text="Who bowls best at the death?")
    assert query.player is None


def test_short_text_is_rejected():
    with pytest.raises(ValidationError) as error:
        QueryIn(text="hi")
    assert error.value.errors()[0]["type"] == "string_too_short"


def test_missing_text_is_rejected():
    with pytest.raises(ValidationError):
        QueryIn(player="V Kohli")


def test_number_is_not_text():
    with pytest.raises(ValidationError):
        QueryIn(text=12345)


def test_from_json():
    query = QueryIn.model_validate_json('{"text": "Kohli vs pace?"}')
    assert query.text == "Kohli vs pace?"
