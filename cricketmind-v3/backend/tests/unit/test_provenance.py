import pytest

from cricketmind_v3.domain.provenance import Provenance


def test_there_are_exactly_three_kinds():
    names = []
    for kind in Provenance:
        names.append(kind.name)
    assert names == ["MEASURED", "DESCRIPTIVE", "PREDICTIVE"]


def test_text_from_json_becomes_a_kind():
    assert Provenance("descriptive") is Provenance.DESCRIPTIVE


def test_unknown_provenance_is_impossible():
    with pytest.raises(ValueError):
        Provenance("guessed")


def test_a_typo_fails_at_once():
    with pytest.raises(AttributeError):
        kind = Provenance.DESCRIPTVE  # typo: missing "I"
        assert kind is None  # never reached: the line above crashes
