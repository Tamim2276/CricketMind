import dataclasses

import pytest

from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import (
    CLAIM_TYPES,
    ClaimType,
    RegistryError,
    index_by_name,
    lookup,
    missing_keys,
)
from cricketmind_v3.domain.subject import CricketSubject, Subject
from tests.examples import c1, c2, c3


def strike_rate_type() -> ClaimType:
    return ClaimType(
        name="strike_rate",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M1",
        required_keys=("player",),
    )


def test_a_claim_type_holds_its_row():
    claim_type = strike_rate_type()
    assert claim_type.name == "strike_rate"
    assert claim_type.provenance == Provenance.DESCRIPTIVE
    assert claim_type.source == "N8.M1"
    assert claim_type.required_keys == ("player",)
    assert claim_type.required_any == ()


def test_a_claim_type_cannot_change():
    claim_type = strike_rate_type()
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim_type.source = "N9.T1"


def test_keys_written_without_the_comma_are_refused():
    with pytest.raises(RegistryError):
        dataclasses.replace(strike_rate_type(), required_keys=("player"))


def test_a_list_of_keys_is_refused():
    with pytest.raises(RegistryError):
        dataclasses.replace(strike_rate_type(), required_any=["team"])


def test_all_ten_claim_types_are_on_the_menu():
    names = []
    for claim_type in CLAIM_TYPES:
        names.append(claim_type.name)
    assert sorted(names) == [
        "dismissal_rate",
        "matchup_difference",
        "p_dismissal_within_k",
        "review_overturn_rate",
        "sr_difference",
        "sr_vs_average",
        "strike_rate",
        "venue_first_innings_avg",
        "weight_transfer",
        "workload_ratio_flag",
    ]


def test_each_kind_comes_from_its_engine():
    # N6 measures, N8 counts, N9 simulates
    expected = {
        "N6": Provenance.MEASURED,
        "N8": Provenance.DESCRIPTIVE,
        "N9": Provenance.PREDICTIVE,
    }
    for claim_type in CLAIM_TYPES:
        engine = claim_type.source[:2]
        assert claim_type.provenance == expected[engine], claim_type.name


def test_every_needed_field_exists_on_a_cricket_subject():
    # a typo like "playr" would make a claim look incomplete forever
    fields = []
    for field in dataclasses.fields(CricketSubject):
        fields.append(field.name)
    for claim_type in CLAIM_TYPES:
        for key in claim_type.required_keys + claim_type.required_any:
            assert key in fields, f"{claim_type.name}: no field '{key}'"


def test_lookup_finds_a_row():
    claim_type = lookup("strike_rate")
    assert claim_type is not None
    assert claim_type.source == "N8.M1"


def test_lookup_finds_every_row():
    for claim_type in CLAIM_TYPES:
        assert lookup(claim_type.name) is claim_type


def test_a_name_not_on_the_menu_gives_none():
    assert lookup("pitch_holding_pace") is None
    assert lookup("Strike_Rate") is None  # names are exact


def test_a_name_listed_twice_is_refused():
    with pytest.raises(RegistryError):
        index_by_name((strike_rate_type(), strike_rate_type()))


def test_nothing_missing_when_the_player_is_given():
    subject = CricketSubject(player="V Kohli")
    assert missing_keys(lookup("strike_rate"), subject) == []


def test_a_missing_player_is_listed():
    assert missing_keys(lookup("strike_rate"), CricketSubject()) == ["player"]


def test_every_missing_field_is_listed_in_order():
    subject = CricketSubject(player="V Kohli")
    missing = missing_keys(lookup("p_dismissal_within_k"), subject)
    assert missing == ["shot", "phase", "bowler_type", "k"]


def test_empty_text_counts_as_missing():
    assert missing_keys(lookup("strike_rate"), CricketSubject(player="")) == ["player"]


def test_review_overturn_rate_needs_a_team_or_a_dismissal_type():
    review = lookup("review_overturn_rate")
    assert missing_keys(review, CricketSubject(team="India")) == []
    assert missing_keys(review, CricketSubject(dismissal_type="lbw")) == []
    assert missing_keys(review, CricketSubject()) == ["team or dismissal_type"]


def test_a_subject_without_the_field_counts_as_missing():
    # another sport's subject has no "player" box at all
    assert missing_keys(lookup("strike_rate"), Subject()) == ["player"]


def test_the_document_claims_are_on_the_menu_and_complete():
    for claim in (c1(), c2(), c3()):
        claim_type = lookup(claim.metric)
        assert claim_type is not None, claim.metric
        assert claim_type.provenance == claim.kind, claim.id
        assert missing_keys(claim_type, claim.subject) == [], claim.id
