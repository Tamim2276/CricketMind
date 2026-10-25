import dataclasses

import pytest

from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import (
    CLAIM_TYPES,
    ClaimType,
    RegistryError,
    check_registry,
    extra_keys,
    index_by_name,
    lookup,
    missing_keys,
)
from cricketmind_v3.domain.render import METRICS
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


def test_every_claim_type_is_on_the_menu():
    # the ten of Day 10, two of the ground's on Day 40, seven measures on Day 44,
    # three results on Day 45
    names = []
    for claim_type in CLAIM_TYPES:
        names.append(claim_type.name)
    assert sorted(names) == [
        "batting_average",
        "boundary_share",
        "bowling_strike_rate",
        "dismissal_rate",
        "dot_ball_share",
        "economy",
        "match_won_by_runs",
        "match_won_by_wickets",
        "matchup_difference",
        "p_dismissal_within_k",
        "review_overturn_rate",
        "runs",
        "sr_difference",
        "sr_vs_average",
        "strike_rate",
        "team_win_share",
        "venue_chasing_win_share",
        "venue_first_innings_avg",
        "venue_wicket_share",
        "weight_transfer",
        "wickets",
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
    missing = missing_keys(lookup("matchup_difference"), subject)
    assert missing == ["bowler", "other_bowler", "phase", "k"]


def test_empty_text_counts_as_missing():
    assert missing_keys(lookup("strike_rate"), CricketSubject(player="")) == ["player"]


def test_review_overturn_rate_needs_a_team_and_men_or_women():
    # Day 42: by team only; "India" is a men's and a women's team
    review = lookup("review_overturn_rate")
    assert missing_keys(review, CricketSubject(team="India", gender="male")) == []
    assert missing_keys(review, CricketSubject(team="India")) == ["gender"]
    lbw = CricketSubject(team="India", gender="male", dismissal_type="lbw")
    assert extra_keys(review, lbw) == ["dismissal_type"]  # not answerable


def test_at_least_one_of_required_any():
    either = ClaimType(
        name="either",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M4",
        required_keys=(),
        required_any=("team", "venue"),
    )
    assert missing_keys(either, CricketSubject(venue="Lord's")) == []
    assert missing_keys(either, CricketSubject()) == ["team or venue"]


def test_a_subject_without_the_field_counts_as_missing():
    # another sport's subject has no "player" box at all
    assert missing_keys(lookup("strike_rate"), Subject()) == ["player"]


def test_the_document_claims_are_on_the_menu_and_complete():
    for claim in (c1(), c2(), c3()):
        claim_type = lookup(claim.metric)
        assert claim_type is not None, claim.metric
        assert claim_type.provenance == claim.kind, claim.id
        assert missing_keys(claim_type, claim.subject) == [], claim.id


# The start-up check (Day 11)


def every_source() -> set[str]:
    sources = set()
    for claim_type in CLAIM_TYPES:
        sources.add(claim_type.source)
    return sources


def every_name() -> set[str]:
    names = set()
    for claim_type in CLAIM_TYPES:
        names.add(claim_type.name)
    return names


GOOD_AGENTS = {"N11b": ["strike_rate", "sr_difference", "p_dismissal_within_k"]}


def test_a_good_setup_has_no_problems():
    assert check_registry(GOOD_AGENTS, every_source(), every_name()) == []


def test_a_claim_type_not_on_the_menu_is_reported():
    agents = {"N11f": ["pitch_holding_pace"]}  # v2's Stadium agent did this (E-002)
    problems = check_registry(agents, every_source(), every_name())
    assert problems == ["N11f: declares 'pitch_holding_pace', which is not on the menu"]


def test_a_missing_engine_is_reported():
    sources = every_source() - {"N8.M4"}
    problems = check_registry(GOOD_AGENTS, sources, every_name())
    assert problems == ["review_overturn_rate: no engine provides N8.M4"]


def test_a_claim_type_with_no_words_is_reported():
    names = every_name() - {"matchup_difference"}
    problems = check_registry(GOOD_AGENTS, every_source(), names)
    assert problems == ["matchup_difference: the renderer has no words for it"]


def test_every_problem_is_reported_not_only_the_first():
    agents = {"N11f": ["pitch_holding_pace"], "N11i": ["sponsor_value"]}
    sources = every_source() - {"N6"}
    problems = check_registry(agents, sources, every_name())
    assert len(problems) == 3


def test_the_real_renderer_has_words_for_every_claim_type():
    assert check_registry(GOOD_AGENTS, every_source(), set(METRICS)) == []


def test_comparisons_and_predictions_need_a_range():
    names = []
    for claim_type in CLAIM_TYPES:
        if claim_type.needs_range:
            names.append(claim_type.name)
    assert sorted(names) == [
        "matchup_difference",
        "p_dismissal_within_k",
        "sr_difference",
        "sr_vs_average",
    ]


def test_a_prediction_without_a_range_is_refused_on_the_menu():
    p_type = lookup("p_dismissal_within_k")
    with pytest.raises(RegistryError):
        dataclasses.replace(p_type, needs_range=False)


# What n counts (Day 20)


def test_most_counts_are_balls():
    assert lookup("strike_rate").unit == "balls"


def test_a_venue_average_counts_innings_and_overturns_count_reviews():
    assert lookup("venue_first_innings_avg").unit == "innings"
    assert lookup("review_overturn_rate").unit == "reviews"


def test_a_wicket_share_counts_wickets():
    assert lookup("venue_wicket_share").unit == "wickets"
    assert lookup("venue_chasing_win_share").unit == "innings"


def test_an_unknown_unit_is_refused():
    with pytest.raises(RegistryError):
        dataclasses.replace(strike_rate_type(), unit="overs")


# Comparisons (Day 21)


def test_the_comparisons_on_the_menu():
    names = []
    for claim_type in CLAIM_TYPES:
        if claim_type.comparison:
            names.append(claim_type.name)
    assert sorted(names) == ["matchup_difference", "sr_difference", "sr_vs_average"]


def test_a_comparison_without_a_range_is_refused_on_the_menu():
    with pytest.raises(RegistryError):
        dataclasses.replace(strike_rate_type(), comparison=True)


# Fields a measure does not ask for (Day 40)


def test_extra_keys_are_the_filled_fields_not_on_the_row():
    venue_type = lookup("venue_first_innings_avg")
    ground = CricketSubject(venue="Lord's", match_format="T20", gender="male")
    assert extra_keys(venue_type, ground) == []
    india = dataclasses.replace(ground, team="India", phase="")
    assert extra_keys(venue_type, india) == ["team"]  # an empty box is not extra


def test_either_of_required_any_is_not_extra():
    either = ClaimType(
        name="either",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M4",
        required_keys=(),
        required_any=("team", "venue"),
    )
    assert extra_keys(either, CricketSubject(team="India")) == []
    assert extra_keys(either, CricketSubject(venue="Lord's")) == []


# Filters (Day 43)


def test_a_batting_measure_takes_the_batting_filters():
    rate = lookup("strike_rate")
    against = CricketSubject(player="V Kohli", opposition="Australia", since="2020")
    assert extra_keys(rate, against) == []
    on_a_date = CricketSubject(player="V Kohli", date="2024-04-12")
    assert extra_keys(rate, on_a_date) == ["date"]  # not a filter of this measure


def test_a_filter_cannot_also_be_required():
    with pytest.raises(RegistryError):
        dataclasses.replace(strike_rate_type(), filters=("player",))


def test_one_matchs_result_is_not_a_sample():
    # Day 45: a margin is a fact about one match; C3's floor does not apply
    assert not lookup("match_won_by_runs").sample
    assert lookup("team_win_share").sample
    assert lookup("team_win_share").unit == "matches"
