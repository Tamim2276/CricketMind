import dataclasses

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c3_evidence import check_c3
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import KOHLI_SPIN, c1, c2, compared


def reasons(claim: Claim, template: str) -> list[str]:
    finding = Finding(agent="N11b", claims=(claim,), template=template)
    found = []
    for failure in check_c3(finding, Context()):
        assert failure.check == "C3"
        found.append(failure.reason)
    return found


def against_spin(n: int) -> Claim:
    # well below the average batter against spin, on n balls (synthetic)
    return compared(value=-40.0, interval=(-90.0, -5.0), n=n, subject=KOHLI_SPIN)


# The plan's example: "struggles against spin" on six balls


def test_struggles_against_spin_on_six_balls_fails():
    assert reasons(against_spin(6), "Against spin, he struggles: {c1}.") == [
        (
            "the number rests on only 6 balls (the floor is 30); "
            "say so, e.g. 'on only a few balls'"
        ),
    ]


def test_the_same_with_a_warning_passes():
    # the warning has no digits, so it also passes C5
    template = "Against spin, on only a few balls, he struggles: {c1}."
    assert reasons(against_spin(6), template) == []


def test_the_warning_must_be_in_the_same_sentence():
    template = "This is a small sample. Against spin, he struggles: {c1}."
    assert len(reasons(against_spin(6), template)) == 1


# The floor


def test_a_big_enough_sample_needs_no_warning():
    assert reasons(c1(), "Kohli's cover drive returns {c1}.") == []  # 50 balls


def test_the_floor_itself_is_enough():
    assert reasons(against_spin(30), "He struggles: {c1}.") == []
    assert len(reasons(against_spin(29), "He struggles: {c1}.")) == 1


def test_one_ball_reads_as_one_ball():
    assert reasons(against_spin(1), "{c1}.")[0].startswith(
        "the number rests on only 1 ball (the floor is 30)"
    )


# Other units have their own floors (D-030)


def venue_average(n: int) -> Claim:
    return Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="venue_first_innings_avg",
        subject=CricketSubject(
            venue="Wankhede Stadium", match_format="T20", gender="male"
        ),
        value=167.0,
        n=n,
        source="N8.M2",
    )


def test_a_venue_average_needs_ten_innings():
    assert reasons(venue_average(9), "{c1}.") == [
        (
            "the number rests on only 9 innings (the floor is 10); "
            "say so, e.g. 'on only a few innings'"
        ),
    ]
    assert reasons(venue_average(10), "{c1}.") == []


def test_a_wicket_share_needs_thirty_wickets():
    # Day 40: spin's share of the wickets at a ground
    share = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="venue_wicket_share",
        subject=CricketSubject(
            venue="Shere Bangla National Stadium",
            match_format="T20",
            gender="male",
            bowler_type="spin",
        ),
        value=0.41,
        n=29,
        source="N8.M2",
    )
    assert reasons(share, "{c1}.") == [
        (
            "the number rests on only 29 wickets (the floor is 30); "
            "say so, e.g. 'on only a few wickets'"
        ),
    ]
    assert reasons(dataclasses.replace(share, n=30), "{c1}.") == []


def test_reviews_need_twenty():
    overturns = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="review_overturn_rate",
        subject=CricketSubject(team="India"),
        value=0.3,
        n=19,
        source="N8.M4",
    )
    assert len(reasons(overturns, "{c1}.")) == 1


# What C3 leaves to others


def test_a_prediction_is_not_judged_here():
    assert reasons(c2(), "{c2}.") == []  # its range shows the uncertainty (C4)


def test_a_claim_type_not_on_the_menu_is_left_to_c1():
    hollow = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="pitch_holding_pace",
        subject=CricketSubject(venue="Wankhede Stadium"),
        value=0.8,
        n=3,
    )
    assert reasons(hollow, "{c1}.") == []


def test_one_matchs_margin_needs_no_small_sample_warning():
    # Day 45: a match result is a fact, not a sample
    margin = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="match_won_by_runs",
        subject=CricketSubject(
            team="Pakistan", opposition="Sri Lanka", date="2026-10-11", gender="male"
        ),
        value=82.0,
        n=1,
        source="N8.M5",
    )
    assert reasons(margin, "{c1}.") == []
    share = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="team_win_share",
        subject=CricketSubject(team="Pakistan", gender="male"),
        value=0.6,
        n=5,
        source="N8.M5",
    )
    assert reasons(share, "{c1}.") == [
        (
            "the number rests on only 5 matches (the floor is 10); "
            "say so, e.g. 'on only a few matches'"
        ),
    ]
