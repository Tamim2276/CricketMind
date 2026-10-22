"""Day 46: the Faith Check's C2 against the real engines on the tiny DuckDB.

The engines' answers, counted by hand on Days 38-45: Farhan 73 off 44 (strike rate
165.9); Fakhar Zaman 22 off 17 (129.4); Rawalpindi's men's T20 first innings 215;
Malinga on 2026-10-11 a workload ratio of 4.00 on 26 balls; Australia's women 2 of 3
reviews upheld; Pakistan beat Sri Lanka by 82 runs.
"""

import dataclasses

import pytest

from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c2_values import check_c2
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from tests.integration.test_duckdb_repo import build

FARHAN = CricketSubject(player="Sahibzada Farhan")


@pytest.fixture
def engines(tmp_path):
    db = build(tmp_path)
    yield engines_on(db)
    db.close()


def claim(metric, subject, value, n, source="N8.M1", interval=None) -> Claim:
    return Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric=metric,
        subject=subject,
        value=value,
        n=n,
        interval=interval,
        source=source,
    )


def reasons(engines, the_claim: Claim) -> list[str]:
    finding = Finding(agent="N11b", claims=(the_claim,), template="It was {c1}.")
    found = []
    for failure in faith_check(finding, engines, Context(), checks=(check_c2,)):
        found.append(failure.reason)
    return found


# The plan's test


def test_a_true_number_passes(engines):
    assert reasons(engines, claim("strike_rate", FARHAN, 165.9, 44)) == []


def test_a_wrong_number_fails(engines):
    assert reasons(engines, claim("strike_rate", FARHAN, 170.0, 44)) == [
        "the claim says 170.0, but the engine gives 165.9 for this subject"
    ]


def test_the_right_number_for_the_wrong_player_fails(engines):
    fakhar = CricketSubject(player="Fakhar Zaman")
    assert reasons(engines, claim("strike_rate", fakhar, 165.9, 44)) == [
        "the claim says 165.9, but the engine gives 129.4 for this subject",
        "the claim rests on 44, but the engine counts 17",
    ]


def test_a_bigger_sample_fails(engines):
    assert reasons(engines, claim("strike_rate", FARHAN, 165.9, 60)) == [
        "the claim rests on 60, but the engine counts 44"
    ]


def test_a_number_the_engines_cannot_give_fails_closed(engines):
    nobody = CricketSubject(player="DG Bradman")
    assert reasons(engines, claim("strike_rate", nobody, 99.9, 50)) == [
        "the engine has no number for this subject"
    ]


# One claim from each engine, through the real wiring


def test_every_engine_answers_through_the_faith_check(engines):
    ground = CricketSubject(
        venue="Rawalpindi Cricket Stadium", match_format="T20", gender="male"
    )
    malinga = CricketSubject(bowler="E Malinga", date="2026-10-11")
    australia = CricketSubject(team="Australia", gender="female")
    match = CricketSubject(
        team="Pakistan", opposition="Sri Lanka", date="2026-10-11", gender="male"
    )
    true_claims = [
        claim("runs", FARHAN, 73.0, 44),
        claim("venue_first_innings_avg", ground, 215.0, 1, source="N8.M2"),
        claim("workload_ratio_flag", malinga, 4.0, 26, source="N8.M3"),
        claim("review_overturn_rate", australia, 2 / 3, 3, source="N8.M4"),
        claim("match_won_by_runs", match, 82.0, 1, source="N8.M5"),
    ]
    for true_claim in true_claims:
        assert reasons(engines, true_claim) == [], true_claim.metric
        wrong = dataclasses.replace(true_claim, value=true_claim.value + 1)
        assert reasons(engines, wrong) != [], true_claim.metric
