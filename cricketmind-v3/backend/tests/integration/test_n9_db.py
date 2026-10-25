"""Days 49-50: N9 T1 on the tiny DuckDB, through the Faith Check and the receipt checker.

Counted by hand (Day 38): Sahibzada Farhan faced 44 balls and was out once. The 32
other men's T20 batters in the fixtures were out 27 times in 435 balls (Day 50).
"""

import dataclasses

import pytest

from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c2_values import check_c2
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.shrinkage import Prior
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from tests.integration.test_duckdb_repo import build
from tests.unit.test_out_within import exact_chances

FARHAN_SIX = CricketSubject(player="Sahibzada Farhan", k=6, match_format="T20")


@pytest.fixture
def engines(tmp_path):
    db = build(tmp_path)
    yield engines_on(db)
    db.close()


def test_his_chance_from_one_out_in_44_balls(engines):
    # the others: 27 outs in 435 balls, and they differ a lot, so normal is worth
    # only 11.6 of his balls: Beta(0.72 + 1, 10.90 + 43)
    found = engines.fact("p_dismissal_within_k", FARHAN_SIX)
    prior = Prior(outs=0.7210, survived=10.8957)
    assert abs(prior.normal() - 27 / 435) < 0.0001
    mid, low, high = exact_chances(1, 44, 6, prior)
    assert found.n == 44
    assert abs(found.value - mid) < 0.01
    assert abs(found.interval[0] - low) < 0.01
    assert abs(found.interval[1] - high) < 0.01


def test_a_shot_has_no_answer_until_the_bridge(engines):
    pull = dataclasses.replace(FARHAN_SIX, shot="Pull")
    assert engines.fact("p_dismissal_within_k", pull) is None


def test_just_come_in(engines):
    # his first five balls: stage 1, where all 32 others have balls
    new = dataclasses.replace(FARHAN_SIX, k=5, faced_so_far=0)
    found = engines.fact("p_dismissal_within_k", new)
    assert found.n == 5  # his own balls 1 to 5
    assert found.receipt.query.count('"faced_so_far":0') == 1


def test_a_stage_too_few_others_reached_has_no_answer(engines):
    # balls 6 onward: only 23 other batters got that far, fewer than 30
    later = dataclasses.replace(FARHAN_SIX, faced_so_far=5)
    assert engines.fact("p_dismissal_within_k", later) is None


def reasons(engines, value, interval) -> list[str]:
    claim = Claim(
        id="c1",
        kind=Provenance.PREDICTIVE,
        metric="p_dismissal_within_k",
        subject=FARHAN_SIX,
        value=value,
        interval=interval,
        source="N9.T1",
    )
    finding = Finding(agent="N11b", claims=(claim,), template="He has {c1}.")
    found = []
    for failure in faith_check(finding, engines, Context(), checks=(check_c2,)):
        found.append(failure.reason)
    return found


def test_the_faith_check_asks_the_engine_again(engines):
    fact = engines.fact("p_dismissal_within_k", FARHAN_SIX)
    assert reasons(engines, fact.value, fact.interval) == []
    assert reasons(engines, 0.30, fact.interval) != []  # a wrong chance
    low, high = fact.interval
    narrow = (low + 0.05, high - 0.05)
    assert reasons(engines, fact.value, narrow) != []  # a range made to look surer


def test_its_receipt_is_confirmed(engines):
    receipt = engines.fact("p_dismissal_within_k", FARHAN_SIX).receipt
    assert check_receipt(receipt, engines) == []
