"""Days 49-50: N9 T1 on the tiny DuckDB, through the Faith Check and the receipt checker.

Counted by hand (Day 38): Sahibzada Farhan faced 44 balls and was out once, opening.
The other 11 men who batted 1 to 3 in the fixtures were out 11 times in 208 balls (Day 50).
"""

import dataclasses

import pytest

from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.domain import shrinkage
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
def engines(tmp_path, monkeypatch):
    # three matches hold only 11 other top-order men, fewer than the 30 a prior needs:
    # here the floor is 5, so the engine has an answer to check (the real floor below)
    monkeypatch.setattr(shrinkage, "FEW_BATTERS", 5)
    db = build(tmp_path)
    yield engines_on(db)
    db.close()


def test_with_the_real_floor_there_is_no_answer(tmp_path):
    db = build(tmp_path)
    assert engines_on(db).fact("p_dismissal_within_k", FARHAN_SIX) is None
    db.close()


def test_his_chance_from_one_out_in_44_balls(engines):
    # the other top-order men: 11 outs in 208 balls, worth 43.1 of his balls:
    # Beta(2.28 + 1, 40.85 + 43)
    found = engines.fact("p_dismissal_within_k", FARHAN_SIX)
    prior = Prior(outs=2.2812, survived=40.8539)
    assert abs(prior.normal() - 11 / 208) < 0.0001
    mid, low, high = exact_chances(1, 44, 6, prior)
    assert found.n == 44
    assert abs(found.value - mid) < 0.01
    assert abs(found.interval[0] - low) < 0.01
    assert abs(found.interval[1] - high) < 0.01


def test_a_shot_has_no_answer_until_the_bridge(engines):
    pull = dataclasses.replace(FARHAN_SIX, shot="Pull")
    assert engines.fact("p_dismissal_within_k", pull) is None


def test_just_come_in(engines):
    # his first five balls: stage 1, where all 11 other top-order men have balls
    new = dataclasses.replace(FARHAN_SIX, k=5, faced_so_far=0)
    found = engines.fact("p_dismissal_within_k", new)
    assert found.n == 5  # his own balls 1 to 5
    assert found.receipt.query.count('"faced_so_far":0') == 1


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
