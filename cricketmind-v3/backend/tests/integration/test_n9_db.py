"""Days 49-50: N9 T1 on the tiny DuckDB, through the Faith Check and the receipt checker.

Counted by hand (Day 38): Sahibzada Farhan faced 44 balls and was out once, opening.
The other 11 men who batted 1 to 3 in the fixtures were out 11 times in 208 balls (Day 50).
Too few to fit a prior (that is tested on its own), so here the fit is replaced by a
fixed prior, and the test checks what the engine hands it.
"""

import dataclasses

import pytest

from cricketmind_v3.application.engines import n9_t1
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


FIXED = Prior(outs=2.5, survived=47.5)  # 5%, worth 50 balls
HANDED = []  # what the engine asked the fit for


def fixed_prior(records):
    HANDED.append(list(records))
    if not records:
        return None
    return FIXED


@pytest.fixture
def engines(tmp_path, monkeypatch):
    HANDED.clear()
    monkeypatch.setattr(n9_t1, "population_prior", fixed_prior)
    db = build(tmp_path)
    yield engines_on(db)
    db.close()


def test_with_the_real_floor_there_is_no_answer(tmp_path):
    db = build(tmp_path)
    assert engines_on(db).fact("p_dismissal_within_k", FARHAN_SIX) is None
    db.close()


def test_his_chance_from_one_out_in_44_balls(engines):
    # the engine hands the fit the other 11 top-order men: 11 outs in 208 balls
    found = engines.fact("p_dismissal_within_k", FARHAN_SIX)
    assert len(HANDED[0]) == 11
    outs = 0
    faced = 0
    for out, balls in HANDED[0]:
        outs += out
        faced += balls
    assert (outs, faced) == (11, 208)
    # and his 1 out in 44 is pulled toward the fixed prior: Beta(2.5 + 1, 47.5 + 43)
    mid, low, high = exact_chances(1, 44, 6, FIXED)
    assert found.n == 44
    assert abs(found.value - mid) < 0.01
    assert abs(found.interval[0] - low) < 0.01
    assert abs(found.interval[1] - high) < 0.01


def test_a_shot_has_no_answer_until_the_bridge(engines):
    pull = dataclasses.replace(FARHAN_SIX, shot="Pull")
    assert engines.fact("p_dismissal_within_k", pull) is None


def test_how_long_he_has_been_in_is_not_asked(engines):
    # tried on Day 53 and not kept (D-066): the menu no longer takes it
    new = dataclasses.replace(FARHAN_SIX, k=5, faced_so_far=0)
    assert engines.fact("p_dismissal_within_k", new) is None


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
