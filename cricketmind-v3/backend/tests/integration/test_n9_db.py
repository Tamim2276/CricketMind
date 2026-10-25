"""Day 49: N9 T1 on the tiny DuckDB, through the Faith Check and the receipt checker.

Counted by hand (Day 38): Sahibzada Farhan faced 44 balls and was out once.
"""

import pytest

from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c2_values import check_c2
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from tests.integration.test_duckdb_repo import build
from tests.unit.test_out_within import exact_chances

FARHAN_SIX = CricketSubject(player="Sahibzada Farhan", k=6)


@pytest.fixture
def engines(tmp_path):
    db = build(tmp_path)
    yield engines_on(db)
    db.close()


def test_his_chance_from_one_out_in_44_balls(engines):
    found = engines.fact("p_dismissal_within_k", FARHAN_SIX)
    mid, low, high = exact_chances(1, 44, 6)
    assert found.n == 44
    assert abs(found.value - mid) < 0.01
    assert abs(found.interval[0] - low) < 0.01
    assert abs(found.interval[1] - high) < 0.01


def test_a_shot_has_no_answer_until_the_bridge(engines):
    pull = CricketSubject(player="Sahibzada Farhan", shot="Pull", k=6)
    assert engines.fact("p_dismissal_within_k", pull) is None


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
