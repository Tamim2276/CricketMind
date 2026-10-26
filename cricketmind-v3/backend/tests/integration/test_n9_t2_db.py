"""Days 51-52: N9 T2 on the tiny DuckDB, through the Faith Check and the receipt checker.

Counted by hand from 1550709.json: Sahibzada Farhan's 44 balls, bowler by bowler: B
Fernando 10, E Malinga 7, M Theekshana 10, MD Shanaka 8 (out once), N Thushara 7, PHKD
Mendis 2. The fixtures are too small to fit a prior (that is tested on its own), so here
the fit is a fixed prior.
"""

import pytest

from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.domain import matchups
from cricketmind_v3.domain.balls import tally
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c2_values import check_c2
from cricketmind_v3.domain.faith.c4_wording import check_c4
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.measures import outs_and_faced
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.shrinkage import Prior
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from tests.integration.test_duckdb_repo import build

FARHAN = "Sahibzada Farhan"
MEN = CricketSubject(match_format="T20", gender="male")
WHAT_IF = CricketSubject(
    player=FARHAN,
    bowler="MD Shanaka",
    other_bowler="B Fernando",
    k=6,
    match_format="T20",
)
FIXED = Prior(outs=2.5, survived=47.5)  # 5%, worth 50 balls


def fixed_prior(records):
    if not records:
        return None
    return FIXED


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


@pytest.fixture
def repo(db):
    return DuckDBDeliveryRepository(db)


@pytest.fixture
def engines(db, monkeypatch):
    monkeypatch.setattr(matchups, "population_prior", fixed_prior)
    return engines_on(db)


# The pairs


def test_his_balls_bowler_by_bowler(repo):
    farhan = repo.person(FARHAN)
    found = {}
    for pair in repo.pairs(MEN):
        if pair.batter == farhan:
            found[pair.bowler] = (pair.order, pair.outs, pair.faced)
    expected = {}
    for name, outs, faced in (
        ("B Fernando", 0, 10),
        ("E Malinga", 0, 7),
        ("M Theekshana", 0, 10),
        ("MD Shanaka", 1, 8),
        ("N Thushara", 0, 7),
        ("PHKD Mendis", 0, 2),
    ):
        expected[repo.person(name)] = ("top", outs, faced)
    assert found == expected


def test_duckdb_counts_as_python_does_for_batters_and_bowlers(repo):
    pairs = repo.pairs(CricketSubject())
    by_batter = {}
    by_bowler = {}
    for pair in pairs:
        outs, faced = by_batter.get(pair.batter, (0, 0))
        by_batter[pair.batter] = (outs + pair.outs, faced + pair.faced)
        outs, faced = by_bowler.get(pair.bowler, (0, 0))
        by_bowler[pair.bowler] = (outs + pair.outs, faced + pair.faced)
    checked = 0
    for column, totals, field in (
        ("batter", by_batter, "player"),
        ("bowler", by_bowler, "bowler"),
    ):
        names = repo.db.execute(f"SELECT DISTINCT {column} FROM deliveries").fetchall()
        for (name,) in names:
            subject = CricketSubject(**{field: name})
            counted = outs_and_faced(tally(repo.deliveries(subject)))
            if counted[1] == 0:
                continue  # a batter who never faced a ball, only ran
            assert totals[repo.person(name)] == counted, name
            checked += 1
    assert checked > 70


def test_the_pairs_come_in_the_same_order_every_time(repo):
    # a receipt is checked to the last digit, so even the order of a sum must not move
    found = repo.pairs(MEN)
    keys = []
    for pair in found:
        keys.append((pair.batter, pair.bowler, pair.order))
    assert keys == sorted(keys)


# The engine, end to end


def test_the_what_if_has_an_answer_with_both_sides(engines):
    found = engines.fact("matchup_difference", WHAT_IF)
    low, high = found.interval
    assert low <= found.value <= high
    # the thinnest record it rests on: Farhan's 44 balls, Fernando's 20 to top-order
    # men, Shanaka's 17 (8 to Farhan, 9 to Maaz Sadaqat; counted by hand from the file)
    assert found.n == 17
    assert len(found.replays.overs) == 100
    assert len(found.replays.versus.overs) == 100


def claim_of(value, interval) -> Claim:
    return Claim(
        id="c1",
        kind=Provenance.PREDICTIVE,
        metric="matchup_difference",
        subject=WHAT_IF,
        value=value,
        interval=interval,
        source="N9.T2",
    )


def reasons(engines, claim, template, checks) -> list[str]:
    finding = Finding(agent="N11b", claims=(claim,), template=template)
    found = []
    for failure in faith_check(finding, engines, Context(), checks=checks):
        found.append(failure.reason)
    return found


def test_the_faith_check_asks_the_engine_again(engines):
    fact = engines.fact("matchup_difference", WHAT_IF)
    said = "Shanaka instead of Fernando: {c1}."
    true = claim_of(fact.value, fact.interval)
    assert reasons(engines, true, said, (check_c2,)) == []
    wrong = claim_of(fact.value + 0.1, (fact.interval[0], fact.interval[1] + 0.1))
    assert reasons(engines, wrong, said, (check_c2,)) != []
    low, high = fact.interval
    narrow = claim_of(fact.value, (low + 0.02, high - 0.02))
    assert reasons(engines, narrow, said, (check_c2,)) != []  # surer than it is


def test_a_range_across_zero_must_say_no_detectable_difference(engines):
    fact = engines.fact("matchup_difference", WHAT_IF)
    low, high = fact.interval
    assert low < 0 < high  # 18 balls cannot separate two bowlers
    claim = claim_of(fact.value, fact.interval)
    honest = "Shanaka instead of Fernando: no detectable difference, {c1}."
    assert reasons(engines, claim, honest, (check_c4,)) == []
    verdict = "Shanaka is the better option: {c1}."
    assert reasons(engines, claim, verdict, (check_c4,)) != []


def test_its_receipt_is_confirmed(engines):
    receipt = engines.fact("matchup_difference", WHAT_IF).receipt
    assert receipt.engine == "N9.T2"
    assert check_receipt(receipt, engines) == []
