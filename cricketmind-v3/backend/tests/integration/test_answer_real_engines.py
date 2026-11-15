"""Day 77: the whole graph on the real engines and the tiny DuckDB; only the model is fake.

The numbers were counted by hand on Days 38-45: Farhan 73 off 44 (strike rate 165.9);
Fakhar Zaman 22 off 17 (129.4). What this adds: the numbers an agent is offered are the
ones the gate gets when it asks again, to the last digit, and each has a receipt that
anyone can run.
"""

import pytest

from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.ports import FakeLLMByWho
from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.names import Player
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.prompt_files import read_rules, read_specs
from tests.integration.test_duckdb_repo import build
from tests.world import READER, REPORTER, ROUTER, reading, routed, words

FARHAN = Player(person_id="aaaa1111", name="Sahibzada Farhan", matches=100)
FAKHAR = Player(person_id="bbbb2222", name="Fakhar Zaman", matches=100)
NAMES = cricket_gazetteer(
    {"Sahibzada Farhan": ("Farhan",), "Fakhar Zaman": ("Fakhar",)}
)


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


def run(db, replies: dict, question: str = "Farhan's strike rate?") -> dict:
    llm = FakeLLMByWho(
        {
            READER: [reading(player="Farhan", match_format=None)],
            ROUTER: [routed("N11b", "N11c")],
            REPORTER: [],
            **replies,
        }
    )
    context = GraphContext(
        names=NAMES,
        llm=llm,
        players=(FARHAN, FAKHAR),
        deliveries=DuckDBDeliveryRepository(db.cursor()),
        engines=engines_on(db.cursor()),
        specs=read_specs(),
        rules=read_rules(),
    )
    return build_graph().invoke({"question": question, "clip": None}, context=context)


def test_a_real_number_reaches_the_reader_through_every_check(db):
    state = run(
        db,
        {
            "Tactical Analysis": [words("Farhan returns {c1}.")],
            "Personal Performance": [words("Farhan returns {c1}.")],
        },
    )
    assert state["withheld"] == ()
    assert state["checks_run"] == 1
    assert "a strike rate of 165.9 [D, 44 balls]" in state["answer"]


def test_the_claim_is_about_exactly_what_the_engine_was_asked(db):
    state = run(db, {"Tactical Analysis": [words("Farhan returns {c1}.")]})
    verified = {}
    for finding in state["verified"]:
        verified[finding.agent] = finding
    (claim,) = verified["N11b"].claims
    assert claim.subject == CricketSubject(player="Sahibzada Farhan")
    assert claim.source == "N8.M1"
    assert claim.n == 44


def test_every_receipt_shown_can_be_made_again(db):
    state = run(db, {"Tactical Analysis": [words("Farhan returns {c1}.")]})
    engines = engines_on(db.cursor())
    assert state["receipts"] != {}
    for receipt in state["receipts"].values():
        assert check_receipt(receipt, engines) == []
    assert "Receipts:" in state["answer"]
    assert "- N8.M1 version 1" in state["answer"]


def test_a_wrong_player_in_the_words_is_stopped_on_real_numbers(db):
    state = run(db, {"Tactical Analysis": [words("Fakhar returns {c1}.")] * 3})
    assert "Withheld: Tactical Analysis: its words did not pass C6." in state["answer"]
    assert "Fakhar" not in state["answer"].split("Withheld")[0]


def test_agents_side_by_side_share_the_one_connection_safely(db):
    # Day 61's bug: two threads on one connection swapped rows, with no error
    for _ in range(5):
        state = run(
            db,
            {
                "Tactical Analysis": [words("Farhan returns {c1}.")],
                "Personal Performance": [words("Farhan returns {c1}.")],
            },
        )
        assert state["withheld"] == ()
