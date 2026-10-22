"""Day 47: receipts, as text and on facts; and where they come from."""

import pytest

from cricketmind_v3.application.faith_check import recompute
from cricketmind_v3.application.ports import FakeDeliveryRepository
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.receipts import (
    Receipt,
    ReceiptError,
    from_line,
    query_text,
    read_query,
    to_line,
)
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import V2_COVER_DRIVE
from tests.unit.test_engine_source import source

KOHLI = CricketSubject(player="V Kohli", phase="powerplay")


def test_the_same_question_gives_the_same_text():
    again = CricketSubject(phase="powerplay", player="V Kohli", shot="")
    assert query_text("strike_rate", KOHLI, None) == query_text(
        "strike_rate", again, None
    )
    other = CricketSubject(player="V Kohli", phase="death")
    assert query_text("strike_rate", KOHLI, None) != query_text(
        "strike_rate", other, None
    )


def test_the_question_comes_back_from_its_text():
    text = query_text("strike_rate", KOHLI, None)
    assert read_query(text) == ("strike_rate", KOHLI, None)


def test_a_receipt_survives_a_file():
    receipt = Receipt(
        data="2c0b9c7d",
        engine="N8.M1",
        version="1",
        query=query_text("sr_vs_average", KOHLI, None),
        value=8.6,
        n=42341,
        interval=(7.3, 9.8),
    )
    assert from_line(to_line(receipt)) == receipt


def test_an_unreadable_receipt_is_refused():
    with pytest.raises(ReceiptError):
        from_line("not a receipt")
    with pytest.raises(ReceiptError):
        read_query('{"metric": "x", "subject": {"kind": "Rugby", "fields": {}}}')


def test_a_receipt_is_not_part_of_the_number():
    receipt = Receipt(
        data="d", engine="N8.M1", version="1", query="q", value=1.0, n=1, interval=None
    )
    assert Fact(value=1.0, n=1, receipt=receipt) == Fact(value=1.0, n=1)


def test_the_real_source_stamps_every_answer():
    kohli = CricketSubject(player="V Kohli")
    balls = FakeDeliveryRepository()
    balls.add(kohli, V2_COVER_DRIVE)
    engines = source(balls)
    engines.data = "fingerprint"
    fact = engines.fact("strike_rate", kohli)
    assert fact.receipt == Receipt(
        data="fingerprint",
        engine="N8.M1",
        version="1",
        query=query_text("strike_rate", kohli, None),
        value=142.0,
        n=50,
        interval=None,
    )


def test_receipts_come_from_the_gates_own_question_never_the_agent():
    # the agent says 141.96; the receipt keeps what the engine said when asked again
    kohli = CricketSubject(player="V Kohli")
    balls = FakeDeliveryRepository()
    balls.add(kohli, V2_COVER_DRIVE)
    claim = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="strike_rate",
        subject=kohli,
        value=141.96,
        n=50,
        source="N8.M1",
    )
    finding = Finding(agent="N11b", claims=(claim,), template="It was {c1}.")
    receipt = recompute(finding, source(balls))["c1"].receipt
    assert receipt.value == 142.0
