import dataclasses

import pytest

from cricketmind_v3.domain.findings import Finding, FindingError, references
from tests.examples import DOCUMENT_TEMPLATE, c1, c2


def test_slots_are_found_in_order():
    assert references(DOCUMENT_TEMPLATE) == ["c1", "c2"]


def test_a_sentence_with_no_slots_has_none():
    assert references("Kohli plays the cover drive well.") == []


def test_the_same_slot_twice_is_listed_twice():
    assert references("{c1}, and again {c1}.") == ["c1", "c1"]


def test_an_unclosed_slot_is_refused():
    with pytest.raises(FindingError):
        references("Kohli's cover drive returns {c1.")


# The Finding itself


def test_the_documents_finding_is_accepted():
    finding = Finding(agent="N11b", claims=(c1(), c2()), template=DOCUMENT_TEMPLATE)
    assert len(finding.claims) == 2


def test_a_slot_with_no_claim_is_refused():
    with pytest.raises(FindingError):
        Finding(agent="N11b", claims=(c1(),), template="Kohli returns {c1}, then {c3}.")


def test_two_claims_with_the_same_id_are_refused():
    with pytest.raises(FindingError):
        Finding(agent="N11b", claims=(c1(), c1()), template="Kohli returns {c1}.")


def test_claims_must_be_a_tuple():
    with pytest.raises(FindingError):
        Finding(agent="N11b", claims=[c1()], template="Kohli returns {c1}.")


def test_an_unclosed_slot_in_a_finding_is_refused():
    with pytest.raises(FindingError):
        Finding(agent="N11b", claims=(c1(),), template="Kohli returns {c1.")


def test_a_finding_cannot_change():
    finding = Finding(agent="N11b", claims=(c1(),), template="Kohli returns {c1}.")
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding.template = "Kohli returns 162.0."
