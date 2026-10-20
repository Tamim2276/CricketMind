from cricketmind_v3.domain.faith.c5_numbers import check_c5
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from tests.examples import DOCUMENT_TEMPLATE, c1, c2


def typed(template: str) -> list[str]:
    # the numbers C5 found typed into the sentence
    claims = (c1(), c2())
    used = []
    for claim in claims:
        if "{" + claim.id + "}" in template:
            used.append(claim)
    finding = Finding(agent="N11b", claims=tuple(used), template=template)
    found = []
    for failure in check_c5(finding, Context()):
        assert failure.check == "C5"
        assert failure.claim == ""  # about the sentence, not one claim
        found.append(failure.reason)
    return found


def test_about_150_typed_into_the_sentence_fails_f3():
    template = "Kohli's cover drive returns {c1}, about 150 in his last match."
    assert typed(template) == [
        "the sentence types '150' itself; every number must come from a claim",
    ]


def test_the_document_template_has_no_typed_numbers():
    assert typed(DOCUMENT_TEMPLATE) == []


def test_slots_are_not_numbers():
    assert typed("{c1} and {c2}.") == []


def test_cricket_names_with_digits_are_allowed():
    assert typed("In T20Is, U-19 and T20 cricket, Kohli returns {c1}.") == []


def test_numbers_glued_to_symbols_are_still_numbers():
    assert len(typed("He returns {c1}, a 45% boundary share.")) == 1
    assert len(typed("He returns {c1}, often 150+.")) == 1
    assert len(typed("He returns {c1} (150).")) == 1


def test_ordinals_are_numbers_too():
    assert typed("In the 2nd innings he returns {c1}.") == [
        "the sentence types '2nd' itself; every number must come from a claim",
    ]


def test_every_typed_number_is_listed():
    assert len(typed("He returns {c1}, between 140 and 160.")) == 2


def test_digits_in_any_script_are_caught():
    assert len(typed("He returns {c1}, about ১৫০.")) == 1  # 150 in Bengali digits


def test_spelled_out_numbers_are_not_caught_a_known_gap():
    # C5 reads digits only; "six" and "fifty" are also cricket words (L-020)
    assert typed("He returns {c1}, about one hundred and fifty.") == []
