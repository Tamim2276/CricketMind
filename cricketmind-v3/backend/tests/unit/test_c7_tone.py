from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c7_tone import check_c7
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from tests.examples import DOCUMENT_TEMPLATE, c1, c2, compared

ABOVE = compared()  # +22 against the average batter, range +8 to +36 (invented)
BELOW = compared(value=-22.0, interval=(-36.0, -8.0))
UNCLEAR = compared(value=5.0, interval=(-10.0, 20.0))


def reasons(template: str, *claims: Claim) -> list[str]:
    finding = Finding(agent="N11b", claims=tuple(claims), template=template)
    found = []
    for failure in check_c7(finding, Context()):
        assert failure.check == "C7"
        found.append(failure.reason)
    return found


# The plan's four tests (the v4 document's "C7 by example")


def test_struggles_beside_a_number_well_above_average_fails_f8():
    assert reasons("Kohli struggles with the cover drive: {c1}.", ABOVE) == [
        "'struggles' says worse, but the comparison says better: range +8.0 to +36.0",
    ]


def test_strong_beside_it_passes():
    assert reasons("Kohli is strong on the cover drive: {c1}.", ABOVE) == []


def test_strong_with_a_range_that_includes_zero_fails():
    assert reasons("Kohli is strong on the cover drive: {c1}.", UNCLEAR) == [
        (
            "'strong' says better, but the range -10.0 to +20.0 includes zero: "
            "it may be no difference at all"
        ),
    ]


def test_a_tone_word_with_no_comparison_fails():
    # a strike rate alone says nothing about better or worse
    assert reasons("Kohli struggles with the cover drive: {c1}.", c1()) == [
        (
            "'struggles' is a judgment with no comparison behind it; back it with "
            "one, such as sr_vs_average, in the same sentence"
        ),
    ]


# Where the comparison has to be


def test_a_comparison_in_another_sentence_does_not_count():
    assert len(reasons("Kohli is strong here. The numbers: {c1}.", ABOVE)) == 1


def test_a_judgment_with_no_number_at_all_fails():
    template = "Kohli is a strong player. His cover drive returns {c1}."
    assert len(reasons(template, c1())) == 1


def test_one_backing_comparison_is_enough():
    both = (
        compared(claim_id="c1", value=5.0, interval=(-10.0, 20.0)),
        compared(claim_id="c2"),
    )
    assert reasons("Kohli is strong: {c1}, and {c2}.", *both) == []


# Words that turn a judgment around


def test_not_strong_says_worse():
    assert reasons("Kohli is not strong on it: {c1}.", BELOW) == []
    assert reasons("Kohli is not strong on it: {c1}.", ABOVE) == [
        "'strong' says worse, but the comparison says better: range +8.0 to +36.0",
    ]


def test_never_struggles_says_better():
    assert reasons("Kohli never struggles here: {c1}.", ABOVE) == []


# What C7 leaves alone


def test_sentences_without_tone_words_pass():
    assert reasons(DOCUMENT_TEMPLATE, c1(), c2()) == []


def test_a_word_inside_a_longer_phrase_is_read_once():
    # "out of form" is one judgment, not also "form"
    assert reasons("He is out of form: {c1}.", BELOW) == []
