from cricketmind_v3.domain.faith.text import (
    claims_in,
    find_phrases,
    has_phrase,
    sentence_of,
    sentences,
    tokens,
)
from cricketmind_v3.domain.findings import Finding
from tests.examples import DOCUMENT_TEMPLATE, c1, c2

# Sentences


def test_the_document_template_has_two_sentences():
    assert sentences(DOCUMENT_TEMPLATE) == [
        "Against pace in the powerplay, Kohli's cover drive returns {c1}.",
        "Six short balls carry {c2} of a dismissal.",
    ]


def test_questions_and_exclamations_end_sentences():
    assert sentences("Is it? Yes! Fine.") == ["Is it?", "Yes!", "Fine."]


def test_a_last_sentence_without_a_full_stop_is_kept():
    assert sentences("One. Two") == ["One.", "Two"]


# Words


def test_words_lose_case_punctuation_and_possessives():
    assert tokens("Kohli's cover-drive!") == ["kohli", "cover", "drive"]


def test_apostrophes_inside_words_go():
    assert tokens("He isn't out") == ["he", "isnt", "out"]
    assert tokens("He isn’t out") == ["he", "isnt", "out"]  # a curly apostrophe


def test_a_slot_reads_as_one_word():
    assert tokens("returns {c1}.") == ["returns", "c1"]


# Phrases


def test_longer_phrases_are_found_first():
    words = tokens("about as likely as not")
    assert find_phrases(words, ["likely", "as likely as not"]) == [
        ("as likely as not", 1),
    ]


def test_a_phrase_can_be_found_twice():
    words = tokens("strong here, strong there")
    assert find_phrases(words, ["strong"]) == [("strong", 0), ("strong", 2)]


def test_phrases_come_back_in_reading_order():
    words = tokens("weak early, strong late")
    assert find_phrases(words, ["strong", "weak"]) == [("weak", 0), ("strong", 2)]


def test_only_whole_words_match():
    assert not has_phrase("He is unlikely to fail", ["likely"])
    assert has_phrase("It is a fifty fifty call", ["fifty-fifty"])


def test_matching_ignores_case():
    assert has_phrase("ON ONLY a few balls", ["on only"])


# Finding a claim's sentence


def test_the_sentence_of_a_claim():
    assert sentence_of(DOCUMENT_TEMPLATE, "c2") == (
        "Six short balls carry {c2} of a dismissal."
    )


def test_a_claim_in_no_sentence():
    assert sentence_of(DOCUMENT_TEMPLATE, "c9") is None


def test_the_claims_in_a_sentence():
    finding = Finding(agent="N11b", claims=(c1(), c2()), template=DOCUMENT_TEMPLATE)
    first = sentences(DOCUMENT_TEMPLATE)[0]
    assert claims_in(first, finding) == [c1()]
