import pytest

from cricketmind_v3.domain.wording import (
    BANDS,
    NEVER,
    WordingError,
    band_for,
    fits,
    needs_hedge,
    vocabulary,
)


def test_there_are_five_bands_from_none_to_almost_certain():
    names = []
    for band in BANDS:
        names.append(band.name)
    assert names == [
        "almost certainly not",
        "probably not",
        "about even",
        "probable",
        "almost certain",
    ]


# The edges (Table 3 of the v4 document): 0.13 and 0.40 belong to the band below,
# 0.60 and 0.87 to the band above, so the bands mirror each other around 0.5


def test_edge_013():
    assert band_for(0.13).name == "almost certainly not"
    assert band_for(0.1301).name == "probably not"


def test_edge_040():
    assert band_for(0.40).name == "probably not"
    assert band_for(0.4001).name == "about even"


def test_edge_060():
    assert band_for(0.5999).name == "about even"
    assert band_for(0.60).name == "probable"


def test_edge_087():
    assert band_for(0.8699).name == "probable"
    assert band_for(0.87).name == "almost certain"


def test_zero_and_one_are_allowed():
    assert band_for(0.0).name == "almost certainly not"
    assert band_for(1.0).name == "almost certain"


def test_a_chance_outside_0_to_1_is_refused():
    with pytest.raises(WordingError):
        band_for(-0.01)
    with pytest.raises(WordingError):
        band_for(1.01)


def test_the_bands_mirror_each_other():
    assert band_for(0.05) is BANDS[0] and band_for(0.95) is BANDS[4]
    assert band_for(0.2) is BANDS[1] and band_for(0.8) is BANDS[3]


# The document's example: 0.405 with range 0.155 to 0.697


def test_the_documents_chance_is_about_even():
    assert band_for(0.405).name == "about even"


def test_a_range_across_bands_needs_a_hedge():
    assert needs_hedge((0.155, 0.697))  # three bands
    assert not needs_hedge((0.45, 0.55))  # all about even


# Which words fit which band


def test_words_fit_their_own_band_only():
    about_even = band_for(0.405)
    assert fits("as likely as not", about_even)
    assert not fits("likely", about_even)  # forbidden: overstates it
    assert not fits("probably", about_even)  # belongs to "probable"
    assert fits("possible", about_even)  # says nothing about size


def test_never_words_fit_no_band():
    for band in BANDS:
        for phrase in NEVER:
            assert not fits(phrase, band), (phrase, band.name)


def test_no_band_forbids_its_own_words():
    for band in BANDS:
        for phrase in band.words:
            assert phrase not in band.forbidden, (phrase, band.name)


def test_each_phrase_belongs_to_one_band():
    seen = []
    for band in BANDS:
        for phrase in band.words:
            assert phrase not in seen, phrase
            seen.append(phrase)


def test_the_vocabulary_has_every_phrase_once():
    phrases = vocabulary()
    assert len(phrases) == len(set(phrases))
    assert "as likely as not" in phrases
    assert "will" in phrases
