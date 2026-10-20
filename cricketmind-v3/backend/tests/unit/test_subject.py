import dataclasses

import pytest

from cricketmind_v3.domain.subject import CricketSubject, Subject


def test_equal_subjects_compare_equal():
    first = CricketSubject(player="V Kohli", shot="Cover Drive")
    second = CricketSubject(player="V Kohli", shot="Cover Drive")
    assert first == second


def test_kohli_and_rohit_are_different_subjects():
    assert CricketSubject(player="V Kohli") != CricketSubject(player="RG Sharma")


def test_an_empty_field_is_none():
    subject = CricketSubject(player="V Kohli")
    assert subject.shot is None
    assert subject.phase is None
    assert subject.bowler is None
    assert subject.k is None


def test_fields_must_be_named():
    with pytest.raises(TypeError):
        CricketSubject("Pull", "V Kohli")  # would swap shot and player


def test_a_subject_cannot_change():
    subject = CricketSubject(player="V Kohli")
    with pytest.raises(dataclasses.FrozenInstanceError):
        subject.player = "RG Sharma"


def test_a_cricket_subject_is_a_subject():
    # the checks only ever see a Subject, so another sport fits them too
    assert isinstance(CricketSubject(player="V Kohli"), Subject)
