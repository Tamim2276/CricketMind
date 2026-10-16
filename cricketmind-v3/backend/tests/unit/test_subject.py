import dataclasses

import pytest

from cricketmind_v3.domain.subject import Subject


def test_equal_subjects_compare_equal():
    first = Subject(player="V Kohli", shot="Cover Drive")
    second = Subject(player="V Kohli", shot="Cover Drive")
    assert first == second


def test_kohli_and_rohit_are_different_subjects():
    assert Subject(player="V Kohli") != Subject(player="RG Sharma")


def test_an_empty_field_is_none():
    subject = Subject(player="V Kohli")
    assert subject.shot is None
    assert subject.phase is None


def test_fields_must_be_named():
    with pytest.raises(TypeError):
        Subject("Pull", "V Kohli")  # would swap shot and player


def test_a_subject_cannot_change():
    subject = Subject(player="V Kohli")
    with pytest.raises(dataclasses.FrozenInstanceError):
        subject.player = "RG Sharma"
