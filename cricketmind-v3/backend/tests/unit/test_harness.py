import pytest

from cricketmind_v3.application.harness import (
    TEMPLATE,
    HarnessError,
    _swap_words,
    cases,
    classes_caught,
    controls_passed,
    correct_finding,
    run,
)

RESULTS = run()


def result(name: str):
    for found in RESULTS:
        if found.case.name == name:
            return found
    raise KeyError(name)


# The correct findings pass every gate (no false alarms)


def test_the_controls_pass_every_gate():
    for gate in ("v2 mode", "v3 design", "v4 design"):
        assert controls_passed(RESULTS, gate) == 2, gate


# Each lie is stopped by exactly the checks meant to stop it


def test_each_broken_version_is_caught_by_its_own_checks():
    for found in RESULTS:
        assert found.fired["v4 design"] == found.case.expected, found.case.name


# The table of E-004 (synthetic)


def test_v2_mode_catches_only_the_wrong_value():
    assert classes_caught(RESULTS, "v2 mode") == ["F1"]


def test_the_v3_design_misses_only_the_tone_lie():
    assert classes_caught(RESULTS, "v3 design") == [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
    ]


def test_the_v4_design_catches_all_eight():
    assert classes_caught(RESULTS, "v4 design") == [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
    ]


def test_f2_is_caught_two_ways():
    assert result("F2a").fired["v4 design"] == ("C2",)  # the claim names Rohit
    assert result("F2b").fired["v4 design"] == ("C6",)  # only the sentence does


# The harness itself is honest


def test_every_lie_class_is_present():
    classes = []
    for case in cases():
        if case.expected != () and case.name[:2] not in classes:
            classes.append(case.name[:2])
    assert classes == ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]


def test_every_broken_version_differs_from_the_correct_one():
    good = correct_finding()
    for case in cases():
        if case.expected != ():
            assert case.finding != good, case.name


def test_a_change_that_does_not_apply_is_refused():
    # a typo in a broken version must not silently leave the finding correct
    with pytest.raises(HarnessError):
        _swap_words(correct_finding(), "Kohli's pul drive", "Rohit's pull")


def test_the_template_types_no_numbers():
    text = TEMPLATE
    for slot in ("{c1}", "{c2}", "{c3}", "{c4}"):
        text = text.replace(slot, "")
    for char in text:
        assert not char.isdigit()
