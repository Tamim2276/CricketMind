import dataclasses

import pytest

from cricketmind_v3.domain.faith.gate import Context, Failure, run_checks
from cricketmind_v3.domain.findings import Finding
from tests.examples import DOCUMENT_TEMPLATE, c1, c2


def document_finding() -> Finding:
    return Finding(agent="N11b", claims=(c1(), c2()), template=DOCUMENT_TEMPLATE)


# two dummy checks: one always passes, one always fails


def always_passes(finding, context):
    return []


def always_fails(finding, context):
    return [Failure(agent=finding.agent, claim="c1", check="D1", reason="dummy")]


def test_a_check_with_nothing_to_say_gives_no_failures():
    assert run_checks(document_finding(), Context(), (always_passes,)) == []


def test_a_failure_is_collected():
    failures = run_checks(document_finding(), Context(), (always_passes, always_fails))
    assert failures == [
        Failure(agent="N11b", claim="c1", check="D1", reason="dummy"),
    ]


def test_every_check_runs_even_after_a_failure():
    def also_fails(finding, context):
        return [Failure(agent=finding.agent, claim="c2", check="D2", reason="dummy")]

    failures = run_checks(document_finding(), Context(), (always_fails, also_fails))
    checks = []
    for failure in failures:
        checks.append(failure.check)
    assert checks == ["D1", "D2"]  # the agent hears both at once, in order


def test_checks_run_in_the_order_given():
    calls = []

    def first(finding, context):
        calls.append("first")
        return []

    def second(finding, context):
        calls.append("second")
        return []

    run_checks(document_finding(), Context(), (first, second))
    assert calls == ["first", "second"]


def test_each_check_sees_the_finding_and_the_context():
    seen = []

    def spy(finding, context):
        seen.append((finding, context))
        return []

    finding = document_finding()
    context = Context()
    run_checks(finding, context, (spy,))
    assert seen[0][0] is finding
    assert seen[0][1] is context


def test_a_failure_cannot_change():
    failure = Failure(agent="N11b", claim="c1", check="C2", reason="wrong value")
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.reason = "fine after all"


# Found on review day (Day 27): fail closed


def test_a_check_that_breaks_holds_the_finding_back():
    def check_c9(finding, context):
        raise KeyError("nothing here")

    failures = run_checks(document_finding(), Context(), (check_c9, always_fails))
    assert failures[0] == Failure(
        agent="N11b",
        claim="",
        check="C9",
        reason="the check could not run (KeyError: 'nothing here')",
    )
    assert failures[1].check == "D1"  # the checks after it still run
