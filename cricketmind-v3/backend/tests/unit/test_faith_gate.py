"""Days 73 and 74: only the chosen agents run; the gate checks them, and a bounded loop
sends back only those that failed. The model and the engines are fakes (tests/world.py)."""

import dataclasses

from langgraph.runtime import Runtime

from cricketmind_v3.application.faith_check import check_with_receipts
from cricketmind_v3.application.graph import nodes, routes
from cricketmind_v3.application.ports import FakeFactSource, LLMError
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.gate import (
    MAX_REGENERATIONS,
    Context,
    Failure,
    withheld_from,
)
from cricketmind_v3.domain.faith.gazetteer import gate_gazetteer
from cricketmind_v3.domain.findings import Finding, latest_findings
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.roster import NAMES, ROSTER
from tests.world import (
    KOHLI,
    QUESTION,
    RATE,
    ask,
    asked_of,
    context_of,
    engines,
    model,
    words,
)
from tests.world import (
    NAMES as PLAYER_NAMES,
)

HONEST = "Kohli returns {c1}."
LIE = "Kohli returns {c1}, close to 150."  # C5: a typed number
TACTICAL = NAMES["N11b"]
PERSONAL = NAMES["N11c"]


def honest_agents() -> dict[str, list]:
    return {TACTICAL: [words(HONEST)], PERSONAL: [words(HONEST)]}


def runs(state: dict, node: str) -> int:
    return state["visited"].count(node)


# Day 73: who runs


def test_only_the_chosen_agents_run():
    state = ask(model(honest_agents()))
    for node in nodes.AGENTS:
        expected = 0
        if node.__name__ in ("n11b_tactical_analysis", "n11c_personal_performance"):
            expected = 1
        assert runs(state, node.__name__) == expected, node.__name__


def test_their_findings_are_collected_side_by_side():
    state = ask(model(honest_agents()))
    agents = set()
    for finding in state["findings"]:
        agents.add(finding.agent)
    assert agents == {"N11b", "N11c"}


def test_the_gate_waits_for_them_all_and_runs_once():
    state = ask(model(honest_agents()))
    assert runs(state, "faith_gate") == 1
    last = max(
        state["visited"].index("n11b_tactical_analysis"),
        state["visited"].index("n11c_personal_performance"),
    )
    assert state["visited"].index("faith_gate") > last


def test_one_chosen_agent_is_enough():
    state = ask(model({TACTICAL: [words(HONEST)]}, chosen=("N11b",)))
    assert runs(state, "n11b_tactical_analysis") == 1
    assert runs(state, "n11c_personal_performance") == 0


def test_with_nobody_chosen_no_agent_runs():
    # the engines have nothing: nobody can answer
    state = ask(model({}), source=FakeFactSource())
    assert state["routing"].chosen == ()
    for node in nodes.AGENTS:
        assert runs(state, node.__name__) == 0
    assert runs(state, "faith_gate") == 0
    assert runs(state, "n12_report_synthesis") == 0
    assert runs(state, "n13_final_output") == 1


def test_four_agents_at_most_whatever_the_model_asks():
    agents = {}
    everyone = []
    for agent in ROSTER:
        if ROSTER[agent] and "strike_rate" in ROSTER[agent]:
            everyone.append(agent)
            agents[NAMES[agent]] = [words(HONEST)]
    assert len(everyone) > 4
    state = ask(model(agents, chosen=tuple(everyone), report=[]))
    ran = []
    for node in nodes.AGENTS:
        if runs(state, node.__name__):
            ran.append(node.__name__)
    assert len(ran) == 4


def test_to_the_chosen_names_their_nodes():
    state = ask(model(honest_agents()))
    assert routes.to_the_chosen(state) == [
        "n11b_tactical_analysis",
        "n11c_personal_performance",
    ]


# Day 74: the plan's three tests


def test_lie_once_is_fixed():
    agents = {TACTICAL: [words(LIE), words(HONEST)], PERSONAL: [words(HONEST)]}
    state = ask(model(agents, report=[words(HONEST)]))
    assert state["withheld"] == ()
    assert len(state["verified"]) == 2
    assert state["failures"] == ()
    assert state["checks_run"] == 2


def test_lie_always_is_withheld():
    agents = {TACTICAL: [words(LIE)] * 5, PERSONAL: [words(HONEST)]}
    state = ask(model(agents))
    (withheld,) = state["withheld"]
    assert withheld.agent == "N11b"
    assert withheld.checks == ("C5",)
    verified = []
    for finding in state["verified"]:
        verified.append(finding.agent)
    assert verified == ["N11c"]


def test_the_loop_never_runs_a_third_time():
    llm = model({TACTICAL: [words(LIE)] * 5, PERSONAL: [words(HONEST)]})
    state = ask(llm)
    assert runs(state, "n11b_tactical_analysis") == 1 + MAX_REGENERATIONS
    assert len(asked_of(llm, TACTICAL)) == 3
    assert state["checks_run"] == 3
    assert runs(state, "faith_gate") == 3
    assert runs(state, "n12_report_synthesis") == 1


# Day 74: only the failing go back, and they hear why


def test_only_the_failing_agent_goes_back():
    agents = {TACTICAL: [words(LIE), words(HONEST)], PERSONAL: [words(HONEST)]}
    state = ask(model(agents, report=[words(HONEST)]))
    assert runs(state, "n11b_tactical_analysis") == 2
    assert runs(state, "n11c_personal_performance") == 1


def test_the_failing_agent_hears_the_reason():
    llm = model({TACTICAL: [words(LIE), words(HONEST)], PERSONAL: [words(HONEST)]})
    ask(llm)
    (first, second) = asked_of(llm, TACTICAL)
    assert "did not pass" not in first[1]
    assert "Your last answer did not pass the checks:" in second[1]
    assert "the sentence types '150' itself" in second[1]


def test_it_hears_only_its_own_reasons():
    llm = model(
        {
            TACTICAL: [words(LIE), words(HONEST)],
            PERSONAL: [words("Kohli is strong: {c1}."), words(HONEST)],
        }
    )
    ask(llm)
    (_, second_tactical) = asked_of(llm, TACTICAL)
    (_, second_personal) = asked_of(llm, PERSONAL)
    assert "strong" not in second_tactical[1]
    assert "150" not in second_personal[1]


def test_a_lie_fixed_on_the_last_try_still_passes():
    agents = {TACTICAL: [words(LIE), words(LIE), words(HONEST)]}
    state = ask(model(agents, chosen=("N11b",)))
    assert state["withheld"] == ()
    assert len(state["verified"]) == 1


def test_every_try_is_kept_and_the_last_counts():
    agents = {TACTICAL: [words(LIE), words(HONEST)]}
    state = ask(model(agents, chosen=("N11b",)))
    assert len(state["findings"]) == 2
    assert state["verified"][0].template == HONEST


def test_an_agent_that_passed_is_not_checked_again():
    # a source that counts each question: the passing agent's number is asked once
    source = _Counting(engines())
    agents = {TACTICAL: [words(LIE), words(HONEST)], PERSONAL: [words(HONEST)]}
    ask(model(agents, report=[words(HONEST)]), source=source)
    # N8 once; the gate: N11b's lie, N11c, then N11b's honest try
    assert source.asked == 4


def test_a_reply_that_cannot_be_read_goes_back_too():
    agents = {TACTICAL: ['{"wrong": 1}', words(HONEST)]}
    state = ask(model(agents, chosen=("N11b",)))
    assert runs(state, "n11b_tactical_analysis") == 2
    assert state["withheld"] == ()


def test_a_model_that_is_down_ends_in_nothing_shown():
    down = [LLMError("Ollama could not answer")] * 5
    state = ask(model({TACTICAL: list(down), PERSONAL: list(down)}))
    assert len(state["withheld"]) == 2
    for withheld in state["withheld"]:
        assert withheld.checks == ("C1",)
    assert state["verified"] == ()
    assert "Nothing is shown" in state["answer"]


def test_no_model_ends_the_same_way_and_never_crashes():
    state = ask(None)
    assert state["verified"] == ()
    assert "Nothing is shown" in state["answer"]


# The gate itself, given a lie no model can tell: a number it did not get from the engines


def lying_claim(value: float = 999.0) -> Claim:
    return Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="strike_rate",
        subject=KOHLI,
        value=value,
        n=RATE.n,
        source="N8.M1",
    )


def gate_state(*findings: Finding) -> dict:
    return {"question": QUESTION, "findings": list(findings), "visited": []}


def gate(state: dict, source=None) -> dict:
    context = context_of(None, source)
    return nodes.faith_gate(state, Runtime(context=context))


def test_a_wrong_number_is_caught_and_its_receipt_is_the_engines():
    rogue = Finding(agent="N11b", claims=(lying_claim(),), template="It is {c1}.")
    out = gate(gate_state(rogue))
    assert out["verified"] == ()
    (failure,) = out["failures"]
    assert failure.check == "C2"
    assert out["receipts"][("N11b", "c1")].value == 135.1


def test_a_true_number_passes():
    honest = Finding(
        agent="N11b", claims=(lying_claim(135.1),), template="Kohli returns {c1}."
    )
    out = gate(gate_state(honest))
    assert out["failures"] == ()
    assert out["verified"] == (honest,)


def test_the_last_try_is_the_one_checked():
    rogue = Finding(agent="N11b", claims=(lying_claim(),), template="It is {c1}.")
    honest = Finding(
        agent="N11b", claims=(lying_claim(135.1),), template="Kohli returns {c1}."
    )
    assert gate(gate_state(rogue, honest))["verified"] == (honest,)
    assert gate(gate_state(honest, rogue))["verified"] == ()


def test_the_budget_is_counted_by_the_gate():
    rogue = Finding(agent="N11b", claims=(lying_claim(),), template="It is {c1}.")
    state = gate_state(rogue)
    assert gate(state)["withheld"] == ()
    state["checks_run"] = 1
    assert gate(state)["withheld"] == ()
    state["checks_run"] = 2
    (withheld,) = gate(state)["withheld"]
    assert withheld.agent == "N11b"


def test_with_no_engines_every_number_fails_closed():
    honest = Finding(
        agent="N11b", claims=(lying_claim(135.1),), template="Kohli returns {c1}."
    )
    failures, receipts = check_with_receipts(
        honest, None, Context(allowed=ROSTER, names=PLAYER_NAMES)
    )
    assert receipts == {}
    assert failures[0].check == "C2"
    assert "never re-checked" in failures[0].reason


def test_after_the_gate_the_failing_go_back_while_the_budget_lasts():
    failure = Failure(agent="N11b", claim="", check="C5", reason="a digit")
    state = {"failures": (failure, failure), "checks_run": 1}
    assert routes.after_gate(state) == ["n11b_tactical_analysis"]
    state["checks_run"] = 2
    assert routes.after_gate(state) == ["n11b_tactical_analysis"]
    state["checks_run"] = 3
    assert routes.after_gate(state) == ["n12_report_synthesis"]


def test_after_the_gate_with_no_failure_the_report_goes_on():
    state = {"failures": (), "checks_run": 1}
    assert routes.after_gate(state) == ["n12_report_synthesis"]


def test_a_failure_of_no_agent_sends_nobody_back():
    failure = Failure(agent="N3", claim="", check="C2", reason="N3 is no agent node")
    state = {"failures": (failure,), "checks_run": 1}
    assert routes.after_gate(state) == ["n12_report_synthesis"]


def test_latest_findings_keeps_each_agents_last():
    first = Finding(agent="N11b", claims=(), template="first")
    other = Finding(agent="N11c", claims=(), template="other")
    last = Finding(agent="N11b", claims=(), template="last")
    assert latest_findings([first, other, last]) == [last, other]


def test_withheld_lists_each_agent_once_with_every_check_and_reason():
    failures = [
        Failure(agent="N11b", claim="c1", check="C5", reason="one"),
        Failure(agent="N11b", claim="c1", check="C5", reason="two"),
        Failure(agent="N11c", claim="c1", check="C7", reason="three"),
        Failure(agent="N11b", claim="", check="C7", reason="four"),
    ]
    (first, second) = withheld_from(failures)
    assert first.agent == "N11b"
    assert first.checks == ("C5", "C7")
    assert first.reasons == ("one", "two", "four")
    assert second.agent == "N11c"


class _Counting:
    """A fact source that counts every question it is asked."""

    def __init__(self, source: FakeFactSource) -> None:
        self.source = source
        self.asked = 0

    def fact(self, metric, subject, clip_id=None) -> Fact | None:
        if metric == "strike_rate":
            self.asked += 1
        return self.source.fact(metric, subject, clip_id)


def test_the_gate_reads_the_gates_names_when_it_has_them():
    # N3's rules keep their list; the gate knows the teams too
    names = gate_gazetteer({"V Kohli": ("Kohli",)}, ("India",), ())
    context = dataclasses.replace(context_of(None), gate_names=names)
    finding = Finding(
        agent="N11b",
        claims=(lying_claim(135.1),),
        template="For India, Kohli returns {c1}.",
    )
    written = nodes.faith_gate(gate_state(finding), Runtime(context=context))
    (failure,) = written["failures"]
    assert failure.check == "C6"


def test_without_them_the_gate_reads_the_readers_names():
    finding = Finding(
        agent="N11b",
        claims=(lying_claim(135.1),),
        template="For India, Kohli returns {c1}.",
    )
    assert gate(gate_state(finding))["failures"] == ()  # India is not on this list
