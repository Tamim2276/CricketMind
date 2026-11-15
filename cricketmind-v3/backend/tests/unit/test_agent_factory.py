"""Day 66: a model's reply becomes a Finding. Only the words come from the model."""

import json

from cricketmind_v3.application.agent_factory import (
    FindingForm,
    finding_from_reply,
    write_finding,
)
from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.application.ports import FakeFactSource, FakeLLM, LLMError
from cricketmind_v3.domain.agent_spec import (
    agent_prompt,
    fact_line,
    make_spec,
    offered_claims,
)
from cricketmind_v3.domain.facts import Answer, Fact
from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.roster import ROSTER
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.prompt_files import read_rules, read_specs

KOHLI = CricketSubject(player="V Kohli", match_format="T20", phase="powerplay")
AHEAD = CricketSubject(player="V Kohli", match_format="T20", phase="powerplay", k=6)
ROHIT = CricketSubject(player="RG Sharma", match_format="T20")
STRIKE_RATE = Answer(
    metric="strike_rate", subject=KOHLI, fact=Fact(value=130.4, n=2855)
)
AGAINST_AVERAGE = Answer(
    metric="sr_vs_average",
    subject=KOHLI,
    fact=Fact(value=5.0, n=2855, interval=(-1.0, 11.0)),
)
CHANCE = Answer(
    metric="p_dismissal_within_k",
    subject=AHEAD,
    fact=Fact(value=0.18, n=2855, interval=(0.1, 0.45)),
)
ANSWERS = [STRIKE_RATE, AGAINST_AVERAGE, CHANCE]
SPECS = read_specs()
RULES = read_rules()
QUESTION = "Kohli out within 6 balls in the T20 powerplay?"


def reply(template: str) -> str:
    return json.dumps({"template": template})


def written(llm, agent="N11b", failures=(), answers=None) -> object:
    if answers is None:
        answers = ANSWERS
    return write_finding(
        agent, QUESTION, answers, llm, SPECS[agent], RULES, failures=failures
    )


# What the model is offered


def test_the_offer_holds_only_this_agents_measures():
    claims = offered_claims("N11l", ANSWERS)  # Player Representation: strike rate only
    assert len(claims) == 1
    assert claims[0].metric == "strike_rate"


def test_every_claim_is_filled_from_the_engines_answer():
    (rate, compared, chance) = offered_claims("N11b", ANSWERS)
    assert rate.id == "c1"
    assert rate.value == 130.4
    assert rate.n == 2855
    assert rate.subject == KOHLI
    assert rate.kind == Provenance.DESCRIPTIVE
    assert rate.source == "N8.M1"
    assert compared.interval == (-1.0, 11.0)
    assert chance.kind == Provenance.PREDICTIVE
    assert chance.source == "N9.T1"
    assert chance.subject == AHEAD


def test_a_number_nobody_found_is_not_offered():
    nothing = Answer(metric="strike_rate", subject=ROHIT, fact=None)
    assert offered_claims("N11b", [nothing]) == ()


def test_the_slots_do_not_depend_on_which_engine_finished_first():
    forward = offered_claims("N11b", [STRIKE_RATE, AGAINST_AVERAGE, CHANCE])
    backward = offered_claims("N11b", [CHANCE, AGAINST_AVERAGE, STRIKE_RATE])
    assert forward == backward


def test_two_sides_of_a_question_keep_their_order_too():
    rohit = Answer(metric="strike_rate", subject=ROHIT, fact=Fact(value=120.0, n=900))
    one = offered_claims("N11l", [STRIKE_RATE, rohit])
    other = offered_claims("N11l", [rohit, STRIKE_RATE])
    assert one == other
    assert len(one) == 2
    assert one[0].id == "c1"
    assert one[1].id == "c2"


def test_a_count_with_no_sample_is_not_offered():
    # a Claim refuses it, so it can never be said
    hollow = Answer(metric="strike_rate", subject=KOHLI, fact=Fact(value=130.4))
    assert offered_claims("N11b", [hollow]) == ()


def test_a_chance_with_no_range_is_not_offered():
    bare = Answer(metric="p_dismissal_within_k", subject=AHEAD, fact=Fact(value=0.18))
    assert offered_claims("N11b", [bare]) == ()


# What the model is told


def test_the_prompt_gives_each_number_with_its_slot_and_subject():
    (rate, _, _) = offered_claims("N11b", ANSWERS)
    assert fact_line(rate) == (
        "{c1} shows: a strike rate of 130.4 [D, 2855 balls]. "
        "It is about: V Kohli, T20, the powerplay."
    )


def test_a_register_id_is_told_as_the_person():
    answer = Answer(
        metric="strike_rate",
        subject=CricketSubject(player="ab12cd34"),
        fact=Fact(value=100.0, n=500),
    )
    (claim,) = offered_claims("N11l", [answer])
    assert "It is about: Imran Khan (Pakistan)." in fact_line(
        claim, {"ab12cd34": "Imran Khan (Pakistan)"}
    )


def test_a_small_sample_is_flagged():
    small = Answer(metric="strike_rate", subject=KOHLI, fact=Fact(value=99.0, n=10))
    (claim,) = offered_claims("N11l", [small])
    assert "A small sample: say so" in fact_line(claim)


def test_a_big_sample_is_not_flagged():
    (claim,) = offered_claims("N11l", [STRIKE_RATE])
    assert "small sample" not in fact_line(claim)


def test_a_range_across_zero_is_flagged():
    (_, compared, _) = offered_claims("N11b", ANSWERS)
    assert '"no detectable difference"' in fact_line(compared)


def test_a_range_clear_of_zero_is_not_flagged():
    clear = Answer(
        metric="sr_vs_average",
        subject=KOHLI,
        fact=Fact(value=22.0, n=500, interval=(8.0, 36.0)),
    )
    (claim,) = offered_claims("N11d", [clear])
    assert "no detectable difference" not in fact_line(claim)


def test_a_chance_comes_with_the_words_that_fit_it():
    (_, _, chance) = offered_claims("N11b", ANSWERS)
    line = fact_line(chance)
    assert "Words that fit this chance: probably not, unlikely." in line
    assert "the estimate is very uncertain" in line  # 10% to 45% crosses a band


def test_a_narrow_chance_does_not_ask_for_a_hedge():
    narrow = Answer(
        metric="p_dismissal_within_k",
        subject=AHEAD,
        fact=Fact(value=0.3, n=2855, interval=(0.25, 0.35)),
    )
    (claim,) = offered_claims("N11b", [narrow])
    assert "very uncertain" not in fact_line(claim)


def test_a_second_try_is_told_why_the_first_failed():
    failure = Failure(agent="N11b", claim="", check="C5", reason="it typed '150'")
    prompt = agent_prompt(QUESTION, offered_claims("N11b", ANSWERS), (failure,))
    assert "Your last answer did not pass the checks:" in prompt
    assert "- C5: it typed '150'" in prompt


def test_a_first_try_is_told_nothing_of_the_kind():
    prompt = agent_prompt(QUESTION, offered_claims("N11b", ANSWERS))
    assert "did not pass" not in prompt
    assert prompt.startswith(f"Question: {QUESTION}")


def test_the_system_prompt_lists_only_measures_with_numbers():
    llm = FakeLLM([reply("It returns {c1}.")])
    written(llm, answers=[STRIKE_RATE])
    system = llm.asked[0][0]
    assert system.startswith("You are Tactical Analysis")
    assert "- strike_rate:" in system
    assert "- sr_vs_average:" not in system


# The reply


def test_the_form_has_only_the_sentence():
    # no box for a number, a measure or a subject: the model cannot type one
    assert list(FindingForm.model_fields) == ["template"]


def test_a_good_reply_becomes_a_finding():
    finding = written(FakeLLM([reply("The powerplay rate is {c1}.")]))
    assert finding.agent == "N11b"
    assert finding.template == "The powerplay rate is {c1}."
    assert finding.unreadable is None
    assert len(finding.claims) == 1
    assert finding.claims[0].metric == "strike_rate"


def test_only_the_claims_the_sentence_uses_are_kept():
    finding = written(FakeLLM([reply("It is {c3}, and {c1}.")]))
    ids = []
    for claim in finding.claims:
        ids.append(claim.id)
    assert ids == ["c1", "c3"]


def test_a_slot_used_twice_is_one_claim():
    finding = written(FakeLLM([reply("{c1}, again {c1}.")]))
    assert len(finding.claims) == 1


def test_a_reply_that_does_not_fit_the_form_is_unreadable():
    finding = written(FakeLLM(['{"sentence": "wrong box"}']))
    assert finding.claims == ()
    assert "did not fit" in finding.unreadable


def test_an_extra_box_with_a_number_is_refused():
    finding = written(FakeLLM(['{"template": "It is {c1}.", "value": 999}']))
    assert "did not fit" in finding.unreadable


def test_a_model_that_gives_no_answer_is_unreadable():
    finding = written(FakeLLM([LLMError("Ollama could not answer")]))
    assert finding.unreadable == (
        "the model's reply did not fit: Ollama could not answer"
    )


def test_the_factory_asks_once_so_the_bound_stays_simple():
    llm = FakeLLM(['{"sentence": "wrong box"}', reply("It is {c1}.")])
    written(llm)
    assert len(llm.asked) == 1
    assert len(llm.replies) == 1  # the gate decides whether to ask again (Day 74)


def test_a_slot_it_was_not_offered_is_unreadable():
    finding = written(FakeLLM([reply("It is {c9}.")]))
    assert finding.unreadable == "N11b: the sentence uses c9, but no claim has that id"


def test_a_made_up_slot_name_is_unreadable():
    finding = written(FakeLLM([reply("It is {strike_rate}.")]))
    assert "no claim has that id" in finding.unreadable


def test_a_slot_never_closed_is_unreadable():
    finding = written(FakeLLM([reply("It is {c1.")]))
    assert "never closed" in finding.unreadable


def test_a_sentence_with_no_number_is_unreadable():
    # words alone would pass every check, so an agent must say a checked number
    finding = written(FakeLLM([reply("Kohli is a fine player to watch.")]))
    assert "uses no slot" in finding.unreadable


def test_an_empty_sentence_is_unreadable():
    assert "no sentence" in written(FakeLLM([reply("   ")])).unreadable


def test_no_model_is_said_and_nothing_breaks():
    finding = write_finding("N11b", QUESTION, ANSWERS, None, SPECS["N11b"], RULES)
    assert finding.unreadable == "no language model is running"


def test_no_prompt_file_is_said():
    llm = FakeLLM([reply("It is {c1}.")])
    finding = write_finding("N11b", QUESTION, ANSWERS, llm, None, RULES)
    assert finding.unreadable == "this agent has no prompt file"
    assert llm.asked == []


def test_no_number_for_this_agent_is_said_and_the_model_is_not_asked():
    llm = FakeLLM([reply("It is {c1}.")])
    finding = written(llm, agent="N11k")  # the medical agent has nothing here
    assert finding.unreadable == "this agent has no number for this question"
    assert llm.asked == []


def test_a_words_only_finding_needs_no_slot():
    finding = finding_from_reply("N11m", "LBW means leg before wicket.", (), False)
    assert finding.unreadable is None
    assert finding.claims == ()


def test_a_words_only_finding_cannot_use_a_slot():
    finding = finding_from_reply("N11m", "It is {c1}.", (), False)
    assert "no claim has that id" in finding.unreadable


# Through the gate


def engines() -> FakeFactSource:
    source = FakeFactSource()
    source.add("strike_rate", KOHLI, STRIKE_RATE.fact)
    source.add("sr_vs_average", KOHLI, AGAINST_AVERAGE.fact)
    source.add("p_dismissal_within_k", AHEAD, CHANCE.fact)
    return source


GATE = Context(
    allowed=ROSTER, names=cricket_gazetteer({"V Kohli": ("Kohli", "Virat Kohli")})
)


def test_a_scripted_finding_passes_all_seven_checks():
    finding = written(
        FakeLLM(
            [
                reply(
                    "In the T20 powerplay Kohli returns {c1}. It is unlikely he is out "
                    "within six balls, {c3}, though the estimate is very uncertain."
                )
            ]
        )
    )
    assert faith_check(finding, engines(), GATE) == []


def test_an_unreadable_reply_fails_at_c1_with_its_reason():
    finding = written(FakeLLM([LLMError("Ollama could not answer")]))
    (failure,) = faith_check(finding, engines(), GATE)
    assert failure.check == "C1"
    assert failure.agent == "N11b"
    assert "Ollama could not answer" in failure.reason


def test_a_digit_the_model_typed_fails_at_c5():
    finding = written(FakeLLM([reply("Kohli returns {c1}, close to 150.")]))
    (failure,) = faith_check(finding, engines(), GATE)
    assert failure.check == "C5"


def test_a_wrong_name_the_model_typed_fails_at_c6():
    names = cricket_gazetteer(
        {"V Kohli": ("Kohli",), "RG Sharma": ("Rohit", "Rohit Sharma")}
    )
    finding = written(FakeLLM([reply("Rohit returns {c1}.")]))
    context = Context(allowed=ROSTER, names=names)
    (failure,) = faith_check(finding, engines(), context)
    assert failure.check == "C6"


def test_every_voice_builds_a_prompt():
    for agent, spec in SPECS.items():
        if not spec.allowed:
            continue  # the Explainer has its own tests
        llm = FakeLLM([reply("It is {c1}.")])
        answer = Answer(
            metric=spec.allowed[0],
            subject=KOHLI,
            fact=Fact(value=0.3, n=500, interval=(0.2, 0.4)),
        )
        finding = write_finding(agent, QUESTION, [answer], llm, spec, RULES)
        assert finding.unreadable is None, agent
        assert spec.name in llm.asked[0][0]


def test_make_spec_is_what_the_factory_reads():
    spec = make_spec("N11b", "Who you work for: someone.")
    llm = FakeLLM([reply("It is {c1}.")])
    write_finding("N11b", QUESTION, ANSWERS, llm, spec, RULES)
    assert "Who you work for: someone." in llm.asked[0][0]
