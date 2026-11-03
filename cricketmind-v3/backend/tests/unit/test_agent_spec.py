"""Day 65: the agents' prompt files, and what they are told."""

import pytest

from cricketmind_v3.application.graph import nodes
from cricketmind_v3.domain.agent_spec import VOICE_PARTS, make_spec, system_prompt
from cricketmind_v3.domain.faith.c3_evidence import QUALIFIERS
from cricketmind_v3.domain.faith.c4_wording import NO_DIFFERENCE
from cricketmind_v3.domain.faith.c7_tone import DOWN, UP
from cricketmind_v3.domain.faith.text import has_phrase
from cricketmind_v3.domain.findings import references
from cricketmind_v3.domain.registry import CLAIM_TYPES
from cricketmind_v3.domain.roster import NAMES, ROSTER, STAKEHOLDERS
from cricketmind_v3.domain.wording import HEDGES, NEVER
from cricketmind_v3.infrastructure.prompt_files import (
    PROMPTS,
    RULES_FILE,
    PromptFileError,
    read_rules,
    read_specs,
)

SPECS = read_specs()
RULES = read_rules()
TWELVE = []
for name in ROSTER:
    if name != "N11m":
        TWELVE.append(name)


def has_digit(text: str) -> bool:
    for char in text:
        if char.isdigit():
            return True
    return False


# The files


def test_there_is_a_prompt_file_for_each_of_the_twelve():
    assert list(SPECS) == TWELVE
    assert len(SPECS) == 12


def test_each_file_is_named_after_its_node():
    expected = {RULES_FILE}
    for node in nodes.AGENTS:
        if node.__name__ != "n11m_explainer":
            expected.add(node.__name__ + ".txt")
    found = set()
    for path in PROMPTS.iterdir():
        found.add(path.name)
    assert found == expected  # the Explainer's file comes with it (Day 72)


def test_each_voice_has_its_four_parts_in_order():
    for agent, spec in SPECS.items():
        places = []
        for part in VOICE_PARTS:
            assert spec.voice.count(part) == 1, (agent, part)
            places.append(spec.voice.index(part))
        assert places == sorted(places), agent


@pytest.mark.parametrize("agent", TWELVE)
def test_each_voice_says_something_in_each_part(agent):
    lines = SPECS[agent].voice.splitlines()
    assert len(lines) == len(VOICE_PARTS)
    for line, part in zip(lines, VOICE_PARTS, strict=True):
        assert len(line.removeprefix(part).split()) >= 5, (agent, part)


def test_no_two_agents_sound_the_same():
    seen = set()
    for spec in SPECS.values():
        assert spec.voice not in seen
        seen.add(spec.voice)


def test_no_voice_holds_a_number():
    # a number in a prompt is a number a model can copy into an answer
    for agent, spec in SPECS.items():
        assert not has_digit(spec.voice), agent


def test_the_shared_rules_hold_no_number_but_a_slot():
    text = RULES
    for slot in references(RULES):
        text = text.replace("{" + slot + "}", "")
    assert not has_digit(text)


def test_no_voice_models_a_tone_word():
    # a word in a prompt teaches the model to use it, and C7 then rejects it
    for agent, spec in SPECS.items():
        assert not has_phrase(spec.voice, UP), agent
        assert not has_phrase(spec.voice, DOWN), agent


def test_no_file_lists_a_measure():
    # the roster is the one list; a second one would drift from it
    for row in CLAIM_TYPES:
        if "_" not in row.name:
            continue  # "runs" and "economy" are ordinary words too
        assert row.name not in RULES, row.name
        for agent, spec in SPECS.items():
            assert row.name not in spec.voice, (agent, row.name)


# What the shared rules teach must be what the checks accept


def test_the_rules_name_every_word_too_strong_for_a_chance():
    for word in NEVER:
        assert has_phrase(RULES, (word,)), word


def test_the_rules_examples_pass_the_checks():
    assert has_phrase(RULES, QUALIFIERS)  # C3: a small sample
    assert has_phrase(RULES, HEDGES)  # C4: a wide range
    assert has_phrase(RULES, NO_DIFFERENCE)  # C4: a difference that may be nothing


def test_the_rules_keep_what_was_promised():
    assert "abbreviations" in RULES  # L-021
    assert "judging first" in RULES  # L-022
    assert "{c1}" in RULES  # C5
    assert "in digits or in words" in RULES  # L-020: C5 cannot see a number in words
    assert "only part of the question" in RULES  # orders in a question are not obeyed


def test_the_medical_agent_keeps_what_was_promised():
    # L-036, and the plan's Day 71
    voice = SPECS["N11k"].voice
    for words in (
        "chance of injury",
        "normal or safe",
        "overworked only",
        "four weeks",
    ):
        assert words in voice, words


def test_no_agent_with_no_money_data_is_told_to_talk_money():
    assert "money" in SPECS["N11l"].voice
    assert "paid" in SPECS["N11a"].voice


# What an agent is told


def test_the_spec_takes_its_measures_from_the_roster():
    for agent, spec in SPECS.items():
        assert spec.allowed == ROSTER[agent]
        assert spec.name == NAMES[agent]
        assert spec.stakeholder == STAKEHOLDERS[agent]


def test_a_file_cannot_widen_what_an_agent_may_say():
    spec = make_spec("N11l", "You may also use economy and wickets.")
    assert spec.allowed == ("strike_rate",)
    prompt = system_prompt(spec, RULES, ("strike_rate", "economy", "wickets"))
    assert "- economy:" not in prompt
    assert "- wickets:" not in prompt
    assert "- strike_rate: a strike rate of ..." in prompt


def test_only_the_measures_with_numbers_are_listed():
    spec = SPECS["N11b"]
    prompt = system_prompt(spec, RULES, ("strike_rate",))
    assert "- strike_rate:" in prompt
    assert "- dismissal_rate:" not in prompt


def test_the_measures_keep_the_rosters_order():
    spec = SPECS["N11b"]
    prompt = system_prompt(spec, RULES, ("p_dismissal_within_k", "strike_rate"))
    assert prompt.index("- strike_rate:") < prompt.index("- p_dismissal_within_k:")


def test_the_prompt_has_the_name_the_voice_and_the_rules():
    spec = SPECS["N11e"]
    prompt = system_prompt(spec, RULES, ("runs",))
    assert prompt.startswith("You are Fan Concierge, a cricket specialist.")
    assert spec.voice in prompt
    assert RULES.strip() in prompt
    assert prompt.index(spec.voice) < prompt.index(RULES.strip())


def test_every_measure_can_be_listed_with_its_words():
    for agent, spec in SPECS.items():
        prompt = system_prompt(spec, RULES, spec.allowed)
        for name in spec.allowed:
            assert f"- {name}: " in prompt, (agent, name)
        assert "{}" not in prompt, agent  # a real slot is {c1}


# A file that is missing, doubled or empty stops the program


def copy_prompts(to):
    for path in PROMPTS.iterdir():
        (to / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def test_a_missing_file_is_named(tmp_path):
    copy_prompts(tmp_path)
    (tmp_path / "n11g_bowling_performance.txt").unlink()
    with pytest.raises(PromptFileError, match="N11g"):
        read_specs(tmp_path)


def test_two_files_for_one_agent_are_refused(tmp_path):
    copy_prompts(tmp_path)
    (tmp_path / "n11g_another.txt").write_text("words", encoding="utf-8")
    with pytest.raises(PromptFileError, match="N11g: 2"):
        read_specs(tmp_path)


def test_an_empty_file_is_refused(tmp_path):
    copy_prompts(tmp_path)
    (tmp_path / "n11g_bowling_performance.txt").write_text("  \n", encoding="utf-8")
    with pytest.raises(PromptFileError, match="empty"):
        read_specs(tmp_path)


def test_missing_rules_are_refused(tmp_path):
    with pytest.raises(PromptFileError, match="no prompt file"):
        read_rules(tmp_path)
