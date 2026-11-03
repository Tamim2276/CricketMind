"""Day 63: who answers: a proposal ranks the agents, and fixed rules decide."""

import pytest

from cricketmind_v3.domain.capabilities import EXPLAINER
from cricketmind_v3.domain.faith.text import tokens
from cricketmind_v3.domain.roster import ROSTER
from cricketmind_v3.domain.routing import (
    BY_KEYWORDS,
    DEFAULT,
    DEFAULT_PAIR,
    KEYWORDS,
    MAX_ACTIVE_AGENTS,
    NOBODY,
    ONLY_THE_EXPLAINER,
    PROPOSED,
    WHOEVER_CAN,
    decide,
    propose_by_keywords,
)

STRIKE_RATE = ("strike_rate",)
# a batter's question, as Day 62 found: eight agents have numbers
BATTER = {
    "N11a": STRIKE_RATE,
    "N11b": STRIKE_RATE,
    "N11c": STRIKE_RATE,
    "N11d": STRIKE_RATE,
    "N11e": STRIKE_RATE,
    "N11h": STRIKE_RATE,
    "N11j": STRIKE_RATE,
    "N11l": STRIKE_RATE,
}
BOWLER_ONLY = {"N11g": ("economy",)}
TEAM = {"N11a": ("team_win_share",), "N11e": ("team_win_share",), "N11i": ("x",)}


# The words


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Should I pick Kohli as my fantasy captain?", ["N11e"]),
        ("What tactics work against Kohli?", ["N11b"]),
        ("Kohli's cover drive or pull?", ["N11c"]),
        ("Is Bumrah overworked?", ["N11k"]),
        ("How often does India overturn a DRS review?", ["N11i"]),
        ("What should the curator know about this pitch?", ["N11f"]),
        ("Is he a good prospect for scouts?", ["N11d"]),
        ("Kohli's strike rate in the T20 powerplay", []),
    ],
)
def test_the_words_point_to_an_agent(question, expected):
    assert propose_by_keywords(question) == expected


@pytest.mark.parametrize(
    "question, not_wanted",
    [
        ("Kohli in the T20 format", "N11c"),  # "form" is inside "format"
        ("Kohli's performance", "N11c"),  # and inside "performance"
        ("the fantastic innings", "N11e"),  # "fan" is inside "fantastic"
        ("the cutter from Bumrah", "N11c"),  # "cut" is inside "cutter"
        ("Did he bowl well in the fielding drill?", "N11b"),  # "field" in "fielding"
    ],
)
def test_only_whole_words_count(question, not_wanted):
    assert not_wanted not in propose_by_keywords(question)


def test_a_phrase_of_two_words_counts():
    assert "N11b" in propose_by_keywords("Should Bumrah bowl to Kohli?")
    assert "N11b" not in propose_by_keywords("How many did Bumrah bowl?")
    # a hyphen is read like a space
    assert propose_by_keywords("Is this a good match-up?") == ["N11b"]


def test_capitals_do_not_matter():
    assert propose_by_keywords("FANTASY TIPS?") == ["N11e"]


def test_the_most_different_words_come_first():
    # one word for the fans, two for the coach
    question = "As a coach, what plan should a fan expect?"
    assert propose_by_keywords(question) == ["N11b", "N11e"]


def test_a_word_said_twice_counts_once():
    # "fan fan" is one word for the fans; a coach's two different words still win
    question = "fan fan fan: what plan would a coach use?"
    assert propose_by_keywords(question) == ["N11b", "N11e"]


def test_a_tie_keeps_the_rosters_order():
    assert propose_by_keywords("What would a fan or a coach say?") == ["N11b", "N11e"]


def test_the_same_question_gives_the_same_answer():
    question = "scouts, fans, coaches, physio, umpire, academy"
    first = propose_by_keywords(question)
    assert first == propose_by_keywords(question)
    assert first == ["N11b", "N11d", "N11e", "N11i", "N11j", "N11k"]


def test_every_stakeholder_agent_has_words_and_the_explainer_has_none():
    assert set(KEYWORDS) == set(ROSTER) - {EXPLAINER}
    for agent, phrases in KEYWORDS.items():
        assert phrases, agent


def test_no_word_points_to_two_agents():
    # one word, one meaning; two would split a question between them for no reason
    seen = {}
    for agent, phrases in KEYWORDS.items():
        for phrase in phrases:
            assert phrase not in seen, (phrase, agent, seen.get(phrase))
            seen[phrase] = agent


def test_no_keyword_is_lost_to_tokens():
    # a phrase with no words in it could never be found
    for phrases in KEYWORDS.values():
        for phrase in phrases:
            assert tokens(phrase), phrase


# The rules


def test_a_proposal_with_numbers_is_chosen():
    routing = decide(["N11e"], BATTER, BY_KEYWORDS)
    assert routing.chosen == ("N11e",)
    assert routing.how == PROPOSED
    assert routing.by == BY_KEYWORDS
    assert routing.proposed == ("N11e",)
    assert routing.no_number == ()


def test_the_proposals_order_is_kept():
    routing = decide(["N11j", "N11a"], BATTER, BY_KEYWORDS)
    assert routing.chosen == ("N11j", "N11a")


def test_at_most_four_answer():
    assert MAX_ACTIVE_AGENTS == 4
    five = ["N11a", "N11b", "N11c", "N11d", "N11e"]
    routing = decide(five, BATTER, BY_KEYWORDS)
    assert routing.chosen == ("N11a", "N11b", "N11c", "N11d")
    assert routing.how == PROPOSED
    assert routing.proposed == tuple(five)  # who was cut can still be read


def test_no_words_means_the_default_pair():
    routing = decide([], BATTER, BY_KEYWORDS)
    assert routing.chosen == DEFAULT_PAIR == ("N11b", "N11c")
    assert routing.how == DEFAULT


def test_the_default_pair_must_have_numbers():
    routing = decide([], {"N11c": ("runs",), "N11e": ("runs",)}, BY_KEYWORDS)
    assert routing.chosen == ("N11c",)
    assert routing.how == DEFAULT


def test_when_the_pair_has_no_numbers_whoever_can_answers():
    routing = decide([], BOWLER_ONLY, BY_KEYWORDS)
    assert routing.chosen == ("N11g",)
    assert routing.how == WHOEVER_CAN
    routing = decide([], TEAM, BY_KEYWORDS)
    assert routing.chosen == ("N11a", "N11e", "N11i")
    assert routing.how == WHOEVER_CAN


def test_the_cap_holds_for_whoever_can_too():
    six = {}
    for agent in ("N11a", "N11d", "N11e", "N11f", "N11g", "N11h"):
        six[agent] = ("x",)
    routing = decide([], six, BY_KEYWORDS)
    assert routing.chosen == ("N11a", "N11d", "N11e", "N11f")
    assert routing.how == WHOEVER_CAN


def test_an_agent_with_no_number_is_never_woken_and_it_is_written_down():
    # "Is Bumrah overworked?" with no date: the medical team has nothing to say (L-078)
    routing = decide(["N11k"], BOWLER_ONLY, BY_KEYWORDS)
    assert routing.chosen == ("N11g",)
    assert routing.how == WHOEVER_CAN
    assert routing.no_number == ("N11k",)


def test_those_with_numbers_stay_when_another_has_none():
    routing = decide(["N11k", "N11e"], BATTER, BY_KEYWORDS)
    assert routing.chosen == ("N11e",)
    assert routing.how == PROPOSED
    assert routing.no_number == ("N11k",)


def test_nobody_capable_wakes_nobody():
    # "Kohli: cover drive or pull?" until shots have labels: never the default pair
    routing = decide(["N11c"], {}, BY_KEYWORDS)
    assert routing.chosen == ()
    assert routing.how == NOBODY
    assert routing.no_number == ("N11c",)


def test_a_rule_goes_to_the_explainer_alone():
    routing = decide([], {EXPLAINER: ()}, BY_KEYWORDS)
    assert routing.chosen == (EXPLAINER,)
    assert routing.how == ONLY_THE_EXPLAINER


def test_a_word_cannot_pull_an_agent_into_a_rules_question():
    # "What is a DRS review?": the words say umpires, the kind says words only
    routing = decide(["N11i"], {EXPLAINER: ()}, BY_KEYWORDS)
    assert routing.chosen == (EXPLAINER,)
    assert routing.no_number == ()  # and no number is missing: a rule wants words


def test_the_explainer_stays_alone():
    routing = decide([EXPLAINER, "N11b"], {EXPLAINER: (), "N11b": ("x",)}, BY_KEYWORDS)
    assert routing.chosen == (EXPLAINER,)


def test_an_agent_named_twice_counts_once():
    routing = decide(["N11e", "N11e", "N11k", "N11k"], BATTER, BY_KEYWORDS)
    assert routing.chosen == ("N11e",)
    assert routing.no_number == ("N11k",)


def test_the_rules_never_choose_an_agent_with_no_number():
    # whoever proposes whatever: every agent, in every order the cap could cut
    every = list(ROSTER)
    for capable in (BATTER, BOWLER_ONLY, TEAM, {}, {EXPLAINER: ()}):
        for proposal in (every, list(reversed(every)), []):
            routing = decide(proposal, capable, BY_KEYWORDS)
            assert len(routing.chosen) <= MAX_ACTIVE_AGENTS
            for agent in routing.chosen:
                assert agent in capable
