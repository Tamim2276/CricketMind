"""Days 68-72: each of the twelve agents and the Explainer, with scripted words.

Every number is made up and comes through a fake engine; every sentence is one a model
could write. Each must pass all seven real checks, and each lie must be caught."""

import json

import pytest

from cricketmind_v3.application.agent_factory import write_finding
from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.application.graph import build as graph_build
from cricketmind_v3.application.ports import FakeFactSource, FakeLLM
from cricketmind_v3.domain.agent_spec import offered_claims
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Answer, Fact
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import RegistryError, lookup
from cricketmind_v3.domain.render import render
from cricketmind_v3.domain.roster import ROSTER
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.prompt_files import read_rules, read_specs

SPECS = read_specs()
RULES = read_rules()
PLAYERS = {
    "V Kohli": ("Kohli", "Virat Kohli"),
    "JJ Bumrah": ("Bumrah",),
    "MA Starc": ("Starc",),
    "JT Ball": ("Ball",),  # the register has them: everyday words are names too
    "E Chance": ("Chance",),
    "YS To": ("To",),
}
GATE = Context(allowed=ROSTER, names=cricket_gazetteer(PLAYERS))

KOHLI = CricketSubject(player="V Kohli", match_format="T20")
AHEAD = CricketSubject(player="V Kohli", match_format="T20", k=6)
PULL = CricketSubject(player="V Kohli", shot="Cover Drive", other_shot="Pull")
BUMRAH = CricketSubject(bowler="JJ Bumrah", match_format="T20")
MATCHUP = CricketSubject(
    player="V Kohli",
    bowler="JJ Bumrah",
    other_bowler="MA Starc",
    k=12,
    match_format="T20",
)
GROUND = CricketSubject(
    venue="Wankhede Stadium", match_format="T20", gender="male", bowler_type="spin"
)
WHOLE_GROUND = CricketSubject(
    venue="Wankhede Stadium", match_format="T20", gender="male"
)
INDIA = CricketSubject(team="India", gender="male")
FINAL = CricketSubject(
    team="Australia", opposition="India", date="2023-11-19", gender="male"
)
WORKLOAD = CricketSubject(bowler="JJ Bumrah", date="2025-01-04")


def answer(metric, subject, value, n=None, interval=None) -> Answer:
    return Answer(
        metric=metric, subject=subject, fact=Fact(value=value, n=n, interval=interval)
    )


RATE = answer("strike_rate", KOHLI, 135.1, 9841)
OUT_RATE = answer("dismissal_rate", KOHLI, 0.042, 9841)
ABOVE = answer("sr_vs_average", KOHLI, 12.0, 9841, (8.0, 16.0))
LEVEL = answer("sr_vs_average", KOHLI, 1.0, 9841, (-2.0, 4.0))
DIFFERENCE = answer("sr_difference", PULL, 30.0, 100, (-16.0, 76.0))
RUNS = answer("runs", KOHLI, 13295.0, 9841)
AVERAGE = answer("batting_average", KOHLI, 41.8, 9841)
BOUNDARIES = answer("boundary_share", KOHLI, 0.18, 9841)
DOTS = answer("dot_ball_share", KOHLI, 0.31, 9841)
CHANCE = answer("p_dismissal_within_k", AHEAD, 0.18, 2855, (0.15, 0.21))
WICKETS = answer("wickets", BUMRAH, 310.0, 7000)
ECONOMY = answer("economy", BUMRAH, 6.8, 7000)
BOWLING_RATE = answer("bowling_strike_rate", BUMRAH, 22.6, 7000)
TWO_BOWLERS = answer("matchup_difference", MATCHUP, 0.02, None, (-0.03, 0.07))
FIRST_INNINGS = answer("venue_first_innings_avg", WHOLE_GROUND, 167.0, 200)
CHASING = answer("venue_chasing_win_share", WHOLE_GROUND, 0.55, 190)
SPIN_SHARE = answer("venue_wicket_share", GROUND, 0.3, 900)
OVERTURNED = answer("review_overturn_rate", INDIA, 0.27, 300)
FEW_REVIEWS = answer("review_overturn_rate", INDIA, 0.27, 12)
WINS = answer("team_win_share", INDIA, 0.62, 800)
BY_RUNS = answer("match_won_by_runs", FINAL, 6.0, 1)
SPIKE = answer("workload_ratio_flag", WORKLOAD, 1.6, 626)
USUAL = answer("workload_ratio_flag", WORKLOAD, 1.32, 626)

EVERY_ANSWER = [
    RATE,
    OUT_RATE,
    ABOVE,
    DIFFERENCE,
    RUNS,
    AVERAGE,
    BOUNDARIES,
    DOTS,
    CHANCE,
    WICKETS,
    ECONOMY,
    BOWLING_RATE,
    TWO_BOWLERS,
    FIRST_INNINGS,
    CHASING,
    SPIN_SHARE,
    OVERTURNED,
    WINS,
    BY_RUNS,
    USUAL,
]


def write(agent: str, answers: list[Answer], template: str) -> Finding:
    llm = FakeLLM([json.dumps({"template": template})])
    return write_finding(agent, "A question", answers, llm, SPECS[agent], RULES)


def checks_failed(finding: Finding, answers: list[Answer]) -> list[str]:
    source = FakeFactSource()
    for each in answers:
        source.add(each.metric, each.subject, each.fact)
    found = []
    for failure in faith_check(finding, source, GATE):
        found.append(failure.check)
    return found


# One honest finding for each agent: it uses its own numbers, and every check passes

HONEST = [
    # Day 67 runs N11b with the real model; these words are scripted
    (
        "N11b",
        [RATE, CHANCE],
        "Kohli returns {c1}. It is unlikely he is out within six balls: {c2}.",
    ),
    # Day 68
    ("N11a", [FIRST_INNINGS, WINS], "India win at {c2}. The ground gives {c1}."),
    ("N11c", [RUNS, AVERAGE], "Kohli has {c1}, with {c2}."),
    ("N11d", [RATE, ABOVE], "Kohli returns {c1}, and {c2}."),
    # Day 69
    ("N11e", [WINS, BY_RUNS], "India win at {c1}. Australia had {c2}."),
    ("N11f", [FIRST_INNINGS, CHASING], "The ground gives {c1}, and {c2}."),
    ("N11g", [WICKETS, ECONOMY], "Bumrah has {c1} at {c2}."),
    # Day 70
    ("N11h", [BOUNDARIES], "Kohli finds {c1}."),
    ("N11i", [OVERTURNED], "India had {c1}."),
    ("N11j", [ABOVE], "Kohli sits at {c1}."),
    # Day 71
    ("N11k", [USUAL], "Bumrah shows {c1}."),
    ("N11l", [RATE], "Kohli returns {c1}."),
]


@pytest.mark.parametrize(("agent", "answers", "template"), HONEST)
def test_each_agent_writes_a_finding_that_passes_all_seven(agent, answers, template):
    finding = write(agent, answers, template)
    assert finding.unreadable is None, finding.unreadable
    assert checks_failed(finding, answers) == []


def test_the_twelve_are_all_here():
    agents = set()
    for agent, _, _ in HONEST:
        agents.add(agent)
    assert agents == set(ROSTER) - {"N11m"}


@pytest.mark.parametrize("agent", list(ROSTER))
def test_each_agent_is_offered_only_its_own_measures(agent):
    for claim in offered_claims(agent, EVERY_ANSWER):
        assert claim.metric in ROSTER[agent]


def test_the_engines_each_agent_was_built_on():
    # the plan's Days 69-71: N11f uses M2, N11g T2, N11i M4, N11k the M3 flag
    expected = {
        "N11f": {"N8.M2"},
        "N11g": {"N8.M1", "N9.T2"},
        "N11i": {"N8.M4"},
        "N11k": {"N8.M3"},
    }
    for agent, engines in expected.items():
        found = set()
        for claim in offered_claims(agent, EVERY_ANSWER):
            found.add(claim.source)
        assert found == engines, agent


def test_nothing_on_the_menu_is_left_without_an_agent_offered_it():
    offered = set()
    for agent in ROSTER:
        for claim in offered_claims(agent, EVERY_ANSWER):
            offered.add(claim.metric)
    wanted = set()
    for each in EVERY_ANSWER:
        wanted.add(each.metric)
    assert offered == wanted


def test_a_measure_outside_the_roster_never_reaches_a_finding():
    # the model has no box for one; a rogue finding is stopped by C1
    claim = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="economy",
        subject=BUMRAH,
        value=6.8,
        n=7000,
        source="N8.M1",
    )
    rogue = Finding(agent="N11l", claims=(claim,), template="Bumrah has {c1}.")
    assert "C1" in checks_failed(rogue, [ECONOMY])


def test_the_menu_is_checked_when_the_graph_is_built():
    # Day 71 (D-007): the check runs on the real roster, and passes for all twelve
    graph_build.check_the_menu()
    graph_build.build_graph()


def test_a_hollow_measure_stops_the_build(monkeypatch):
    monkeypatch.setitem(ROSTER, "N11a", ("pitch_holding_pace",))
    with pytest.raises(RegistryError, match="pitch_holding_pace"):
        graph_build.build_graph()


def test_a_measure_with_no_engine_stops_the_build(monkeypatch):
    monkeypatch.delitem(graph_build.ENGINE_VERSIONS, "N8.M4")
    with pytest.raises(RegistryError, match="review_overturn_rate"):
        graph_build.build_graph()


# Words that must fail, agent by agent


def test_a_small_sample_must_say_so():
    assert checks_failed(
        write("N11i", [FEW_REVIEWS], "India had {c1}."), [FEW_REVIEWS]
    ) == ["C3"]
    honest = write("N11i", [FEW_REVIEWS], "On only a few reviews, India had {c1}.")
    assert checks_failed(honest, [FEW_REVIEWS]) == []


def test_a_difference_that_may_be_nothing_must_say_so():
    words = "Bumrah against Starc shows {c1}."
    assert "C4" in checks_failed(write("N11g", [TWO_BOWLERS], words), [TWO_BOWLERS])
    honest = "Kohli against Bumrah or Starc shows no detectable difference: {c1}."
    assert checks_failed(write("N11g", [TWO_BOWLERS], honest), [TWO_BOWLERS]) == []


def test_praise_needs_a_comparison_that_points_the_same_way():
    assert "C7" in checks_failed(
        write("N11l", [RATE], "Kohli is strong: {c1}."), [RATE]
    )
    assert (
        checks_failed(write("N11d", [ABOVE], "Kohli is strong: {c1}."), [ABOVE]) == []
    )
    assert "C7" in checks_failed(
        write("N11d", [LEVEL], "Kohli is strong: {c1}."), [LEVEL]
    )


def test_a_chance_cannot_be_called_certain():
    finding = write("N11b", [CHANCE], "He will be out: {c1}.")
    assert "C4" in checks_failed(finding, [CHANCE])


# N11k: its words about load follow the flag (Day 71, L-036)


def test_the_medical_agent_may_say_overworked_with_a_spike():
    finding = write("N11k", [SPIKE], "Bumrah looks overworked: {c1}.")
    assert checks_failed(finding, [SPIKE]) == []


def test_the_medical_agent_may_not_say_overworked_without_one():
    # the day Bumrah broke down at Sydney, the ratio was 1.32 (N-018)
    finding = write("N11k", [USUAL], "Bumrah looks overworked: {c1}.")
    source = FakeFactSource()
    source.add(USUAL.metric, USUAL.subject, USUAL.fact)
    (failure,) = faith_check(finding, source, GATE)
    assert failure.check == "C4"
    assert failure.reason == "'overworked' needs a spike, and the ratio is 1.32"


@pytest.mark.parametrize(
    "words",
    [
        "Bumrah shows {c1}, which is normal.",
        "Bumrah shows {c1}, so he is safe.",
        "Bumrah shows {c1}, a low risk.",
        "Bumrah shows {c1}, with a chance of injury.",
        "Bumrah shows {c1}. He may break down.",  # any sentence of the finding
        "Bumrah is injured. He shows {c1}.",
    ],
)
def test_the_medical_agent_never_says_more_than_the_flag(words):
    # with a spike too: these are never backed
    assert "C4" in checks_failed(write("N11k", [SPIKE], words), [SPIKE])


def test_the_four_weeks_balls_always_show():
    finding = write("N11k", [USUAL], "Bumrah shows {c1}.")
    assert render(finding) == (
        "Bumrah shows a workload ratio of 1.32, close to their usual week "
        "[D, 626 balls in four weeks]."
    )


def test_a_flag_is_about_a_count_a_small_sample_of_it_warns_too():
    assert lookup("workload_ratio_flag").unit == "balls"
    few = answer("workload_ratio_flag", WORKLOAD, 1.32, 12)
    assert "C3" in checks_failed(write("N11k", [few], "Bumrah shows {c1}."), [few])


# The Explainer: words only (Day 72)


def explain(words: str) -> Finding:
    llm = FakeLLM([json.dumps({"template": words})])
    return write_finding("N11m", "What is LBW?", [], llm, SPECS["N11m"], RULES)


def test_an_explanation_passes():
    words = (
        "LBW means leg before wicket. The batter is out when the ball would have hit "
        "the stumps, and it struck the leg first."
    )
    finding = explain(words)
    assert finding.unreadable is None
    assert finding.claims == ()
    assert checks_failed(finding, []) == []


def test_a_digit_in_an_explanation_is_rejected():
    assert "C5" in checks_failed(explain("An over has 6 balls."), [])


def test_a_number_in_words_is_fine_in_an_explanation():
    assert checks_failed(explain("An over has six balls."), []) == []


def test_a_players_name_in_an_explanation_is_rejected():
    finding = explain("LBW is how Kohli was out.")
    (failure,) = faith_check(finding, FakeFactSource(), GATE)
    assert failure.check == "C6"
    assert failure.reason == "an explanation may not name a player: 'kohli'"


def test_a_name_at_the_start_of_an_explanation_is_rejected():
    assert "C6" in checks_failed(explain("Bumrah bowls it straight."), [])


def test_everyday_words_are_not_names_in_an_explanation():
    # "ball", "chance" and "to" are players in the register
    words = "A ball is called wide when the batter has no chance to reach it."
    assert checks_failed(explain(words), []) == []


def test_a_shot_may_be_named_in_an_explanation():
    assert checks_failed(explain("A pull is played across the line."), []) == []


def test_praise_in_an_explanation_is_rejected():
    assert "C7" in checks_failed(explain("It is a good rule."), [])


def test_a_slot_in_an_explanation_cannot_be_read():
    finding = explain("LBW is {c1}.")
    assert "no claim has that id" in finding.unreadable
    assert checks_failed(finding, []) == ["C1"]


def test_a_claim_in_an_explanation_is_rejected():
    claim = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="strike_rate",
        subject=KOHLI,
        value=135.1,
        n=9841,
        source="N8.M1",
    )
    rogue = Finding(agent="N11m", claims=(claim,), template="It is {c1}.")
    assert "C1" in checks_failed(rogue, [RATE])
