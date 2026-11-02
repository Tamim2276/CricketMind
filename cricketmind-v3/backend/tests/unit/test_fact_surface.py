"""Day 61: N8 and N9 ask for every number that fits the question."""

from cricketmind_v3.application.fact_surface import ask_engines, surface, ways_to_ask
from cricketmind_v3.application.ports import FakeFactSource
from cricketmind_v3.domain.facts import Answer, Fact
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.faith.v2_mode import check_v2
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import c1

KOHLI = CricketSubject(player="V Kohli", match_format="T20")
KOHLI_NEXT_6 = CricketSubject(player="V Kohli", match_format="T20", k=6)
BATTING = [
    "strike_rate",
    "dismissal_rate",
    "sr_vs_average",
    "runs",
    "batting_average",
    "boundary_share",
    "dot_ball_share",
]
BOWLING = ["wickets", "economy", "bowling_strike_rate"]


def asked(answers: list[Answer]) -> list[tuple[str, CricketSubject]]:
    found = []
    for answer in answers:
        found.append((answer.metric, answer.subject))
    return found


def metrics(answers: list[Answer]) -> list[str]:
    found = []
    for answer in answers:
        found.append(answer.metric)
    return found


def test_each_way_of_asking_once():
    assert ways_to_ask(KOHLI) == [KOHLI]
    assert ways_to_ask(KOHLI_NEXT_6) == [KOHLI_NEXT_6, KOHLI]


def test_a_comparison_as_a_whole_and_by_each_side():
    whole = CricketSubject(player="V Kohli", bowler="B", other_bowler="S", k=6)
    assert ways_to_ask(whole) == [
        whole,
        CricketSubject(player="V Kohli", bowler="B", other_bowler="S"),
        CricketSubject(player="V Kohli", bowler="B"),
        CricketSubject(player="V Kohli", bowler="B", k=6),
        CricketSubject(player="V Kohli", bowler="S"),
        CricketSubject(player="V Kohli", bowler="S", k=6),
    ]


def test_a_batter_gets_the_batting_measures_and_nothing_else():
    answers = ask_engines(KOHLI, "stats", "N8", FakeFactSource())
    assert metrics(answers) == BATTING


def test_a_bowler_gets_the_bowling_measures():
    bumrah = CricketSubject(bowler="JJ Bumrah", match_format="T20")
    assert metrics(ask_engines(bumrah, "stats", "N8", FakeFactSource())) == BOWLING


def test_a_batter_against_a_bowler_gets_both():
    both = CricketSubject(player="V Kohli", bowler="JJ Bumrah")
    answers = ask_engines(both, "stats", "N8", FakeFactSource())
    assert metrics(answers) == BATTING + BOWLING


def test_each_engine_asks_only_its_own_rows():
    assert ask_engines(KOHLI, "stats", "N9", FakeFactSource()) == []  # no k: no future
    answers = ask_engines(KOHLI_NEXT_6, "prediction", "N9", FakeFactSource())
    assert asked(answers) == [("p_dismissal_within_k", KOHLI_NEXT_6)]


def test_k_looks_ahead_so_the_past_is_counted_without_it():
    answers = ask_engines(KOHLI_NEXT_6, "prediction", "N8", FakeFactSource())
    assert metrics(answers) == BATTING
    for answer in answers:
        assert answer.subject == KOHLI


def test_a_what_if_is_a_matchup_and_a_chance_on_each_side():
    whole = CricketSubject(
        player="V Kohli",
        bowler="JJ Bumrah",
        other_bowler="MA Starc",
        match_format="T20",
        k=12,
    )
    bumrah = CricketSubject(
        player="V Kohli", bowler="JJ Bumrah", match_format="T20", k=12
    )
    starc = CricketSubject(
        player="V Kohli", bowler="MA Starc", match_format="T20", k=12
    )
    assert asked(ask_engines(whole, "prediction", "N9", FakeFactSource())) == [
        ("matchup_difference", whole),
        ("p_dismissal_within_k", bumrah),
        ("p_dismissal_within_k", starc),
    ]


def test_a_ground_and_a_team_ask_their_own_engines():
    ground = CricketSubject(venue="Wankhede", match_format="T20", gender="male")
    assert metrics(ask_engines(ground, "stats", "N8", FakeFactSource())) == [
        "venue_first_innings_avg",
        "venue_chasing_win_share",
    ]
    team = CricketSubject(team="India", gender="male")
    assert metrics(ask_engines(team, "stats", "N8", FakeFactSource())) == [
        "review_overturn_rate",
        "team_win_share",
    ]


def test_a_measure_from_the_clip_is_not_asked_of_these_engines():
    # weight_transfer needs nothing from the subject, so it "fits" an empty one (N6, I5)
    assert ask_engines(CricketSubject(), "video", "N8", FakeFactSource()) == []
    assert ask_engines(CricketSubject(), "video", "N9", FakeFactSource()) == []


def test_each_number_is_filed_under_exactly_what_was_asked():
    source = FakeFactSource()
    source.add("strike_rate", KOHLI, Fact(value=135.0, n=100))
    rohit = CricketSubject(player="RG Sharma", match_format="T20")
    source.add("strike_rate", rohit, Fact(value=99.0, n=50))
    answers = ask_engines(KOHLI, "stats", "N8", source)
    assert answers[0].subject == KOHLI
    assert answers[0].fact == Fact(value=135.0, n=100)  # never Rohit's


def test_no_answer_is_kept_but_is_not_a_number():
    source = FakeFactSource()
    source.add("strike_rate", KOHLI, Fact(value=135.0, n=100))
    source.add("runs", KOHLI, Fact(value=4000.0, n=100))
    answers = ask_engines(KOHLI, "stats", "N8", source)
    assert len(answers) == len(BATTING)  # asked seven
    assert answers[1].fact is None  # dismissal_rate: no answer
    assert surface(answers) == (135.0, 4000.0)


def test_a_rule_asks_for_nothing_and_no_engines_give_nothing():
    source = FakeFactSource()
    assert ask_engines(KOHLI, "rules", "N8", source) == []
    assert ask_engines(KOHLI, "stats", "N8", None) == []


def test_the_surface_is_what_v2s_gate_looks_at():
    answers = [Answer(metric="runs", subject=KOHLI, fact=Fact(value=142.0, n=50))]
    context = Context(surface=surface(answers))
    # c1 says 142.0 about another subject and another measure: v2 sees only the number
    finding = Finding(agent="N11b", claims=(c1(),), template="{c1}.")
    assert check_v2(finding, context) == []
    empty = Context(surface=surface([]))
    assert len(check_v2(finding, empty)) == 1
