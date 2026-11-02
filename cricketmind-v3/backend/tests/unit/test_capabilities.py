"""Day 62: which agents can answer this question."""

from cricketmind_v3.application.fact_surface import ask_engines
from cricketmind_v3.domain.capabilities import capable_agents, measures_with_numbers
from cricketmind_v3.domain.facts import Answer, Fact
from cricketmind_v3.domain.shot_dna import ShotDNA, stub_shot_dna
from cricketmind_v3.domain.subject import CricketSubject

NO_CLIP = stub_shot_dna(CricketSubject(), None)
KOHLI = CricketSubject(player="V Kohli", match_format="T20")


class AnswersEverything:
    """Engines with a number for every question: what is left is the menu's own rule."""

    def fact(self, metric, subject, clip_id=None):
        return Fact(value=1.0, n=100, interval=(0.0, 2.0))


def surface_of(subject: CricketSubject, kind: str = "stats") -> list[Answer]:
    source = AnswersEverything()
    n8 = ask_engines(subject, kind, "N8", source)
    n9 = ask_engines(subject, kind, "N9", source)
    return n8 + n9


def can_say(subject: CricketSubject, kind: str = "stats") -> dict:
    return capable_agents(kind, surface_of(subject, kind), NO_CLIP)


def all_measures(capable: dict) -> set[str]:
    found = set()
    for names in capable.values():
        found.update(names)
    return found


# The plan's three rules


def test_a_matchup_needs_two_bowlers():
    one = CricketSubject(player="V Kohli", bowler="B", match_format="T20", k=6)
    assert "matchup_difference" not in all_measures(can_say(one))
    two = CricketSubject(
        player="V Kohli", bowler="B", other_bowler="S", match_format="T20", k=6
    )
    capable = can_say(two)
    assert "matchup_difference" in capable["N11b"]
    assert "matchup_difference" in capable["N11g"]


def test_a_venue_claim_needs_a_venue():
    assert "N11f" not in can_say(KOHLI)
    ground = CricketSubject(venue="Wankhede", match_format="T20", gender="male")
    assert can_say(ground)["N11f"] == (
        "venue_first_innings_avg",
        "venue_chasing_win_share",
    )


def test_a_posture_needs_a_clip_and_a_measurement():
    posture = (("weight_transfer", 0.6),)
    measured = ShotDNA(subject=KOHLI, clip="clip.mp4", posture=posture)
    assert capable_agents("video", [], measured) == {
        "N11c": ("weight_transfer",),
        "N11j": ("weight_transfer",),
    }
    # a clip and nothing measured from it, as until Phase 8
    assert capable_agents("video", [], stub_shot_dna(KOHLI, "clip.mp4")) == {}
    # a posture with no clip cannot exist (I5)
    no_clip = ShotDNA(subject=KOHLI, posture=posture)
    assert capable_agents("stats", [], no_clip) == {}


def test_only_a_measured_row_counts_as_a_posture():
    wrong = (("strike_rate", 150.0), ("not_on_the_menu", 1.0))
    dna = ShotDNA(subject=KOHLI, clip="clip.mp4", posture=wrong)
    assert measures_with_numbers([], dna) == set()


# Real numbers only


def test_a_measure_with_no_number_counts_for_nothing():
    answers = [
        Answer(metric="strike_rate", subject=KOHLI, fact=None),
        Answer(metric="runs", subject=KOHLI, fact=Fact(value=4000.0, n=100)),
    ]
    assert capable_agents("stats", answers, NO_CLIP) == {
        "N11c": ("runs",),
        "N11e": ("runs",),
    }


def test_an_agent_is_left_out_when_none_of_its_measures_has_a_number():
    answers = [Answer(metric="economy", subject=KOHLI, fact=None)]
    assert capable_agents("stats", answers, NO_CLIP) == {}


def test_nobody_capable_is_said_plainly():
    assert capable_agents("stats", [], NO_CLIP) == {}


def test_a_small_sample_still_counts():
    # C3 lets it through with a warning in the sentence, so an agent can still say it
    answers = [Answer(metric="runs", subject=KOHLI, fact=Fact(value=7.0, n=2))]
    assert "N11c" in capable_agents("stats", answers, NO_CLIP)


# Who can answer what


def test_a_batter_wakes_the_batting_agents_only():
    capable = can_say(KOHLI)
    assert list(capable) == [
        "N11a",
        "N11b",
        "N11c",
        "N11d",
        "N11e",
        "N11h",
        "N11j",
        "N11l",
    ]
    assert capable["N11l"] == ("strike_rate",)
    assert "boundary_share" in capable["N11h"]


def test_a_bowler_wakes_the_bowling_agent_only():
    bumrah = CricketSubject(bowler="JJ Bumrah", match_format="T20")
    assert can_say(bumrah) == {"N11g": ("wickets", "economy", "bowling_strike_rate")}


def test_a_workload_wakes_the_medical_team_too():
    spell = CricketSubject(bowler="JJ Bumrah", date="2024-04-12")
    assert can_say(spell) == {"N11k": ("workload_ratio_flag",)}


def test_a_team_wakes_its_three():
    capable = can_say(CricketSubject(team="India", gender="male"))
    assert capable == {
        "N11a": ("team_win_share",),
        "N11e": ("team_win_share",),
        "N11i": ("review_overturn_rate",),
    }


def test_a_prediction_goes_to_the_coaches_and_keeps_the_past_for_the_rest():
    capable = can_say(CricketSubject(player="V Kohli", match_format="T20", k=6))
    assert "p_dismissal_within_k" in capable["N11b"]
    assert "strike_rate" in capable["N11c"]
    assert "p_dismissal_within_k" not in capable["N11c"]


def test_each_agent_gets_only_its_own_measures():
    capable = can_say(CricketSubject(player="V Kohli", bowler="JJ Bumrah"))
    assert "economy" in capable["N11g"]
    assert "economy" not in capable["N11c"]
    assert "strike_rate" not in capable["N11g"]


def test_a_rule_goes_to_the_explainer_alone():
    assert capable_agents("rules", [], NO_CLIP) == {"N11m": ()}


def test_the_explainer_is_not_called_for_a_number():
    assert "N11m" not in can_say(KOHLI)
    assert "N11m" not in capable_agents("stats", [], NO_CLIP)
