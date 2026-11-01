"""Day 59: the five kinds by the rules, the read-back, and the honest answer."""

from cricketmind_v3.domain.kinds import kind_by_rules
from cricketmind_v3.domain.readback import cant_check_answer, read_back
from cricketmind_v3.domain.reading import AskValue, Reading, Unsure
from cricketmind_v3.domain.subject import CricketSubject

KOHLI = CricketSubject(player="V Kohli")
NOBODY = CricketSubject()


def test_the_five_kinds_by_the_rules():
    assert kind_by_rules("Is this a cover drive?", NOBODY, has_clip=True) == "video"
    assert kind_by_rules("Is Kohli injured?", KOHLI, has_clip=False) == "cant_check"
    assert (
        kind_by_rules("Kohli's salary this season?", KOHLI, has_clip=False)
        == "cant_check"
    )
    assert (
        kind_by_rules("Kohli out within 6 balls?", KOHLI, has_clip=False)
        == "prediction"
    )
    assert kind_by_rules("What is LBW?", NOBODY, has_clip=False) == "rules"
    assert kind_by_rules("Kohli's average?", KOHLI, has_clip=False) == "stats"


def test_a_law_word_with_somebody_in_it_is_not_a_rules_question():
    # "lbw": alone, the rules ask what it is; with a player, it is his record
    assert (
        kind_by_rules("How often is Kohli out lbw?", KOHLI, has_clip=False) == "stats"
    )
    assert kind_by_rules("How often is a batter out lbw?", NOBODY, has_clip=False) == (
        "rules"
    )


def test_a_clip_beats_every_word():
    assert kind_by_rules("Is Kohli injured?", KOHLI, has_clip=True) == "video"


def reading(**fields) -> Reading:
    return Reading(subject=CricketSubject(**fields), by="model", kind="prediction")


def test_the_read_back_says_what_was_read():
    read = reading(
        player="V Kohli", k=6, match_format="T20", phase="powerplay", bowler_type="pace"
    )
    assert read_back(read) == (
        "I read this as: a prediction: V Kohli, out within 6 balls, T20, the powerplay, "
        "against pace."
    )


def test_the_read_back_says_what_it_asks_and_does_not_know():
    read = Reading(
        subject=CricketSubject(bowler="JJ Bumrah", other_bowler="MA Starc"),
        by="model",
        kind="prediction",
        unsure=(Unsure(field="player", written="Smith", candidates=("a", "b", "c")),),
        asks=(AskValue(field="k", written="the next over", why="his next 6 balls?"),),
        unknown=("Zorro",),
    )
    assert read_back(read).split("\n") == [
        "I read this as: a prediction: facing JJ Bumrah or MA Starc.",
        "Which Smith? several players have this name (3).",
        "About the next over: his next 6 balls?",
        "I do not know Zorro.",
    ]


def test_a_register_id_is_read_back_as_who_he_is():
    read = Reading(
        subject=CricketSubject(player="a1a1a1a1"),
        by="model",
        kind="stats",
        shown=(("a1a1a1a1", "Imran Khan (Pakistan, 1971-1992)"),),
    )
    assert read_back(read) == (
        "I read this as: a question about what happened: Imran Khan (Pakistan, 1971-1992)."
    )


def test_an_empty_reading_still_says_its_kind():
    read = Reading(subject=NOBODY, by="model", kind="rules")
    assert read_back(read) == "I read this as: a question about the rules."


def test_the_honest_answer_says_what_can_be_asked_instead():
    read = Reading(subject=KOHLI, by="model", kind="cant_check")
    answer = cant_check_answer(read)
    assert answer.startswith("I can't check that:")
    assert "You could ask instead: V Kohli's record" in answer
    general = cant_check_answer(Reading(subject=NOBODY, by="model", kind="cant_check"))
    assert "a player's record, a ground, or a team's results" in general
