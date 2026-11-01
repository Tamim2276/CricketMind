"""Day 56: reading a question by simple rules."""

from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.reading import (
    appears_in,
    read_by_rules,
    says_date,
    says_digits,
    says_innings,
    says_number,
)
from cricketmind_v3.domain.subject import CricketSubject

NAMES = cricket_gazetteer(
    {
        "V Kohli": ("Kohli", "Virat Kohli", "Virat"),
        "JJ Bumrah": ("Bumrah", "Jasprit Bumrah"),
        "MA Starc": ("Starc", "Mitchell Starc"),
        "BKG Mendis": ("Mendis",),
        "PHKD Mendis": ("Mendis", "Kamindu Mendis"),
    }
)


def read(question: str) -> CricketSubject:
    return read_by_rules(question, NAMES)


def test_the_example_question():
    assert read(
        "Kohli out within 6 balls in the T20 powerplay against pace?"
    ) == CricketSubject(
        player="V Kohli", k=6, match_format="T20", phase="powerplay", bowler_type="pace"
    )


def test_a_name_after_against_bowls():
    assert read("Virat Kohli's strike rate against Bumrah") == CricketSubject(
        player="V Kohli", bowler="JJ Bumrah"
    )
    assert read("Kohli vs Bumrah").bowler == "JJ Bumrah"


def test_two_bowlers_for_a_what_if():
    subject = read("Kohli against Bumrah or Starc, who gets him out sooner?")
    assert subject.player == "V Kohli"
    assert subject.bowler == "JJ Bumrah"
    assert subject.other_bowler == "MA Starc"


def test_two_shots_for_a_comparison():
    subject = read("Kohli: cover drive or pull?")
    assert subject.shot == "Cover Drive"
    assert subject.other_shot == "Pull"


def test_other_ways_of_saying_it():
    subject = read("Kohli at the death against spinners")
    assert subject.phase == "death"
    assert subject.bowler_type == "spin"
    assert read("Kohli in the power play").phase == "powerplay"


def test_a_name_two_players_share_is_never_guessed():
    assert read("Mendis out within 6 balls?").player is None
    assert read("Kamindu Mendis out within 6 balls?").player == "PHKD Mendis"


def test_a_name_said_twice_fills_one_field():
    assert read("Kohli, is Kohli's pull good? The pull, I mean") == CricketSubject(
        player="V Kohli", shot="Pull"
    )
    assert read("Kohli against Bumrah: does Bumrah get him?").other_bowler is None


def test_k_counts_balls_not_overs():
    assert read("Kohli out within 12 balls?").k == 12
    assert read("Kohli out within 6 overs?").k is None
    assert read("Kohli out within six balls?").k is None  # a word, not a digit


def test_the_format_names_of_d047():
    assert read("Kohli in T20Is").match_format == "T20I"
    assert read("Kohli's T20I strike rate").match_format == "T20I"
    assert read("Kohli in T20 cricket").match_format == "T20"
    assert read("Kohli in ODIs").match_format == "ODI"
    assert read("Kohli in Tests").match_format == "Test"


def test_women():
    assert read("India women's average in ODIs").gender == "female"
    assert read("India's average in ODIs").gender is None


def test_nothing_known_reads_as_empty():
    assert read("What is LBW?") == CricketSubject()
    assert read("") == CricketSubject()


# Day 58: the checks on a model's reading


def test_a_name_must_be_in_the_question_word_for_word():
    question = "Would Bumrah or Starc get Smith out sooner?"
    assert appears_in("Starc", question)
    assert not appears_in("Stern", question)
    assert not appears_in("Mitchell Starc", question)  # more than it says
    assert appears_in("M.S. Dhoni", "MS Dhoni's average")
    assert not appears_in("", question)


def test_a_number_in_digits_or_words():
    assert says_number(6, "within 6 balls")
    assert says_number(6, "within six balls")
    assert not says_number(12, "within 6 balls")
    assert not says_number(6, "within 16 balls")


def test_every_digit_run_of_a_year_or_season():
    assert says_digits("2024/25", "the 2024/25 season")
    assert not says_digits("2024/25", "the 2024 season")
    assert not says_digits("2020", "since 2019")
    assert not says_digits("last season", "last season")  # no digits at all


def test_which_innings():
    assert says_innings(1, "batting first")
    assert says_innings(2, "when chasing")
    assert says_innings(2, "in the 2nd innings")
    assert not says_innings(1, "when chasing")


def test_a_day_needs_its_year_month_and_day():
    assert says_date("2024-04-12", "on 12 April 2024")
    assert says_date("2024-04-12", "on 12 Apr 2024")
    assert says_date("2024-04-12", "on 2024-04-12")
    assert not says_date("2024-04-12", "in April 2024")  # no day
    assert not says_date("2024-04-12", "on 12 May 2024")  # another month
    assert not says_date("2024-04-12", "on 12 April 2023")  # another year
    assert not says_date("not a day", "on 12 April 2024")
