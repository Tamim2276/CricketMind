"""Day 58, part 1: N3 reads with the model, and the code checks every box. The fake
model's replies are the mistakes real models made on Day 57 (L-069)."""

import json

from cricketmind_v3.application.ports import FakeLLM, LLMError
from cricketmind_v3.application.reader import QuestionForm, read_question
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.grounds import Ground
from cricketmind_v3.domain.names import Player
from cricketmind_v3.domain.reading import AskValue
from cricketmind_v3.domain.subject import CricketSubject

PLAYERS = (
    Player(person_id="ba607b88", name="V Kohli", aliases=("Virat Kohli",), matches=600),
    Player(person_id="462411b3", name="JJ Bumrah", matches=300),
    Player(person_id="7c7d63a2", name="MA Starc", matches=350),
    Player(person_id="30a45b23", name="SPD Smith", matches=400, teams=("Australia",)),
    Player(person_id="6f0ef3ea", name="GC Smith", matches=300, teams=("South Africa",)),
    Player(person_id="2f39d6a1", name="BKG Mendis", matches=150),
    Player(
        person_id="d5f0b6c3",
        name="PHKD Mendis",
        aliases=("Kamindu Mendis",),
        matches=80,
    ),
)
NAMES = cricket_gazetteer({"V Kohli": ("Kohli",)})
GROUNDS = (
    Ground(
        name="Shere Bangla National Stadium", places=("Dhaka", "Mirpur"), matches=497
    ),
    Ground(
        name="M Chinnaswamy Stadium, Bangalore",
        other_names=("M Chinnaswamy Stadium, Bengaluru", "M.Chinnaswamy Stadium"),
        places=("Bangalore", "Bengaluru"),
        matches=173,
    ),
    Ground(name="Eden Gardens", places=("Kolkata",), matches=176),
    Ground(name="Eden Park Outer Oval", places=("Auckland",), matches=90),
    Ground(name="Wankhede Stadium", places=("Mumbai",), matches=213),
    Ground(name="Dr DY Patil Sports Academy, Mumbai", places=("Mumbai",), matches=73),
    Ground(name="Lord's", places=("London",), matches=312),
    Ground(name="Lords", places=("Bermuda",), matches=3),
)


def form(**given) -> str:
    """A reply as a model would send it: every field, null unless given."""
    values = dict.fromkeys(QuestionForm.model_fields)
    values.update(given)
    return json.dumps(values)


def read(question: str, *replies):
    llm = FakeLLM(list(replies))
    return read_question(question, llm, PLAYERS, NAMES, GROUNDS), llm


def test_the_model_reads_and_the_register_names_the_player():
    reading, _ = read(
        "Kohli out within 6 balls in the T20 powerplay against pace?",
        form(
            player="Kohli",
            k=6,
            match_format="T20",
            phase="powerplay",
            bowler_type="pace",
        ),
    )
    assert reading.by == "model"
    assert reading.subject == CricketSubject(
        player="V Kohli", k=6, match_format="T20", phase="powerplay", bowler_type="pace"
    )


def test_a_made_up_name_is_never_used():
    # Day 57: qwen2.5:7b read "Starc" as "Stern"
    reading, _ = read(
        "Would Bumrah or Starc get Smith out sooner in Tests?",
        form(player="Bumrah", bowler="Stern", match_format="Test"),
    )
    assert reading.not_in_question == ("Stern",)
    assert reading.subject.bowler is None


def test_a_name_several_players_have_is_asked_back():
    reading, _ = read(
        "Would Bumrah or Starc get Smith out sooner in Tests?",
        form(
            player="Smith", bowler="Bumrah", other_bowler="Starc", match_format="Test"
        ),
    )
    assert reading.subject == CricketSubject(
        bowler="JJ Bumrah", other_bowler="MA Starc", match_format="Test"
    )
    assert len(reading.unsure) == 1
    asked = reading.unsure[0]
    assert (asked.field, asked.written) == ("player", "Smith")
    assert [player.name for player in asked.candidates] == ["SPD Smith", "GC Smith"]


def test_a_fuller_name_settles_it():
    reading, _ = read(
        "Kamindu Mendis out within 6 balls?", form(player="Kamindu Mendis", k=6)
    )
    assert reading.subject.player == "PHKD Mendis"
    assert reading.unsure == ()


def test_nobody_in_the_register():
    reading, _ = read("Zorro's strike rate?", form(player="Zorro"))
    assert reading.unknown == ("Zorro",)
    assert reading.subject == CricketSubject()


def test_the_word_null_is_not_a_name():
    # Day 57: qwen2.5:7b wrote "null" as a word, three times
    reading, _ = read("What is LBW?", form(player="null", bowler="null"))
    assert reading.subject == CricketSubject()
    assert reading.not_in_question == ("null", "null")


def test_a_filter_nobody_asked_for_is_dropped():
    reading, _ = read(
        "Kohli's strike rate?",
        form(player="Kohli", phase="powerplay", bowler_type="pace", match_format="T20"),
    )
    assert reading.subject == CricketSubject(player="V Kohli")
    assert reading.not_in_question == (
        "phase=powerplay",
        "bowler_type=pace",
        "match_format=T20",
    )


def test_other_ways_of_saying_a_value():
    reading, _ = read(
        "Kohli at the death against seamers in T20Is",
        form(player="Kohli", phase="death", bowler_type="pace", match_format="T20I"),
    )
    assert reading.subject == CricketSubject(
        player="V Kohli", phase="death", bowler_type="pace", match_format="T20I"
    )


def test_every_number_must_be_in_the_question():
    reading, _ = read(
        "Kohli out within six balls when chasing, since 2019, in the 2024/25 season?",
        form(player="Kohli", k=6, innings=2, since="2020", season="2024/25"),
    )
    assert reading.subject == CricketSubject(
        player="V Kohli", k=6, innings=2, season="2024/25"
    )
    assert reading.not_in_question == ("since=2020",)
    reading, _ = read("Kohli out within 6 balls?", form(player="Kohli", k=12))
    assert reading.subject.k is None
    assert reading.not_in_question == ("k=12",)


def test_a_match_day_in_words():
    reading, _ = read(
        "Who won India against Australia on 19 November 2023?",
        form(team="India", opposition="Australia", date="2023-11-19"),
    )
    assert reading.subject.date == "2023-11-19"


def test_a_reply_that_does_not_fit_is_asked_again_with_the_reason():
    # Day 57: "T20 powerplay against pace" is not a format
    reading, llm = read(
        "Kohli out within 6 balls in the T20 powerplay?",
        form(player="Kohli", k=6, match_format="T20 powerplay against pace"),
        form(player="Kohli", k=6, match_format="T20", phase="powerplay"),
    )
    assert reading.by == "model"
    assert reading.subject.match_format == "T20"
    assert len(llm.asked) == 2
    assert "did not fit the form" in llm.asked[1][1]


def test_twice_wrong_and_the_rules_read_it():
    reading, llm = read(
        "Kohli out within 6 balls in the T20 powerplay?",
        "not json",
        form(player="Kohli", k="six"),
    )
    assert len(llm.asked) == 2  # one retry, no more
    assert reading.by == "rules"
    assert reading.subject == CricketSubject(
        player="V Kohli", k=6, match_format="T20", phase="powerplay"
    )
    assert reading.notes[0].startswith("read by rules: the model's reply did not fit")


def test_no_model_reads_by_rules():
    reading = read_question("Kohli out within 6 balls?", None, PLAYERS, NAMES)
    assert reading.by == "rules"
    assert reading.subject == CricketSubject(player="V Kohli", k=6)


def test_ollama_not_running_reads_by_rules():
    reading, _ = read(
        "Kohli out within 6 balls?",
        LLMError("Ollama could not answer"),
        LLMError("Ollama could not answer"),
    )
    assert reading.by == "rules"
    assert reading.subject.player == "V Kohli"


def test_a_team_or_ground_with_no_gender_is_mens():
    reading, _ = read(
        "India's win share at Mirpur?", form(team="India", venue="Mirpur")
    )
    assert reading.subject.gender == "male"
    assert reading.subject.venue == "Shere Bangla National Stadium"
    assert reading.not_looked_up == (("team", "India"),)
    assert "men's" in reading.notes[0]
    reading, _ = read(
        "India women's win share at Mirpur?",
        form(team="India", venue="Mirpur", gender="female"),
    )
    assert reading.subject.gender == "female"
    assert reading.notes == ()


def test_a_player_question_gets_no_gender():
    reading, _ = read("Kohli's strike rate?", form(player="Kohli"))
    assert reading.subject.gender is None


# Day 58, part 2a


def test_the_next_over_is_asked_back_never_guessed():
    reading, _ = read("Kohli out in the next over?", form(player="Kohli"))
    assert reading.subject.k is None
    assert reading.asks == (
        AskValue(
            field="k",
            written="the next over",
            why="an over is six balls, but he may face fewer: his next 6 balls?",
            suggestion="6",
        ),
    )
    reading, _ = read("Kohli out in the next two overs?", form(player="Kohli"))
    assert reading.asks[0].suggestion == "12"


def test_a_day_without_its_year_is_asked_back():
    reading, _ = read(
        "Who won India against Australia on 19 November?",
        form(team="India", opposition="Australia", date="2023-11-19"),
    )
    assert reading.subject.date is None
    assert reading.asks == (
        AskValue(field="date", written="19 November", why="which year?"),
    )
    assert reading.not_in_question == ()


def test_a_number_of_balls_in_words():
    reading, _ = read("Kohli out within twenty four balls?", form(player="Kohli", k=24))
    assert reading.subject.k == 24


def test_a_side_in_the_question_tells_two_smiths_apart():
    reading, _ = read(
        "Australia's Smith out within 6 balls?",
        form(player="Smith", team="Australia", k=6),
    )
    assert reading.subject.player == "SPD Smith"
    assert reading.unsure == ()
    assert "Smith read as the only one of 2" in reading.notes[1]


def test_only_a_checked_value_is_a_clue():
    # the model adds "Australia", which the question never says: no clue, so asked
    reading, _ = read("Smith's strike rate?", form(player="Smith", team="Australia"))
    assert reading.not_in_question == ("Australia",)
    assert reading.subject.player is None
    assert reading.unsure[0].written == "Smith"


def test_a_misspelt_name_is_asked_back_with_close_spellings():
    reading, _ = read("Kholi's strike rate?", form(player="Kholi"))
    assert reading.subject.player is None
    assert reading.unknown == ()
    assert reading.unsure[0].why == "nobody has this name; did you mean one of these?"
    assert reading.unsure[0].candidates[0].name == "V Kohli"


def test_a_bowling_word_is_not_a_name():
    # the real run: qwen3:8b put "pace" in a player's box, and N3 offered "U Peace"
    reading, _ = read(
        "Kohli out within 6 balls in the T20 powerplay against pace?",
        form(
            player="Kohli", bowler="pace", k=6, match_format="T20", bowler_type="pace"
        ),
    )
    assert reading.subject.bowler is None
    assert reading.subject.bowler_type == "pace"
    assert '"pace" is not the name of a player' in reading.notes
    assert reading.unknown == () and reading.unsure == ()


# Day 58, part 2b: grounds


def test_a_ground_by_its_place():
    reading, _ = read("Spin's share of wickets at Mirpur?", form(venue="Mirpur"))
    assert reading.subject.venue == "Shere Bangla National Stadium"
    assert reading.subject.gender == "male"  # a ground, and no gender named


def test_a_ground_by_any_of_its_names():
    reading, _ = read("Totals at Chinnaswamy?", form(venue="Chinnaswamy"))
    assert reading.subject.venue == "M Chinnaswamy Stadium, Bangalore"
    reading, _ = read("Totals at M.Chinnaswamy?", form(venue="M.Chinnaswamy"))
    assert reading.subject.venue == "M Chinnaswamy Stadium, Bangalore"


def test_lords_in_london_not_lords_in_bermuda():
    reading, _ = read("First innings at Lord's?", form(venue="Lord's"))
    assert reading.subject.venue == "Lord's"
    assert "Lord's read as Lord's: by far the most-played of 2 grounds" in reading.notes


def test_several_grounds_close_are_asked():
    reading, _ = read("Totals at Eden?", form(venue="Eden"))
    assert reading.subject.venue is None
    asked = reading.unsure[0]
    assert (asked.field, asked.written) == ("venue", "Eden")
    assert [g.name for g in asked.candidates] == [
        "Eden Gardens",
        "Eden Park Outer Oval",
    ]
    reading, _ = read("Totals in Mumbai?", form(venue="Mumbai"))
    assert len(reading.unsure[0].candidates) == 2  # Wankhede 213, DY Patil 73: asked


def test_a_ground_nobody_knows():
    reading, _ = read("Totals at Hogwarts?", form(venue="Hogwarts"))
    assert reading.unknown == ("Hogwarts",)
    assert reading.subject.venue is None


def test_a_ground_not_in_the_question():
    reading, _ = read("Kohli's strike rate?", form(player="Kohli", venue="Wankhede"))
    assert reading.subject.venue is None
    assert reading.not_in_question == ("Wankhede",)
    assert reading.subject.gender is None  # no ground was really named
