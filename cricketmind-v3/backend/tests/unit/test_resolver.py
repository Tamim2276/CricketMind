"""Day 58, part 2a: which player a name means, by a small agentic search (D-055). The
fake model plays its part; the code takes only what it can check."""

import json

from cricketmind_v3.application.ports import FakeLLM, LLMError
from cricketmind_v3.application.resolver import LOOK_UPS, find_player
from cricketmind_v3.domain.names import Clues, Player


def player(person_id, name, matches, teams=(), years=(2010, 2020), aliases=()):
    return Player(
        person_id=person_id,
        name=name,
        aliases=aliases,
        matches=matches,
        teams=teams,
        formats=("ODI", "T20", "T20I", "Test"),
        first_year=years[0],
        last_year=years[1],
        gender="male",
    )


V_KOHLI = player("ba607b88", "V Kohli", 837, ("India",), (2008, 2026))
A_KOHLI = player("aaaa0003", "A Kohli", 41, ("Germany",), (2020, 2026))
STEVE = player("30a45b23", "SPD Smith", 574, ("Australia",), (2010, 2026))
DWAYNE = player("1c4c1b5f", "DR Smith", 319, ("West Indies",), (2004, 2019))
GRAEME = player("6f0ef3ea", "GC Smith", 314, ("South Africa",), (2002, 2014))
KUSAL = player("2f39d6a1", "BKG Mendis", 402, ("Sri Lanka",), (2015, 2026))
KAMINDU = player(
    "d5f0b6c3", "PHKD Mendis", 165, ("Sri Lanka",), (2018, 2026), ("Kamindu Mendis",)
)
AJANTHA = player("b1a2c3d4", "BAW Mendis", 156, ("Sri Lanka",), (2008, 2016))
PLAYERS = (V_KOHLI, A_KOHLI, STEVE, DWAYNE, GRAEME, KUSAL, KAMINDU, AJANTHA)


def step(search=None, choose=None, clue=None) -> str:
    return json.dumps({"search": search, "choose": choose, "clue": clue})


def find(written, question, clues=None, replies=None):
    llm = None
    if replies is not None:
        llm = FakeLLM(list(replies))
    if clues is None:
        clues = Clues()
    return find_player("player", written, question, clues, PLAYERS, llm), llm


def test_one_player_has_the_name():
    found, _ = find("Bumrah", "Bumrah?")
    assert found.player is None and found.unsure is None  # nobody: not in PLAYERS
    found, _ = find("Kamindu Mendis", "Kamindu Mendis out within 6 balls?")
    assert found.player == KAMINDU
    assert found.note is None  # nothing to explain


def test_clearly_the_most_played_is_kept_and_said():
    found, _ = find("Kohli", "Kohli out within 6 balls?")
    assert found.player == V_KOHLI
    assert (
        found.note
        == "Kohli read as V Kohli: by far the most-played of 2 players called Kohli"
    )


def test_a_checked_clue_tells_them_apart():
    found, _ = find("Smith", "Australia's Smith?", Clues(teams=("Australia",)))
    assert found.player == STEVE
    assert "the only one of 3" in found.note


def test_close_and_no_model_is_asked_back():
    found, _ = find("Smith", "Smith's strike rate?")
    assert found.player is None
    assert [p.name for p in found.unsure.candidates] == [
        "SPD Smith",
        "DR Smith",
        "GC Smith",
    ]
    assert found.look_ups == 1


def test_the_model_searches_again_with_a_fuller_name():
    question = "Is Kamindu Mendis better than his namesakes?"
    found, llm = find("Mendis", question, replies=[step(search="Kamindu Mendis")])
    assert found.player == KAMINDU
    assert found.look_ups == 2
    assert "Mendis" in llm.asked[0][1] and "BAW Mendis" in llm.asked[0][1]


def test_a_fuller_name_not_in_the_question_is_not_searched():
    found, _ = find("Mendis", "Mendis out?", replies=[step(search="Kamindu Mendis")])
    assert found.player is None
    assert len(found.unsure.candidates) == 3


def test_the_models_pick_stands_only_on_a_clue_that_fits_him_alone():
    question = "The Australian Smith: out within 6 balls?"
    found, _ = find(
        "Smith", question, replies=[step(choose="30a45b23", clue="The Australian")]
    )
    assert found.player == STEVE
    assert found.note == 'Smith read as SPD Smith: "The Australian" in the question'


def test_a_pick_whose_clue_is_not_in_the_question_is_refused():
    found, _ = find(
        "Smith", "Smith out?", replies=[step(choose="30a45b23", clue="the Australian")]
    )
    assert found.player is None and found.unsure is not None


def test_a_pick_whose_clue_fits_another_too_is_refused():
    # 2012 is in all three careers: it tells nobody apart
    found, _ = find(
        "Smith", "Smith in 2012?", replies=[step(choose="30a45b23", clue="2012")]
    )
    assert found.player is None


def test_a_pick_the_clue_does_not_fit_is_refused():
    found, _ = find(
        "Smith",
        "The Australian Smith?",
        replies=[step(choose="6f0ef3ea", clue="Australian")],
    )
    assert found.player is None  # GC Smith never played for Australia


def test_a_pick_of_nobody_shown_is_refused():
    found, _ = find(
        "Smith",
        "The Australian Smith?",
        replies=[step(choose="ffffffff", clue="Australian")],
    )
    assert found.player is None


def test_at_most_three_look_ups():
    question = "Was Ravi Kumar Singh out?"
    singhs = (
        player("s1", "A Kumar Singh", 100),
        player("s2", "B Kumar Singh", 90),
        player("s3", "Ravi Kumar Singh", 80),
        player("s4", "Ravi Kumar Singh", 70),
    )
    llm = FakeLLM(
        [
            step(search="Kumar Singh"),
            step(search="Ravi Kumar Singh"),
            step(search="Was Ravi Kumar Singh"),
            step(search="never asked"),
        ]
    )
    found = find_player("player", "Singh", question, Clues(), singhs, llm)
    assert found.look_ups == LOOK_UPS == 3
    assert len(llm.asked) == 3  # the fourth reply is never needed
    assert found.unsure is not None


def test_a_model_that_fails_leaves_it_to_the_user():
    found, _ = find("Smith", "Smith?", replies=[LLMError("Ollama could not answer")])
    assert found.unsure is not None


def test_a_misspelt_name_offers_close_spellings():
    found, _ = find("Kholi", "Kholi out within 6 balls?")
    assert found.player is None
    assert found.unsure.why == "nobody has this name; did you mean one of these?"
    assert [p.name for p in found.unsure.candidates] == ["V Kohli", "A Kohli"]


def test_a_name_somebody_has_never_offers_spellings():
    found, _ = find("Kohli", "Kohli?")
    assert found.unsure is None


def test_nobody_and_nothing_close():
    found, _ = find("Zorro", "Zorro?")
    assert found.player is None and found.unsure is None
