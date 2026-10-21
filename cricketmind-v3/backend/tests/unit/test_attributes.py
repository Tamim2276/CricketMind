"""Day 34: turning the source's style words into right/left and pace/spin."""

from pathlib import Path

import pytest

from cricketmind_v3.pipeline.attributes import (
    batting_hand,
    bowler_type,
    read_hand_labels,
    read_player_meta,
)

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "player_meta_sample.csv"


# Batting hand: both ways the source writes it


def test_batting_hands():
    assert batting_hand("Right-hand bat") == "right"
    assert batting_hand("Right hand Bat") == "right"
    assert batting_hand("Left hand Bat") == "left"
    assert batting_hand("Left-hand bat") == "left"


def test_no_batting_hand_from_a_blank_or_a_bowling_style():
    assert batting_hand("") is None
    # a bowling style typed in the batting column at the source
    assert batting_hand("Left-arm medium") is None


# Bowler type


def test_pace_bowlers():
    assert bowler_type("Right arm Fast medium") == "pace"
    assert bowler_type("Right-arm fast-medium") == "pace"
    assert bowler_type("Left arm Medium fast") == "pace"
    assert bowler_type("Right arm Slow medium") == "pace"  # slow medium is a seamer


def test_spin_bowlers():
    assert bowler_type("Right arm Offbreak") == "spin"
    assert bowler_type("Right-arm offbreak") == "spin"
    assert bowler_type("Legbreak Googly") == "spin"
    assert bowler_type("Slow left-arm orthodox") == "spin"
    assert bowler_type("Left arm Wrist spin") == "spin"


def test_two_styles_take_the_first():
    assert bowler_type("Right arm Medium, Right arm Offbreak") == "pace"
    assert bowler_type("Right arm Offbreak, Slow Left arm Orthodox") == "spin"
    # the case that tells "first style" from "any style": spin first, pace second
    assert bowler_type("Right arm Offbreak, Right arm Medium") == "spin"


def test_unknown_styles():
    assert bowler_type("") is None
    assert bowler_type("Right arm Bowler") is None


# Reading the downloaded file


def test_the_sample_file():
    rows = read_player_meta(SAMPLE)
    by_id = {}
    for row in rows:
        by_id[row.person_id] = row
    assert len(rows) == 8  # a repeated id and a missing id are left out
    malinga = by_id["5750bcb4"]
    assert (malinga.known_as, malinga.full_name) == ("Eshan Malinga", "Eshan Malinga")
    assert (malinga.batting_hand, malinga.bowler_type) == ("right", "pace")
    assert by_id["dabbd0ae"].batting_hand is None  # the wrong-column one
    assert by_id["1aba45cd"].bowler_type is None  # a batter with no bowling style


def test_hand_labels(tmp_path):
    labels = tmp_path / "hand.csv"
    labels.write_text(
        "person_id,batting_hand,bowler_type\nee1b6c27,left,pace\n6d9560f3,right,\n",
        encoding="utf-8",
    )
    rows = read_hand_labels(labels)
    assert (rows[0].batting_hand, rows[0].bowler_type) == ("left", "pace")
    assert (rows[1].batting_hand, rows[1].bowler_type) == ("right", None)


def test_a_hand_label_must_be_a_known_word(tmp_path):
    labels = tmp_path / "hand.csv"
    labels.write_text(
        "person_id,batting_hand,bowler_type\nx,right,quick\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_hand_labels(labels)
