"""Day 33: the players table built from the fixtures' register ids."""

import duckdb
import pytest

from cricketmind_v3.domain.names import find_players
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.players import build_players, load_players
from tests.integration.test_ingest import make_zip, quiet


@pytest.fixture
def players(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    with duckdb.connect(str(path)) as db:
        build_players(db)
        return load_players(db)


def by_name(players, name):
    for player in players:
        if player.name == name:
            return player
    raise KeyError(name)


def test_one_row_per_person(players):
    # 22 + 24 + 22 players, but E Malinga and MD Shanaka played two of the matches
    assert len(players) == 22 + 24 + 22 - 2


def test_one_person_two_teams_is_one_player(players):
    malinga = by_name(players, "E Malinga")
    assert malinga.matches == 2  # Sri Lanka and Sunrisers Hyderabad


def test_umpires_are_not_players(players):
    with pytest.raises(KeyError):
        by_name(players, "Faisal Afridi")


def test_searching_the_real_names(players):
    mendis = []
    for player in find_players("Mendis", players):
        mendis.append(player.name)
    assert sorted(mendis) == ["BKG Mendis", "PHKD Mendis"]
    assert find_players("Malinga", players)[0].name == "E Malinga"
    assert find_players("Eshan Malinga", players)[0].name == "E Malinga"
    assert find_players("Nithish Kumar Reddy", players)[0].name == "Nithish Kumar Reddy"
