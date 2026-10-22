"""Day 40: the M2 engine on a tiny DuckDB file made from the three fixture matches.

Counted by hand from the files, one match per ground, all T20s of 20 overs:
- Lord's (a women's match): England 150/4 in 20 overs; Australia chased 151 and won.
- Maharaja Yadavindra Singh International Cricket Stadium (New Chandigarh): Rajasthan
  Royals 243/8, and won; two of their eight wickets were run outs.
- Rawalpindi Cricket Stadium: Pakistan 215/3, and won. Ten bowlers' wickets: Theekshana,
  Shanaka and Malinga in the first innings; Salman Mirza 2, Mohammad Imran 3, Saim Ayub
  and Shadab Khan in the second.
"""

import dataclasses

import duckdb
import pytest

from cricketmind_v3.application.engines.m2_venues import (
    venue_chasing_win_share,
    venue_first_innings_avg,
    venue_wicket_share,
)
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.domain.venues import Wicket
from cricketmind_v3.infrastructure.duckdb_venues import DuckDBVenueRepository
from cricketmind_v3.pipeline.attributes import (
    Attributes,
    build_attributes,
    read_player_meta,
)
from cricketmind_v3.pipeline.derived import build_derived
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.players import build_players
from tests.integration.test_attributes_join import SAMPLE
from tests.integration.test_ingest import make_zip, quiet

LORDS = CricketSubject(venue="Lord's", match_format="T20", gender="female")
CHANDIGARH = CricketSubject(
    venue="Maharaja Yadavindra Singh International Cricket Stadium",
    match_format="T20",
    gender="male",
)
RAWALPINDI = CricketSubject(
    venue="Rawalpindi Cricket Stadium", match_format="T20", gender="male"
)

# bowler types given by hand for this test, to the four Rawalpindi bowlers the sample
# leaves out (the sample knows Theekshana, Malinga and Shadab Khan)
BY_HAND = {
    "MD Shanaka": "pace",
    "Salman Mirza": "pace",
    "Mohammad Imran": "pace",
    "Saim Ayub": "spin",
}


def build(folder, by_hand=False):
    path = folder / "cricket.duckdb"
    ingest(make_zip(folder), path, say=quiet)
    db = duckdb.connect(str(path))
    build_players(db)
    hand = []
    if by_hand:
        hand = labels(db, BY_HAND)
    build_attributes(db, read_player_meta(SAMPLE), hand)
    build_derived(db)
    return db


def labels(db, types: dict[str, str]) -> list[Attributes]:
    made = []
    for name, bowler_type in types.items():
        person_id = db.execute(
            "SELECT person_id FROM players WHERE name = ?", [name]
        ).fetchone()[0]
        made.append(
            Attributes(
                person_id=person_id,
                known_as=None,
                full_name=None,
                country=None,
                batting_hand=None,
                bowler_type=bowler_type,
                batting_style=None,
                bowling_style=None,
                source="hand-labelled",
            )
        )
    return made


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


@pytest.fixture
def repo(db):
    return DuckDBVenueRepository(db)


# The first innings, and the chase


def test_first_innings_averages(repo):
    assert venue_first_innings_avg(repo, LORDS) == Fact(value=150.0, n=1)
    assert venue_first_innings_avg(repo, CHANDIGARH) == Fact(value=243.0, n=1)
    assert venue_first_innings_avg(repo, RAWALPINDI) == Fact(value=215.0, n=1)


def test_who_won_chasing(repo):
    assert venue_chasing_win_share(repo, LORDS) == Fact(value=1.0, n=1)
    assert venue_chasing_win_share(repo, RAWALPINDI) == Fact(value=0.0, n=1)


def test_another_format_or_ground_has_no_answer(repo):
    odi = CricketSubject(venue="Lord's", match_format="ODI", gender="female")
    men = CricketSubject(venue="Lord's", match_format="T20", gender="male")
    assert venue_first_innings_avg(repo, men) is None  # no men's match there
    assert venue_first_innings_avg(repo, odi) is None
    nowhere = CricketSubject(venue="Eden Park", match_format="T20", gender="male")
    assert venue_first_innings_avg(repo, nowhere) is None


def test_the_hundred_is_not_a_t20(db, repo):
    # five balls an over: 100 balls an innings, not 120
    db.execute("UPDATE matches SET balls_per_over = 5 WHERE match_id = '1490709'")
    assert venue_first_innings_avg(repo, LORDS) is None


# Pace and spin


def test_every_wicket_comes_back_counted(repo):
    wickets = repo.wickets_at(CHANDIGARH)
    total = 0
    run_outs = 0
    for wicket, times in wickets.items():
        total += times
        if wicket.kind == "run out":
            run_outs += times
    assert total == 18  # 8 + 10
    assert run_outs == 2
    malinga = Wicket(
        match_type="T20", scheduled_balls=120, bowler_type="pace", kind="caught"
    )
    assert wickets[malinga] == 1  # the only bowler there the sample knows


def test_too_few_known_bowlers_gives_no_answer(repo):
    # the sample knows the type of 3 of the 10 bowlers' wickets at Rawalpindi
    spin = CricketSubject(
        venue="Rawalpindi Cricket Stadium",
        match_format="T20",
        gender="male",
        bowler_type="spin",
    )
    assert venue_wicket_share(repo, spin) is None


def test_spins_share_with_every_bowler_known(tmp_path):
    db = build(tmp_path, by_hand=True)
    repo = DuckDBVenueRepository(db)
    spin = CricketSubject(
        venue="Rawalpindi Cricket Stadium",
        match_format="T20",
        gender="male",
        bowler_type="spin",
    )
    # spin: Theekshana, Saim Ayub, Shadab Khan; pace: the other seven
    assert venue_wicket_share(repo, spin) == Fact(value=0.3, n=10)
    db.close()


def test_a_super_over_wicket_is_not_counted(db, repo):
    # a copy of a wicket ball, as if it fell in a super over
    db.execute(
        "INSERT INTO deliveries SELECT * REPLACE (3 AS innings, true AS super_over) "
        "FROM deliveries WHERE match_id = '1490709' AND wicket_kind IS NOT NULL LIMIT 1"
    )
    total = 0
    for times in repo.wickets_at(LORDS).values():
        total += times
    assert total == 7  # 4 + 3, as before


def test_a_grounds_mens_and_womens_matches_are_kept_apart(repo):
    # the Lord's match is women's: no men's innings or wickets there
    men = dataclasses.replace(LORDS, gender="male")
    assert repo.innings_at(men) == []
    assert repo.wickets_at(men) == {}
    assert len(repo.innings_at(LORDS)) == 2
