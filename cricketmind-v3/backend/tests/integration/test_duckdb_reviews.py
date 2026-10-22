"""Day 42: the M4 engine on a tiny DuckDB file made from the three fixture matches.

Every review in the files, counted by hand:
- England v Australia, women's (Lord's): Australia 3 (Wyatt upheld, Capsey struck down,
  Voll upheld); England 2 (Knight struck down, Mooney upheld).
- Rajasthan Royals v Sunrisers Hyderabad: Rajasthan Royals 1 (Klaasen, upheld).
- Pakistan v Sri Lanka: Sri Lanka 1 (struck down); Pakistan 1 (Igalagamage, upheld).
"""

import duckdb
import pytest

from cricketmind_v3.application.engines.m4_reviews import review_overturn_rate
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_reviews import DuckDBReviewRepository
from cricketmind_v3.pipeline.derived import build_derived
from cricketmind_v3.pipeline.ingest import ingest
from tests.integration.test_ingest import make_zip, quiet


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    connection = duckdb.connect(str(path))
    build_derived(connection)
    yield connection
    connection.close()


@pytest.fixture
def repo(db):
    return DuckDBReviewRepository(db)


def team(name: str, gender: str) -> CricketSubject:
    return CricketSubject(team=name, gender=gender)


def test_each_teams_reviews(repo):
    assert repo.decisions(team("Australia", "female")) == {
        "upheld": 2,
        "struck down": 1,
    }
    assert review_overturn_rate(repo, team("Australia", "female")) == Fact(
        value=2 / 3, n=3
    )
    assert review_overturn_rate(repo, team("England", "female")) == Fact(value=0.5, n=2)
    assert review_overturn_rate(repo, team("Rajasthan Royals", "male")) == Fact(
        value=1.0, n=1
    )
    assert review_overturn_rate(repo, team("Sri Lanka", "male")) == Fact(value=0.0, n=1)


def test_mens_and_womens_teams_are_kept_apart(repo):
    # Australia's reviews here are the women's; the men's team has none
    assert review_overturn_rate(repo, team("Australia", "male")) is None
    assert review_overturn_rate(repo, team("India", "male")) is None


def test_a_technical_failure_is_left_out(db, repo):
    db.execute(
        "UPDATE reviews SET review_decision = 'technical failure' "
        "WHERE review_batter = 'A Capsey'"
    )
    assert review_overturn_rate(repo, team("Australia", "female")) == Fact(
        value=1.0, n=2
    )
