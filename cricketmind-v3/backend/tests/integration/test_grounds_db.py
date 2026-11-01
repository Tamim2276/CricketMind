"""Day 58, part 2b: the merge in the data, on a small hand-made database: every match of
a ground ends up under its main name, before anything counts by ground."""

from datetime import date

import duckdb

from cricketmind_v3.pipeline.derived import BUILD, build_derived
from cricketmind_v3.pipeline.grounds import load_grounds, merge_grounds
from cricketmind_v3.pipeline.ingest import ingest
from tests.integration.test_ingest import make_zip, quiet

MATCHES = [
    ("1", "M Chinnaswamy Stadium, Bangalore", "Bangalore", date(2010, 4, 1)),
    ("2", "M Chinnaswamy Stadium, Bangalore", "Bangalore", date(2012, 4, 1)),
    ("3", "M Chinnaswamy Stadium, Bengaluru", "Bengaluru", date(2023, 4, 1)),
    ("4", "M.Chinnaswamy Stadium", "Bengaluru", date(2024, 4, 1)),
    ("5", "Lord's", "London", date(2019, 7, 14)),
    ("6", "Lords", "Bermuda", date(2019, 8, 1)),
]


def database():
    db = duckdb.connect()
    db.execute(
        "CREATE TABLE matches (match_id VARCHAR, venue VARCHAR, city VARCHAR, date DATE)"
    )
    db.execute("CREATE TABLE match_venues (match_id VARCHAR, venue_key VARCHAR)")
    for match_id, venue, city, day in MATCHES:
        db.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)", [match_id, venue, city, day]
        )
        db.execute("INSERT INTO match_venues VALUES (?, ?)", [match_id, venue])
    return db


def venues_table(db):
    for sql in BUILD:
        if "TABLE venues AS" in sql:
            db.execute(sql)


def test_every_match_of_a_ground_gets_its_main_name():
    db = database()
    assert merge_grounds(db) == 2  # two other names joined
    keys = db.execute(
        "SELECT match_id, venue_key FROM match_venues ORDER BY match_id"
    ).fetchall()
    assert keys == [
        ("1", "M Chinnaswamy Stadium, Bangalore"),
        ("2", "M Chinnaswamy Stadium, Bangalore"),
        ("3", "M Chinnaswamy Stadium, Bangalore"),
        ("4", "M Chinnaswamy Stadium, Bangalore"),
        ("5", "Lord's"),
        ("6", "Lords"),  # Bermuda stays apart
    ]


def test_the_register_knows_every_name_and_place():
    db = database()
    merge_grounds(db)
    venues_table(db)
    grounds = {}
    for ground in load_grounds(db):
        grounds[ground.name] = ground
    chinnaswamy = grounds["M Chinnaswamy Stadium, Bangalore"]
    assert chinnaswamy.matches == 4
    assert chinnaswamy.other_names == (
        "M Chinnaswamy Stadium, Bengaluru",
        "M.Chinnaswamy Stadium",
    )
    assert chinnaswamy.places == ("Bangalore", "Bengaluru")
    assert (chinnaswamy.first_year, chinnaswamy.last_year) == (2010, 2024)
    assert sorted(grounds) == ["Lord's", "Lords", "M Chinnaswamy Stadium, Bangalore"]


def test_the_whole_build_merges_before_it_counts(tmp_path):
    # the Day 36 rule keeps "Lord's, London" and "Lords, London" apart; the merge joins
    # them before the venues table counts matches by ground
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    with duckdb.connect(str(path)) as db:
        db.execute(
            "UPDATE matches SET venue = 'Lords, London', city = 'London' "
            "WHERE match_id = '1550709'"
        )
        build_derived(db)
        keys = db.execute(
            "SELECT venue_key FROM match_venues WHERE match_id IN ('1490709', '1550709')"
        ).fetchall()
        assert keys[0] == keys[1]
        assert db.execute(
            "SELECT matches, names_used FROM venues WHERE venue_key = ?", [keys[0][0]]
        ).fetchone() == (2, 2)
