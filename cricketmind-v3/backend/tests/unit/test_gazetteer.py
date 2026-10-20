from cricketmind_v3.domain.faith.gazetteer import (
    PERSON,
    SHOT,
    SHOTS,
    Entry,
    cricket_gazetteer,
)

# the fifteen classes of CricShot10k (as listed in the v2 config)
CRICSHOT10K = [
    "Cover Drive",
    "Defensive",
    "Down The Wicket",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted Legside",
    "Lofted Offside",
    "Pull",
    "Reverse Sweep",
    "Scoop",
    "Square Cut",
    "Straight Drive",
    "Sweep",
    "Upper Cut",
]


def test_all_fifteen_shots_are_known():
    assert sorted(SHOTS) == sorted(CRICSHOT10K)


def test_a_player_and_every_way_of_saying_him():
    names = cricket_gazetteer({"V Kohli": ("Kohli", "Virat Kohli")})
    kohli = [Entry(value="V Kohli", fields=PERSON)]
    assert names["v kohli"] == kohli
    assert names["kohli"] == kohli
    assert names["virat kohli"] == kohli


def test_shots_phases_and_bowler_types_are_there():
    names = cricket_gazetteer({})
    assert names["cover drive"] == [Entry(value="Cover Drive", fields=SHOT)]
    assert names["cover drives"] == [Entry(value="Cover Drive", fields=SHOT)]
    assert names["power play"] == [Entry(value="powerplay", fields=("phase",))]
    assert names["seamers"] == [Entry(value="pace", fields=("bowler_type",))]


def test_one_surname_can_belong_to_two_players():
    names = cricket_gazetteer({"RG Sharma": ("Sharma",), "I Sharma": ("Sharma",)})
    assert names["sharma"] == [
        Entry(value="RG Sharma", fields=PERSON),
        Entry(value="I Sharma", fields=PERSON),
    ]


def test_a_name_written_two_ways_is_stored_once():
    names = cricket_gazetteer({"V Kohli": ("Kohli", "KOHLI")})
    assert len(names["kohli"]) == 1
