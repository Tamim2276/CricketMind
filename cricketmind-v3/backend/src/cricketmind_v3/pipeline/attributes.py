"""Batting hand and bowler type for each player (Days 34-35).

Route A: the `player_meta` table of the cricketdata package (ESPNcricinfo's player pages,
16,101 players), downloaded once as a CSV, no R needed. Route B: a short hand-made file
for players it misses; a hand label wins over the downloaded one.

Run from backend/ after the ingest:
    uv run python -m cricketmind_v3.pipeline.attributes ../data/cricket.duckdb ../data/raw/player_meta.csv
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

# words that make a style pace or spin; pace is looked for first: "slow medium" is pace
PACE_WORDS = ("fast", "medium")
SPIN_WORDS = (
    "offbreak",
    "off break",
    "legbreak",
    "leg break",
    "googly",
    "orthodox",
    "wrist",
    "chinaman",
    "spin",
    "slow",
)


def batting_hand(style: str) -> str | None:
    """'Right-hand bat' or 'Right hand Bat' -> 'right'; anything else -> None."""
    text = style.strip().lower()
    if "bat" not in text:
        return None  # blank, or a bowling style typed in the wrong column
    if text.startswith("right"):
        return "right"
    if text.startswith("left"):
        return "left"
    return None


def bowler_type(style: str) -> str | None:
    """'Right arm Fast medium' -> 'pace'; 'Legbreak googly' -> 'spin'; unknown -> None.

    A player with two styles ("Right arm Medium, Right arm Offbreak") is taken by the
    first, his main one.
    """
    first = style.split(",")[0].strip().lower()
    for word in PACE_WORDS:
        if word in first:
            return "pace"
    for word in SPIN_WORDS:
        if word in first:
            return "spin"
    return None


@dataclass(frozen=True, kw_only=True)
class Attributes:
    """One row of the attributes table."""

    person_id: str  # the Cricsheet register id
    known_as: str | None  # "Virat Kohli": becomes a name to search by
    full_name: str | None
    country: str | None
    batting_hand: str | None  # "right" or "left"
    bowler_type: str | None  # "pace" or "spin"
    batting_style: str | None  # as written at the source
    bowling_style: str | None
    source: str  # "cricketdata" or "hand-labelled"


CREATE_ATTRIBUTES = """
CREATE OR REPLACE TABLE attributes (
    person_id VARCHAR PRIMARY KEY,
    known_as VARCHAR,
    full_name VARCHAR,
    country VARCHAR,
    batting_hand VARCHAR,
    bowler_type VARCHAR,
    batting_style VARCHAR,
    bowling_style VARCHAR,
    source VARCHAR NOT NULL
)
"""


def read_player_meta(path: Path) -> list[Attributes]:
    """The downloaded CSV, one row per player with a Cricsheet id; the first of a
    repeated id is kept."""
    found = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            person_id = (row.get("cricsheet_id") or "").strip()
            if not person_id or person_id in found:
                continue
            batting = (row.get("batting_style") or "").strip()
            bowling = (row.get("bowling_style") or "").strip()
            found[person_id] = Attributes(
                person_id=person_id,
                known_as=_text(row.get("name")),
                full_name=_text(row.get("full_name")),
                country=_text(row.get("country")),
                batting_hand=batting_hand(batting),
                bowler_type=bowler_type(bowling),
                batting_style=batting or None,
                bowling_style=bowling or None,
                source="cricketdata",
            )
    return list(found.values())


def read_hand_labels(path: Path) -> list[Attributes]:
    """Route B: person_id, batting_hand (right/left/blank), bowler_type (pace/spin/blank)."""
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Attributes(
                    person_id=row["person_id"].strip(),
                    known_as=None,
                    full_name=None,
                    country=None,
                    batting_hand=_label(row.get("batting_hand"), ("right", "left")),
                    bowler_type=_label(row.get("bowler_type"), ("pace", "spin")),
                    batting_style=None,
                    bowling_style=None,
                    source="hand-labelled",
                )
            )
    return rows


def _text(value: str | None) -> str | None:
    value = (value or "").strip()  # the source pads some names with a space
    return value or None


def _label(value: str | None, allowed: tuple[str, ...]) -> str | None:
    value = (value or "").strip().lower()
    if value == "":
        return None
    if value not in allowed:
        raise ValueError(f"'{value}' is not one of {allowed}")
    return value


def build_attributes(db, meta: list[Attributes], hand: list[Attributes]) -> int:
    """Store the attributes; a hand label replaces the downloaded one for that person."""
    chosen = {}
    for row in meta:
        chosen[row.person_id] = row
    for row in hand:
        old = chosen.get(row.person_id)
        if old is not None:
            # keep the names and the source's own words, take the hand-made labels
            row = Attributes(
                person_id=row.person_id,
                known_as=old.known_as,
                full_name=old.full_name,
                country=old.country,
                batting_hand=row.batting_hand,
                bowler_type=row.bowler_type,
                batting_style=old.batting_style,
                bowling_style=old.bowling_style,
                source="hand-labelled",
            )
        chosen[row.person_id] = row

    db.execute(CREATE_ATTRIBUTES)
    values = []
    for row in chosen.values():
        values.append(
            [
                row.person_id,
                row.known_as,
                row.full_name,
                row.country,
                row.batting_hand,
                row.bowler_type,
                row.batting_style,
                row.bowling_style,
                row.source,
            ]
        )
    if values:
        db.executemany(
            "INSERT INTO attributes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values
        )
    db.execute(CREATE_JOINED)
    return len(values)


# Day 35: every ball, with the batter's and bowler's ids, hand and type. LEFT JOINs keep
# a ball whose player has no attributes; one person per name in a match, and one row
# per person in attributes, so no ball is doubled.
CREATE_JOINED = """
CREATE OR REPLACE VIEW deliveries_full AS
SELECT d.*,
       pb.person_id AS batter_id,
       pw.person_id AS bowler_id,
       ab.batting_hand,
       aw.bowler_type
FROM deliveries d
LEFT JOIN people pb ON pb.match_id = d.match_id AND pb.name = d.batter
LEFT JOIN people pw ON pw.match_id = d.match_id AND pw.name = d.bowler
LEFT JOIN attributes ab ON ab.person_id = pb.person_id
LEFT JOIN attributes aw ON aw.person_id = pw.person_id
"""


@dataclass(frozen=True, kw_only=True)
class Coverage:
    """How many balls have a known bowler type and batting hand."""

    balls: int
    with_bowler_type: int
    with_batting_hand: int
    by_format: list  # (match type, gender, balls, share with bowler type, with hand)
    unknown_bowlers: list  # (bowler id, name, balls): the ones to hand-label first


def coverage(db, top: int = 20) -> Coverage:
    balls, typed, handed = db.execute(
        "SELECT COUNT(*), COUNT(bowler_type), COUNT(batting_hand) FROM deliveries_full"
    ).fetchone()
    by_format = db.execute(
        """
        SELECT match_type, gender, COUNT(*) AS balls,
               ROUND(AVG(CASE WHEN bowler_type IS NULL THEN 0 ELSE 1 END), 3),
               ROUND(AVG(CASE WHEN batting_hand IS NULL THEN 0 ELSE 1 END), 3)
        FROM deliveries_full GROUP BY match_type, gender ORDER BY balls DESC
        """
    ).fetchall()
    unknown = db.execute(
        f"""
        SELECT bowler_id, MIN(bowler) AS name, COUNT(*) AS balls
        FROM deliveries_full WHERE bowler_type IS NULL
        GROUP BY bowler_id ORDER BY balls DESC, name LIMIT {int(top)}
        """
    ).fetchall()
    return Coverage(
        balls=balls,
        with_bowler_type=typed,
        with_batting_hand=handed,
        by_format=by_format,
        unknown_bowlers=unknown,
    )


def coverage_report(found: Coverage) -> str:
    typed = _share(found.with_bowler_type, found.balls)
    handed = _share(found.with_batting_hand, found.balls)
    lines = [
        f"balls: {found.balls}",
        f"with a bowler type: {found.with_bowler_type} ({typed})",
        f"with a batting hand: {found.with_batting_hand} ({handed})",
        "",
        "| Format | Gender | Balls | With bowler type | With batting hand |",
        "|---|---|---|---|---|",
    ]
    for match_type, gender, balls, typed, handed in found.by_format:
        lines.append(
            f"| {match_type} | {gender} | {balls} | {typed:.1%} | {handed:.1%} |"
        )
    lines.append("")
    lines.append("Bowlers with no type, most balls first (hand-label these first):")
    for bowler_id, name, balls in found.unknown_bowlers:
        lines.append(f"- {name} ({bowler_id}): {balls} balls")
    return "\n".join(lines)


def _share(part: int, whole: int) -> str:
    if whole == 0:
        return "no balls"
    return f"{part / whole:.1%}"


def main(arguments: list[str]) -> None:
    if len(arguments) not in (2, 3):
        print(
            "usage: python -m cricketmind_v3.pipeline.attributes "
            "<database> <player_meta.csv> [hand_labels.csv]"
        )
        sys.exit(2)
    meta = read_player_meta(Path(arguments[1]))
    hand = []
    if len(arguments) == 3:
        hand = read_hand_labels(Path(arguments[2]))
    with duckdb.connect(arguments[0]) as db:
        stored = build_attributes(db, meta, hand)
        print(f"{stored} players with attributes ({len(hand)} hand-labelled)")
        print(coverage_report(coverage(db)))


if __name__ == "__main__":
    main(sys.argv[1:])
