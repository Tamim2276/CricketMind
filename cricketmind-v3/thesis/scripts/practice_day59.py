"""Day 59's practice questions: 30 questions, 6 of each kind, each asked 3 ways (90
wordings). Written by Claude for practice, never the locked exam's. Each wording has the
reading a perfect reader gives (the kind and every subject field) and the form a perfect
model fills, so that the set itself is checked before any model reads it.

Run from backend/ to write thesis/practice/day59_readings.jsonl:
    uv run python ../thesis/scripts/practice_day59.py
"""

import json
import sys

from ledger import PROJECT

OUT = PROJECT / "thesis" / "practice" / "day59_readings.jsonl"
CLIP = "clip.mp4"
RASHID = "5f547c8b"  # Rashid Khan of Afghanistan: two players have his name (D-078)

# (group, kind, clip, expected subject, [(wording, the perfect form's names), ...]);
# the perfect form also holds every value in the expected subject
GROUPS = [
    (
        "S1",
        "stats",
        None,
        {
            "player": "V Kohli",
            "opposition": "Australia",
            "since": "2020",
            "gender": "male",
        },
        [
            (
                "What is Kohli's batting average against Australia since 2020?",
                {"player": "Kohli"},
            ),
            (
                "Virat Kohli's average vs Australia from 2020 on?",
                {"player": "Virat Kohli"},
            ),
            (
                "Since 2020, how has Kohli averaged against Australia?",
                {"player": "Kohli"},
            ),
        ],
    ),
    (
        "S2",
        "stats",
        None,
        {"player": "Babar Azam", "match_format": "ODI"},
        [
            ("What is Babar Azam's strike rate in ODIs?", {"player": "Babar Azam"}),
            ("Babar Azam ODI strike rate?", {"player": "Babar Azam"}),
            (
                "How quickly does Babar Azam score in ODI cricket?",
                {"player": "Babar Azam"},
            ),
        ],
    ),
    (
        "S3",
        "stats",
        None,
        {"bowler": "JJ Bumrah", "phase": "death", "match_format": "T20"},
        [
            ("What is Bumrah's economy at the death in T20s?", {"bowler": "Bumrah"}),
            (
                "Jasprit Bumrah's economy rate in the death overs, T20 cricket?",
                {"bowler": "Jasprit Bumrah"},
            ),
            (
                "In T20s, how many runs an over does Bumrah give away at the death?",
                {"bowler": "Bumrah"},
            ),
        ],
    ),
    (
        "S4",
        "stats",
        None,
        {"team": "India", "venue": "Shere Bangla National Stadium", "gender": "male"},
        [
            (
                "What is India's win share at Mirpur?",
                {"team": "India", "venue": "Mirpur"},
            ),
            (
                "How often does India win at Mirpur?",
                {"team": "India", "venue": "Mirpur"},
            ),
            (
                "India's record of wins at the Shere Bangla National Stadium?",
                {"team": "India", "venue": "Shere Bangla National Stadium"},
            ),
        ],
    ),
    (
        "S5",
        "stats",
        None,
        {"bowler": RASHID, "batting_hand": "left", "match_format": "T20"},
        [
            (
                "What is Rashid Khan's economy against left-handers in T20s?",
                {"bowler": "Rashid Khan"},
            ),
            (
                "Rashid Khan bowling to left-handed batters in T20 cricket: his economy?",
                {"bowler": "Rashid Khan"},
            ),
            (
                "In T20s, how economical is Rashid Khan against left-handers?",
                {"bowler": "Rashid Khan"},
            ),
        ],
    ),
    (
        "S6",
        "stats",
        None,
        {
            "player": "SPD Smith",
            "team": "Australia",
            "match_format": "Test",
            "gender": "male",
        },
        [
            (
                "What is Smith's strike rate in Tests for Australia?",
                {"player": "Smith"},
            ),
            ("Australia's Smith: strike rate in Test cricket?", {"player": "Smith"}),
            ("How fast does Smith score for Australia in Tests?", {"player": "Smith"}),
        ],
    ),
    (
        "P1",
        "prediction",
        None,
        {
            "player": "V Kohli",
            "k": 6,
            "match_format": "T20",
            "phase": "powerplay",
            "bowler_type": "pace",
        },
        [
            (
                "Kohli out within 6 balls in the T20 powerplay against pace?",
                {"player": "Kohli"},
            ),
            (
                (
                    "What is the chance Virat Kohli gets out in his next six balls of pace in "
                    "the T20 powerplay?"
                ),
                {"player": "Virat Kohli"},
            ),
            (
                (
                    "Against pace in the powerplay of a T20, how likely is Kohli to be out "
                    "within six balls?"
                ),
                {"player": "Kohli"},
            ),
        ],
    ),
    (
        "P2",
        "prediction",
        None,
        {
            "player": "SPD Smith",
            "bowler": "JJ Bumrah",
            "other_bowler": "MA Starc",
            "match_format": "Test",
        },
        [
            (
                "Would Bumrah or Starc get Steve Smith out sooner in Tests?",
                {"player": "Steve Smith", "bowler": "Bumrah", "other_bowler": "Starc"},
            ),
            (
                (
                    "In Test cricket, who is likelier to dismiss Steve Smith first, Bumrah or "
                    "Starc?"
                ),
                {"player": "Steve Smith", "bowler": "Bumrah", "other_bowler": "Starc"},
            ),
            (
                "Steve Smith in Tests: is he more at risk against Bumrah or against Starc?",
                {"player": "Steve Smith", "bowler": "Bumrah", "other_bowler": "Starc"},
            ),
        ],
    ),
    (
        "P3",
        "prediction",
        None,
        {"player": "Babar Azam", "k": 12, "match_format": "ODI"},
        [
            ("Babar Azam out within 12 balls in ODIs?", {"player": "Babar Azam"}),
            (
                (
                    "What are the chances Babar Azam is dismissed in his next twelve balls in "
                    "an ODI?"
                ),
                {"player": "Babar Azam"},
            ),
            (
                "In ODIs, how likely is Babar Azam to be out within 12 balls?",
                {"player": "Babar Azam"},
            ),
        ],
    ),
    (
        "P4",
        "prediction",
        None,
        {"player": "RG Sharma", "k": 6, "bowler_type": "spin", "match_format": "T20I"},
        [
            (
                "Rohit Sharma out within 6 balls against spin in T20Is?",
                {"player": "Rohit Sharma"},
            ),
            (
                "Chance Rohit Sharma is out to spinners in his next 6 balls in a T20I?",
                {"player": "Rohit Sharma"},
            ),
            (
                (
                    "In T20Is, how likely is Rohit Sharma to lose his wicket within six balls "
                    "of spin?"
                ),
                {"player": "Rohit Sharma"},
            ),
        ],
    ),
    (
        "P5",
        "prediction",
        None,
        {"player": "PHKD Mendis", "k": 10, "match_format": "Test"},
        [
            (
                "Kamindu Mendis out within 10 balls in Tests?",
                {"player": "Kamindu Mendis"},
            ),
            (
                "What is the chance Kamindu Mendis is out in his next ten balls in a Test?",
                {"player": "Kamindu Mendis"},
            ),
            (
                (
                    "In Test cricket, how likely is Kamindu Mendis to be dismissed within 10 "
                    "balls?"
                ),
                {"player": "Kamindu Mendis"},
            ),
        ],
    ),
    (
        "P6",
        "prediction",
        None,
        {
            "player": "Shakib Al Hasan",
            "k": 6,
            "bowler_type": "pace",
            "match_format": "ODI",
        },
        [
            (
                "Shakib Al Hasan out within 6 balls against pace in ODIs?",
                {"player": "Shakib Al Hasan"},
            ),
            (
                "Chance Shakib gets out in his next six balls from fast bowlers in an ODI?",
                {"player": "Shakib"},
            ),
            (
                (
                    "In ODIs, how likely is Shakib Al Hasan to be out within 6 balls of pace "
                    "bowling?"
                ),
                {"player": "Shakib Al Hasan"},
            ),
        ],
    ),
    (
        "V1",
        "video",
        CLIP,
        {"shot": "Cover Drive"},
        [
            ("Is this a cover drive?", {}),
            ("Is the shot in this clip a cover drive?", {}),
            ("Cover drive or not, in this video?", {}),
        ],
    ),
    (
        "V2",
        "video",
        CLIP,
        {"player": "V Kohli", "shot": "Pull"},
        [
            ("Is Kohli playing a pull here?", {"player": "Kohli"}),
            ("Is this Virat Kohli's pull shot?", {"player": "Virat Kohli"}),
            ("In this clip, does Kohli play the pull?", {"player": "Kohli"}),
        ],
    ),
    (
        "V3",
        "video",
        CLIP,
        {},
        [
            ("Is his weight on the front foot in this clip?", {}),
            ("Where is the batter's weight in this video?", {}),
            ("Is he leaning forward into the shot here?", {}),
        ],
    ),
    (
        "V4",
        "video",
        CLIP,
        {},
        [
            ("What shot is this?", {}),
            ("Which shot does the batter play in this clip?", {}),
            ("Name the shot in this video.", {}),
        ],
    ),
    (
        "V5",
        "video",
        CLIP,
        None,  # the order of the two shots follows each wording
        [
            (
                "Is this a reverse sweep or a sweep?",
                {"shot": "Reverse Sweep", "other_shot": "Sweep"},
            ),
            (
                "Sweep or reverse sweep in this clip?",
                {"shot": "Sweep", "other_shot": "Reverse Sweep"},
            ),
            (
                "Does he play a reverse sweep here, or just a sweep?",
                {"shot": "Reverse Sweep", "other_shot": "Sweep"},
            ),
        ],
    ),
    (
        "V6",
        "video",
        CLIP,
        {"player": "Babar Azam"},
        [
            (
                "Is Babar Azam's head over his front knee in this clip?",
                {"player": "Babar Azam"},
            ),
            (
                "In this video, does Babar Azam keep his head over the front knee?",
                {"player": "Babar Azam"},
            ),
            (
                "Babar Azam here: head over the front knee or not?",
                {"player": "Babar Azam"},
            ),
        ],
    ),
    (
        "R1",
        "rules",
        None,
        {},
        [
            ("What is LBW?", {}),
            ("Explain the lbw law.", {}),
            ("When is a batter out leg before wicket?", {}),
        ],
    ),
    (
        "R2",
        "rules",
        None,
        {},
        [
            ("What is a free hit?", {}),
            ("When does a batter get a free hit?", {}),
            ("Explain free hits in cricket.", {}),
        ],
    ),
    (
        "R3",
        "rules",
        None,
        {},
        [
            ("How does DRS work?", {}),
            ("What is the decision review system?", {}),
            ("Explain how a side reviews an umpire's decision.", {}),
        ],
    ),
    (
        "R4",
        "rules",
        None,
        {},
        [
            ("What is a no-ball?", {}),
            ("When does the umpire call a no ball?", {}),
            ("Explain the no-ball law.", {}),
        ],
    ),
    (
        "R5",
        "rules",
        None,
        {},
        [
            ("What is the follow-on?", {}),
            ("When can a side enforce the follow-on?", {}),
            ("Explain the follow-on rule.", {}),
        ],
    ),
    (
        "R6",
        "rules",
        None,
        {},
        [
            ("What is a wide?", {}),
            ("When is a delivery called a wide?", {}),
            ("Explain wides in cricket.", {}),
        ],
    ),
    (
        "C1",
        "cant_check",
        None,
        {"player": "V Kohli"},
        [
            ("Is Kohli injured?", {"player": "Kohli"}),
            ("Does Virat Kohli have an injury right now?", {"player": "Virat Kohli"}),
            ("Is Kohli fit to play after his injury?", {"player": "Kohli"}),
        ],
    ),
    (
        "C2",
        "cant_check",
        None,
        {"player": "Babar Azam"},
        [
            ("What is Babar Azam's salary?", {"player": "Babar Azam"}),
            ("How much does Babar Azam earn a year?", {"player": "Babar Azam"}),
            ("Babar Azam's yearly pay?", {"player": "Babar Azam"}),
        ],
    ),
    (
        "C3",
        "cant_check",
        None,
        {"player": "RG Sharma"},
        [
            ("How much does Rohit Sharma earn?", {"player": "Rohit Sharma"}),
            ("What is Rohit Sharma's IPL salary?", {"player": "Rohit Sharma"}),
            (
                "Rohit Sharma's contract: how much is it worth?",
                {"player": "Rohit Sharma"},
            ),
        ],
    ),
    (
        "C4",
        "cant_check",
        None,
        {"player": "SPD Smith"},
        [
            ("Is Steve Smith going to retire this year?", {"player": "Steve Smith"}),
            ("When will Steve Smith retire?", {"player": "Steve Smith"}),
            ("Is Steve Smith retiring soon?", {"player": "Steve Smith"}),
        ],
    ),
    (
        "C5",
        "cant_check",
        None,
        {"player": "V Kohli"},
        [
            ("Who is Kohli's wife?", {"player": "Kohli"}),
            ("Is Virat Kohli married?", {"player": "Virat Kohli"}),
            ("Tell me about Kohli's family.", {"player": "Kohli"}),
        ],
    ),
    (
        "C6",
        "cant_check",
        None,
        {"player": "Babar Azam"},
        [
            ("How does Babar Azam feel about the captaincy?", {"player": "Babar Azam"}),
            (
                "Is Babar Azam upset about losing the captaincy?",
                {"player": "Babar Azam"},
            ),
            ("What does Babar Azam think of his critics?", {"player": "Babar Azam"}),
        ],
    ),
]


def rows() -> list[dict]:
    # the perfect form names people as the question writes them, and every other
    # value as the subject has it
    found = []
    for group, kind, clip, expected, wordings in GROUPS:
        for number, (question, said) in enumerate(wordings, start=1):
            subject = expected
            if subject is None:  # V5: the shots, in each wording's order
                subject = dict(said)
            form = {"kind": kind}
            for field, value in subject.items():
                if field in ("player", "bowler", "other_bowler") or field == "gender":
                    continue  # names come as written; gender only when the question says it
                form[field] = value
            for field, value in said.items():
                form[field] = value
            found.append(
                {
                    "id": f"{group}-{number}",
                    "kind": kind,
                    "question": question,
                    "clip": clip,
                    "form": form,
                    "expected": subject,
                }
            )
    return found


def main() -> None:
    written = rows()
    assert len(written) == 90, len(written)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = ""
    for row in written:
        text += json.dumps(row, ensure_ascii=False) + "\n"
    with open(OUT, "w", encoding="utf-8", newline="\n") as out:
        out.write(text)
    print(f"wrote {len(written)} wordings to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
