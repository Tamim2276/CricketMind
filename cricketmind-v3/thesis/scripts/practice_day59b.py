"""Day 59's second practice set: 30 new questions, 6 of each kind, each asked 3 ways (90
wordings), written before the fixes they will judge are coded. Never the exam's, and none
of the first set's questions. Each wording has its expected reading, the measure on the
menu it asks for (reported, deciding nothing), and the form a perfect model fills.

One group on purpose asks for a number no engine counts (Kohli's centuries): CricketMind
cannot check it, so its expected kind is cant_check (D-080).

Run from backend/ to write thesis/practice/day59b_readings.jsonl:
    uv run python ../thesis/scripts/practice_day59b.py
"""

import json
import sys

from ledger import PROJECT

OUT = PROJECT / "thesis" / "practice" / "day59b_readings.jsonl"
CLIP = "clip.mp4"

# (group, kind, clip, measure, expected subject, [(wording, the perfect form's names)])
GROUPS = [
    (
        "S1",
        "stats",
        None,
        "batting_average",
        {"player": "JE Root", "match_format": "Test", "since": "2021"},
        [
            (
                "What is Joe Root's batting average in Tests since 2021?",
                {"player": "Joe Root"},
            ),
            ("Joe Root Test average from 2021 on?", {"player": "Joe Root"}),
            (
                "How well has Joe Root averaged in Test cricket since 2021?",
                {"player": "Joe Root"},
            ),
        ],
    ),
    (
        "S2",
        "stats",
        None,
        "wickets",
        {"bowler": "PJ Cummins", "opposition": "India", "gender": "male"},
        [
            (
                "How many wickets has Pat Cummins taken against India?",
                {"bowler": "Pat Cummins"},
            ),
            ("Cummins' wickets against India?", {"bowler": "Cummins"}),
            (
                "Against India, how many batters has Pat Cummins dismissed?",
                {"bowler": "Pat Cummins"},
            ),
        ],
    ),
    (
        "S3",
        "stats",
        None,
        "team_win_share",
        {
            "team": "England",
            "opposition": "Australia",
            "venue": "Lord's",
            "gender": "male",
        },
        [
            (
                "What is England's win share against Australia at Lord's?",
                {"team": "England", "opposition": "Australia", "venue": "Lord's"},
            ),
            (
                "How often do England beat Australia at Lord's?",
                {"team": "England", "opposition": "Australia", "venue": "Lord's"},
            ),
            (
                "England against Australia at Lord's: how many of those matches did England win?",
                {"team": "England", "opposition": "Australia", "venue": "Lord's"},
            ),
        ],
    ),
    (
        "S4",
        "stats",
        None,
        "boundary_share",
        {"player": "Babar Azam", "match_format": "T20I"},
        [
            (
                "What share of Babar Azam's runs come in boundaries in T20Is?",
                {"player": "Babar Azam"},
            ),
            ("Babar Azam boundary share in T20Is?", {"player": "Babar Azam"}),
            (
                "In T20Is, how much of Babar Azam's scoring is fours and sixes?",
                {"player": "Babar Azam"},
            ),
        ],
    ),
    (
        "S5",
        "stats",
        None,
        "economy",
        {"bowler": "MA Starc", "phase": "powerplay", "match_format": "ODI"},
        [
            ("What is Starc's economy in the powerplay in ODIs?", {"bowler": "Starc"}),
            (
                "Mitchell Starc powerplay economy rate, ODIs?",
                {"bowler": "Mitchell Starc"},
            ),
            (
                "In ODIs, how many runs an over does Starc concede in the powerplay?",
                {"bowler": "Starc"},
            ),
        ],
    ),
    (
        "S6",
        "stats",
        None,
        "venue_first_innings_avg",
        {"venue": "Wankhede Stadium", "match_format": "T20", "gender": "male"},
        [
            (
                "What is the average first-innings score at the Wankhede in T20s?",
                {"venue": "Wankhede"},
            ),
            (
                "Wankhede Stadium first innings average, T20 cricket?",
                {"venue": "Wankhede Stadium"},
            ),
            (
                "In T20s, how many does the side batting first usually make at the Wankhede?",
                {"venue": "Wankhede"},
            ),
        ],
    ),
    (
        "P1",
        "prediction",
        None,
        "p_dismissal_within_k",
        {"player": "JE Root", "k": 18, "match_format": "Test"},
        [
            ("Joe Root out within 18 balls in Tests?", {"player": "Joe Root"}),
            (
                "What is the chance Joe Root is out in his next eighteen balls in a Test?",
                {"player": "Joe Root"},
            ),
            (
                "In Test cricket, how likely is Joe Root to be dismissed within 18 balls?",
                {"player": "Joe Root"},
            ),
        ],
    ),
    (
        "P2",
        "prediction",
        None,
        "matchup_difference",
        {
            "player": "V Kohli",
            "bowler": "PJ Cummins",
            "other_bowler": "MA Starc",
            "match_format": "ODI",
        },
        [
            (
                "Would Cummins or Starc get Kohli out sooner in ODIs?",
                {"player": "Kohli", "bowler": "Cummins", "other_bowler": "Starc"},
            ),
            (
                "In ODIs, who is likelier to dismiss Kohli first, Cummins or Starc?",
                {"player": "Kohli", "bowler": "Cummins", "other_bowler": "Starc"},
            ),
            (
                "Kohli in ODIs: is he more at risk against Cummins or against Starc?",
                {"player": "Kohli", "bowler": "Cummins", "other_bowler": "Starc"},
            ),
        ],
    ),
    (
        "P3",
        "prediction",
        None,
        "p_dismissal_within_k",
        {"player": "Babar Azam", "k": 6, "bowler_type": "spin", "match_format": "T20"},
        [
            (
                "Babar Azam out within 6 balls against spin in T20s?",
                {"player": "Babar Azam"},
            ),
            (
                "Chance Babar Azam falls to spinners in his next six balls in a T20?",
                {"player": "Babar Azam"},
            ),
            (
                "In T20s, how likely is Babar Azam to be out within 6 balls of spin?",
                {"player": "Babar Azam"},
            ),
        ],
    ),
    (
        "P4",
        "prediction",
        None,
        "p_dismissal_within_k",
        {"player": "Shakib Al Hasan", "k": 12, "match_format": "Test"},
        [
            ("Shakib out within 12 balls in Tests?", {"player": "Shakib"}),
            (
                "What are the chances Shakib Al Hasan is dismissed in his next twelve balls in a Test?",
                {"player": "Shakib Al Hasan"},
            ),
            (
                "In Test cricket, how likely is Shakib to lose his wicket within 12 balls?",
                {"player": "Shakib"},
            ),
        ],
    ),
    (
        "P5",
        "prediction",
        None,
        "p_dismissal_within_k",
        {"player": "PHKD Mendis", "k": 6, "bowler_type": "pace", "match_format": "ODI"},
        [
            (
                "Kamindu Mendis out within 6 balls against pace in ODIs?",
                {"player": "Kamindu Mendis"},
            ),
            (
                "Chance Kamindu Mendis gets out to the quicks in his next six balls in an ODI?",
                {"player": "Kamindu Mendis"},
            ),
            (
                "In ODIs, how likely is Kamindu Mendis to be out within 6 balls of pace?",
                {"player": "Kamindu Mendis"},
            ),
        ],
    ),
    (
        "P6",
        "prediction",
        None,
        "p_dismissal_within_k",
        {"player": "RG Sharma", "k": 8, "phase": "powerplay", "match_format": "T20"},
        [
            (
                "Rohit Sharma out within 8 balls in the T20 powerplay?",
                {"player": "Rohit Sharma"},
            ),
            (
                "What is the chance Rohit Sharma is out in his next eight balls in a T20 powerplay?",
                {"player": "Rohit Sharma"},
            ),
            (
                "In the powerplay of a T20, how likely is Rohit Sharma to be out within 8 balls?",
                {"player": "Rohit Sharma"},
            ),
        ],
    ),
    (
        "V1",
        "video",
        CLIP,
        None,
        {"shot": "Straight Drive"},
        [
            ("Is this a straight drive?", {}),
            ("Is the shot in this clip a straight drive?", {}),
            ("Straight drive or not, in this video?", {}),
        ],
    ),
    (
        "V2",
        "video",
        CLIP,
        None,
        {"player": "JE Root", "shot": "Sweep"},
        [
            ("Is Joe Root playing a sweep here?", {"player": "Joe Root"}),
            ("Is this Joe Root's sweep shot?", {"player": "Joe Root"}),
            ("In this clip, does Joe Root play the sweep?", {"player": "Joe Root"}),
        ],
    ),
    (
        "V3",
        "video",
        CLIP,
        None,
        {},
        [
            ("Is the batter's stride long in this clip?", {}),
            ("How long is his stride in this video?", {}),
            ("Does he take a big step forward here?", {}),
        ],
    ),
    (
        "V4",
        "video",
        CLIP,
        None,
        {},
        [
            ("What kind of shot did he play here?", {}),
            ("Which stroke is played in this clip?", {}),
            ("Tell me the shot in this video.", {}),
        ],
    ),
    (
        "V5",
        "video",
        CLIP,
        None,
        None,  # the order of the two shots follows each wording
        [
            ("Is this a hook or a pull?", {"shot": "Hook", "other_shot": "Pull"}),
            ("Pull or hook in this clip?", {"shot": "Pull", "other_shot": "Hook"}),
            (
                "Does he play a hook here, or a pull?",
                {"shot": "Hook", "other_shot": "Pull"},
            ),
        ],
    ),
    (
        "V6",
        "video",
        CLIP,
        "weight_transfer",
        {"player": "Babar Azam"},
        [
            (
                "Does Babar Azam move his weight forward in this video?",
                {"player": "Babar Azam"},
            ),
            (
                "In this clip, how well does Babar Azam transfer his weight?",
                {"player": "Babar Azam"},
            ),
            (
                "Babar Azam here: is his weight going forward or back?",
                {"player": "Babar Azam"},
            ),
        ],
    ),
    (
        "R1",
        "rules",
        None,
        None,
        {},
        [
            ("What is a run out?", {}),
            ("When is a batter run out?", {}),
            ("Explain the run out law.", {}),
        ],
    ),
    (
        "R2",
        "rules",
        None,
        None,
        {},
        [
            ("What is a super over?", {}),
            ("When is a super over played?", {}),
            ("Explain how a super over works.", {}),
        ],
    ),
    (
        "R3",
        "rules",
        None,
        None,
        {},
        [
            ("What is a dead ball?", {}),
            ("When does the umpire call dead ball?", {}),
            ("Explain the dead ball law.", {}),
        ],
    ),
    (
        "R4",
        "rules",
        None,
        None,
        {},
        [
            ("What is a stumping?", {}),
            ("How is a batter stumped?", {}),
            ("Explain stumpings in cricket.", {}),
        ],
    ),
    (
        "R5",
        "rules",
        None,
        None,
        {},
        [
            ("What is the Duckworth-Lewis method?", {}),
            ("How does the DLS method set a target?", {}),
            ("Explain the DLS rule for rain-shortened matches.", {}),
        ],
    ),
    (
        "R6",
        "rules",
        None,
        None,
        {},
        [
            ("What is a bouncer?", {}),
            ("How many bouncers may a bowler bowl in an over?", {}),
            ("Explain the bouncer law.", {}),
        ],
    ),
    (
        "C1",
        "cant_check",
        None,
        None,
        {"player": "JE Root"},
        [
            ("Is Joe Root injured?", {"player": "Joe Root"}),
            ("Does Joe Root have a back injury?", {"player": "Joe Root"}),
            ("Is Joe Root fit for the next Test?", {"player": "Joe Root"}),
        ],
    ),
    (
        "C2",
        "cant_check",
        None,
        None,
        {"player": "PHKD Mendis"},
        [
            ("How much is Kamindu Mendis paid?", {"player": "Kamindu Mendis"}),
            (
                "What does Kamindu Mendis earn from his central contract?",
                {"player": "Kamindu Mendis"},
            ),
            ("Kamindu Mendis's salary?", {"player": "Kamindu Mendis"}),
        ],
    ),
    (
        "C3",
        "cant_check",
        None,
        None,
        {"player": "Babar Azam"},
        [
            ("Will Babar Azam be made captain again?", {"player": "Babar Azam"}),
            (
                "Is Babar Azam going to get the captaincy back?",
                {"player": "Babar Azam"},
            ),
            ("Does the board want Babar Azam as captain?", {"player": "Babar Azam"}),
        ],
    ),
    (
        "C4",
        "cant_check",
        None,
        None,
        {"player": "RG Sharma"},
        [
            (
                "Is Rohit Sharma retiring after the World Cup?",
                {"player": "Rohit Sharma"},
            ),
            ("When will Rohit Sharma retire?", {"player": "Rohit Sharma"}),
            ("Has Rohit Sharma decided to quit Tests?", {"player": "Rohit Sharma"}),
        ],
    ),
    (
        "C5",
        "cant_check",
        None,
        None,
        {"player": "SPD Smith"},
        [
            (
                "Who is Steve Smith's best friend in the dressing room?",
                {"player": "Steve Smith"},
            ),
            ("Does Steve Smith get on with his captain?", {"player": "Steve Smith"}),
            ("What does Steve Smith do on rest days?", {"player": "Steve Smith"}),
        ],
    ),
    (
        "C6",
        "cant_check",
        None,
        None,
        {"player": "V Kohli", "match_format": "ODI"},
        [
            ("How many centuries has Kohli scored in ODIs?", {"player": "Kohli"}),
            ("Kohli's ODI hundreds: how many?", {"player": "Kohli"}),
            (
                "In ODIs, how many times has Virat Kohli made a hundred?",
                {"player": "Virat Kohli"},
            ),
        ],
    ),
]


def rows() -> list[dict]:
    # the perfect form names people as the question writes them, and every other
    # value as the subject has it
    found = []
    for group, kind, clip, measure, expected, wordings in GROUPS:
        for number, (question, said) in enumerate(wordings, start=1):
            subject = expected
            if subject is None:  # V5: the shots, in each wording's order
                subject = dict(said)
            form = {"kind": kind}
            for field, value in subject.items():
                if field in ("player", "bowler", "other_bowler", "gender"):
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
                    "measure": measure,
                    "form": form,
                    "expected": subject,
                }
            )
    return found


def main() -> None:
    written = rows()
    assert len(written) == 90, len(written)
    text = ""
    for row in written:
        text += json.dumps(row, ensure_ascii=False) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as out:
        out.write(text)
    print(f"wrote {len(written)} wordings to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
