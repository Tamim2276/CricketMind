"""What every answer starts with (Day 59): "I read this as: ...", so a user sees a
misreading before trusting the answer. And the honest answer to a question nobody can
check: what it cannot say, and what can be asked instead.
"""

from cricketmind_v3.domain.reading import Reading

KIND_WORDS = {
    "stats": "a question about what happened",
    "prediction": "a prediction",
    "video": "a question about the clip",
    "rules": "a question about the rules",
    "cant_check": "a question no ball-by-ball record can answer",
}
PHASE_WORDS = {
    "powerplay": "the powerplay",
    "middle": "the middle overs",
    "death": "the death overs",
}
INNINGS_WORDS = {
    1: "batting first",
    2: "batting second",
    3: "in the third innings",
    4: "in the fourth innings",
}


def read_back(reading: Reading) -> str:
    """ "I read this as: a prediction: V Kohli, out within 6 balls, T20, the powerplay,
    against pace." Then what is asked back, and who is unknown."""
    subject = reading.subject
    shown = dict(reading.shown)  # a register id -> who he is
    parts = []
    if subject.player is not None:
        parts.append(shown.get(subject.player, subject.player))
    if subject.bowler is not None:
        bowler = shown.get(subject.bowler, subject.bowler)
        if subject.other_bowler is not None:
            other = shown.get(subject.other_bowler, subject.other_bowler)
            parts.append(f"facing {bowler} or {other}")
        else:
            parts.append(f"facing {bowler}")
    if subject.team is not None:
        parts.append(subject.team)
    if subject.opposition is not None:
        parts.append(f"against {subject.opposition}")
    if subject.venue is not None:
        parts.append(f"at {subject.venue}")
    if subject.shot is not None:
        if subject.other_shot is not None:
            parts.append(f"{subject.shot} or {subject.other_shot}")
        else:
            parts.append(subject.shot)
    if subject.k is not None:
        parts.append(f"out within {subject.k} balls")
    if subject.match_format is not None:
        parts.append(subject.match_format)
    if subject.gender == "female":
        parts.append("women's cricket")
    elif subject.gender == "male":
        parts.append("men's cricket")
    if subject.phase is not None:
        parts.append(PHASE_WORDS.get(subject.phase, subject.phase))
    if subject.bowler_type is not None:
        parts.append(f"against {subject.bowler_type}")
    if subject.innings is not None:
        parts.append(INNINGS_WORDS.get(subject.innings, f"innings {subject.innings}"))
    if subject.batting_hand is not None:
        parts.append(f"against {subject.batting_hand}-handers")
    if subject.season is not None:
        parts.append(f"the {subject.season} season")
    if subject.since is not None:
        parts.append(f"since {subject.since}")
    if subject.before is not None:
        parts.append(f"before {subject.before}")
    if subject.date is not None:
        parts.append(f"on {subject.date}")
    if subject.side is not None:
        parts.append(f"reviews by the {subject.side} side")

    lines = []
    if parts:
        lines.append(f"I read this as: {KIND_WORDS[reading.kind]}: {', '.join(parts)}.")
    else:
        lines.append(f"I read this as: {KIND_WORDS[reading.kind]}.")
    for unsure in reading.unsure:
        lines.append(
            f"Which {unsure.written}? {unsure.why} ({len(unsure.candidates)})."
        )
    for ask in reading.asks:
        lines.append(f"About {ask.written}: {ask.why}")
    for name in reading.unknown:
        lines.append(f"I do not know {name}.")
    return "\n".join(lines)


def cant_check_answer(reading: Reading) -> str:
    """What CricketMind cannot say, and what it can say instead."""
    lines = [
        (
            "I can't check that: CricketMind knows what happened on the field, ball by "
            "ball, not injuries, money, or what players think or feel."
        )
    ]
    subject = reading.subject
    who = subject.player or subject.bowler
    if who is not None:
        who = dict(reading.shown).get(who, who)
        lines.append(
            f"You could ask instead: {who}'s record, how many balls {who} has bowled "
            "in the last four weeks, or the chance of getting out in the next few balls."
        )
    else:
        lines.append(
            "You could ask instead about a player's record, a ground, or a team's results."
        )
    return " ".join(lines)
