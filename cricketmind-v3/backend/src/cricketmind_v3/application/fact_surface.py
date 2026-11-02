"""N8 and N9: every number the engines can give for a question (Day 61).

The numbers are worked out here, once, and written in the notebook. Agents read that
list; they never ask an engine, and they never make a number (invariant I1).
"""

import dataclasses

from cricketmind_v3.application.ports import FactSource
from cricketmind_v3.domain.evidence import sides
from cricketmind_v3.domain.facts import Answer
from cricketmind_v3.domain.registry import CLAIM_TYPES, fits
from cricketmind_v3.domain.subject import CricketSubject


def ways_to_ask(subject: CricketSubject) -> list[CricketSubject]:
    """The whole question, and each side of it, with k and without.

    "Kohli against Bumrah or Starc, within 12 balls" is a matchup as a whole, and it also
    holds Kohli against Bumrah and Kohli against Starc, past (no k) and future (with k).
    """
    wanted = [subject, dataclasses.replace(subject, k=None)]
    for side in sides(subject):
        wanted.append(side)
        wanted.append(dataclasses.replace(side, k=subject.k))
    found = []
    for each in wanted:
        if each not in found:
            found.append(each)
    return found


def ask_engines(
    subject: CricketSubject, kind: str, engine: str, source: FactSource | None
) -> list[Answer]:
    """Ask this engine ("N8" or "N9") for every menu row that fits, the same rule the
    engines themselves apply. No answer is kept too: the menu allowed it, and there was
    nothing to count."""
    if source is None or kind == "rules":
        return []  # a rule is explained in words, from no numbers
    answers = []
    for asked in ways_to_ask(subject):
        for row in CLAIM_TYPES:
            if row.source.split(".")[0] != engine:
                continue
            if fits(row, asked):
                fact = source.fact(row.name, asked)
                answers.append(Answer(metric=row.name, subject=asked, fact=fact))
    return answers


def surface(answers: list[Answer]) -> tuple[float, ...]:
    """Every number the engines gave for this question: all v2's gate can look at."""
    numbers = []
    for answer in answers:
        if answer.fact is not None:
            numbers.append(answer.fact.value)
    return tuple(numbers)
