"""A judgment hidden in a question (Day 76, D-022): "Why is Kohli bad against spin?"

N3 marks it with C7's own words, so a question and an answer are held to one list. It is a
plain look at the words, with no model: adding a box to the model's form made its reading
worse once (D-080). A judgment in other words is missed (L-084).
"""

from dataclasses import dataclass

from cricketmind_v3.domain.faith.c7_tone import DOWN, UP, negated
from cricketmind_v3.domain.faith.text import find_phrases, tokens

PREMISE_AGENT = "N3"  # N3 marks it, and its test is filed under N3


@dataclass(frozen=True, kw_only=True)
class Premise:
    word: str  # "bad", the word in the question
    better: bool  # does it say the player does better? "not good" says worse


def hidden_judgment(question: str) -> Premise | None:
    """The first judgment in the question, or None."""
    words = tokens(question)
    for phrase, start in find_phrases(words, UP + DOWN):
        better = phrase in UP
        if negated(words, start):
            better = not better
        return Premise(word=phrase, better=better)
    return None
