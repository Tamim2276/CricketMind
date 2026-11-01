"""The graph's state: one shared notebook that every node reads and writes (Day 55).

Each key is a page. A node returns only the pages it changed, and LangGraph writes them in.
"""

import operator
from typing import Annotated, TypedDict

from cricketmind_v3.domain.evidence import Evidence
from cricketmind_v3.domain.question_context import QuestionContext
from cricketmind_v3.domain.reading import Reading
from cricketmind_v3.domain.shot_dna import ShotDNA
from cricketmind_v3.domain.subject import CricketSubject


class GraphState(TypedDict, total=False):
    question: str  # what the user typed
    clip: str | None  # a video clip's path, or None for a text-only question
    # every node signs here; nodes that run side by side sign at the same time, so the
    # reducer joins their lists instead of letting one overwrite the other
    visited: Annotated[list[str], operator.add]
    subject: CricketSubject  # who and what the question is about (N3, Day 56)
    reading: Reading  # the subject, and what N3 could not put in it (Day 58)
    shot_dna: ShotDNA  # the shot asked about, and what the clip showed (N6, Day 60)
    # the balls of each side of the question, filed under it (N7a, Day 60)
    evidence: tuple[Evidence, ...]
    context: QuestionContext  # who the people are, where the ground is (N7b, Day 60)
    answer: str  # what N13 shows
