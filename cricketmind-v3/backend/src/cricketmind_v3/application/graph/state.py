"""The graph's state: one shared notebook that every node reads and writes (Day 55).

Each key is a page. A node returns only the pages it changed, and LangGraph writes them in.
"""

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    question: str  # what the user typed
    clip: str | None  # a video clip's path, or None for a text-only question
    # every node signs here; nodes that run side by side sign at the same time, so the
    # reducer joins their lists instead of letting one overwrite the other
    visited: Annotated[list[str], operator.add]
    answer: str  # what N13 shows
