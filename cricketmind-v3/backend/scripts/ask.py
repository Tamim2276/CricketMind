"""Ask CricketMind one question in the terminal (Day 79, M3).

It refuses to run until the exam is locked (D-070). Run from backend/, with Ollama running
and no CricShot training on the GPU:
    uv run python scripts/ask.py ../data/cricket.duckdb qwen2.5:7b "What is LBW?"
"""

import sys

import duckdb

from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.pipeline.lock_exam import require_locked_exam
from cricketmind_v3.pipeline.real_runs import live_context


def main(database: str, model: str, question: str) -> None:
    require_locked_exam()
    db = duckdb.connect(database, read_only=True)
    context = live_context(db, LangChainLLM(model))  # the door that stayed (D-024)
    state = build_graph().invoke({"question": question, "clip": None}, context=context)
    print(state["answer"])
    db.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
