"""Day 78: ten runs of the whole graph with the real model, to fix the prompts from.

It refuses to run until the exam is locked (D-070). Run from backend/, with Ollama running
and no CricShot training on the GPU:
    uv run python scripts/real_runs.py ../data/cricket.duckdb qwen2.5:7b

Results go to thesis/results/day78_runs.jsonl as they come; run it again after a power cut
and it goes on from where it stopped. To run again after changing a prompt, give the
results another name as a third argument.
"""

import sys
from pathlib import Path

import duckdb

from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.pipeline.lock_exam import require_locked_exam
from cricketmind_v3.pipeline.real_runs import (
    live_context,
    read_questions,
    run_all,
    whole_graph,
)

THESIS = Path(__file__).resolve().parents[2] / "thesis"
QUESTIONS = THESIS / "practice" / "day78_questions.txt"


def main(database: str, model: str, name: str = "day78_runs") -> None:
    require_locked_exam()
    out = THESIS / "results" / f"{name}.jsonl"
    db = duckdb.connect(database, read_only=True)
    context = live_context(db, LangChainLLM(model))  # the door that stayed (D-024)

    def run_one(question: str) -> dict:
        return whole_graph(question, context)

    ran = run_all(read_questions(QUESTIONS), out, run_one)
    db.close()
    print(f"\n{ran} questions run. Results: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
