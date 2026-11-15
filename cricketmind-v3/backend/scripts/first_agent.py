"""Day 67: the first real agent: Tactical Analysis (N11b), with the real model.

It refuses to run until the exam is locked (D-070). Run from backend/, with Ollama running
and no CricShot training on the GPU:
    uv run python scripts/first_agent.py ../data/cricket.duckdb qwen2.5:7b

Results go to thesis/results/day67_first_agent.jsonl as they come; run it again after a
power cut and it goes on from where it stopped. Done when 5 real findings pass C1-C7.
"""

import json
import sys
from pathlib import Path

import duckdb

from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.pipeline.lock_exam import require_locked_exam
from cricketmind_v3.pipeline.real_runs import (
    live_context,
    one_agent,
    read_questions,
    run_all,
)

THESIS = Path(__file__).resolve().parents[2] / "thesis"
QUESTIONS = THESIS / "practice" / "day67_questions.txt"
OUT = THESIS / "results" / "day67_first_agent.jsonl"


def main(database: str, model: str) -> None:
    require_locked_exam()
    db = duckdb.connect(database, read_only=True)
    context = live_context(db, LangChainLLM(model))  # the door that stayed (D-024)

    def run_one(question: str) -> dict:
        return one_agent("N11b", question, context)

    run_all(read_questions(QUESTIONS), OUT, run_one)
    db.close()

    passed = 0
    rows = OUT.read_text(encoding="utf-8").splitlines()
    for line in rows:
        if json.loads(line)["passed"]:
            passed += 1
    print(f"\n{passed} of {len(rows)} findings passed C1-C7. Results: {OUT}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
