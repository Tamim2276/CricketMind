"""Ask a local model to read a few questions into a fixed shape (Day 57, by hand).

Run from backend/, with Ollama running and no CricShot training on the GPU:
    uv run python scripts/ask_llm.py qwen2.5:7b
    uv run python scripts/ask_llm.py qwen3:8b

The shape here is a small one, only for this try; N3's real one comes on Day 58.
"""

import sys
import time

from pydantic import BaseModel, ConfigDict

from cricketmind_v3.application.ports import LLMError
from cricketmind_v3.infrastructure.ollama_llm import OllamaLLM


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player: str | None  # the batter, as the question names him
    bowler: str | None  # the bowler, if the question names one
    balls: int | None  # "within 6 balls" -> 6
    match_format: str | None  # "T20", "ODI", "Test"


SYSTEM = (
    "You read cricket questions. Fill in only what the question itself says. "
    "Use null for anything it does not say. Never guess, never add a number."
)
QUESTIONS = [
    "Kohli out within 6 balls in the T20 powerplay against pace?",
    "What is Babar Azam's strike rate in ODIs?",
    "Would Bumrah or Starc get Smith out sooner in Tests?",
    "What is LBW?",
]


def main(model: str) -> None:
    llm = OllamaLLM(model)
    for question in QUESTIONS:
        started = time.time()
        try:
            reading = llm.answer(SYSTEM, question, Reading)
        except LLMError as error:
            print(f"FAILED  {question}\n        {error}")
            continue
        print(f"{time.time() - started:5.1f}s  {question}\n        {reading}")


if __name__ == "__main__":
    main(sys.argv[1])
