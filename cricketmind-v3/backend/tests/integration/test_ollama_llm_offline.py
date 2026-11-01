"""Days 57 and 58: the real Ollama client, and LangChain, pointed at a port where
nothing listens: the error CricketMind gets when Ollama is not running. No model runs."""

import pytest
from pydantic import BaseModel

from cricketmind_v3.application.ports import LLMError
from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.infrastructure.ollama_llm import OllamaLLM


class Reading(BaseModel):
    player: str | None


def test_ollama_not_running_is_an_llm_error():
    llm = OllamaLLM("qwen2.5:7b", host="http://127.0.0.1:9")  # nothing listens on 9
    with pytest.raises(LLMError, match="could not answer"):
        llm.answer("system", "prompt", Reading)


def test_ollama_not_running_through_langchain_is_an_llm_error():
    llm = LangChainLLM("qwen2.5:7b", host="http://127.0.0.1:9")  # nothing listens on 9
    with pytest.raises(LLMError, match="could not answer"):
        llm.answer("system", "prompt", Reading)
