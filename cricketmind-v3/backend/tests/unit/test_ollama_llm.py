"""Day 57: the Ollama adapter, with a fake client: what it sends, and what it does with
the reply. No model runs here."""

from types import SimpleNamespace

import ollama
import pytest
from pydantic import BaseModel

from cricketmind_v3.application.ports import LLMError
from cricketmind_v3.infrastructure.ollama_llm import SEED, OllamaLLM


class Reading(BaseModel):
    player: str | None
    balls: int | None


class FakeClient:
    """Stands in for ollama.Client: keeps each request, sends back a set reply."""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.requests = []

    def chat(self, **request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def test_what_it_sends():
    client = FakeClient('{"player": "V Kohli", "balls": 6}')
    llm = OllamaLLM("qwen2.5:7b", client=client)
    llm.answer("You read cricket questions.", "Kohli out within 6 balls?", Reading)
    assert client.requests == [
        {
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "system", "content": "You read cricket questions."},
                {"role": "user", "content": "Kohli out within 6 balls?"},
            ],
            "format": Reading.model_json_schema(),
            "options": {"temperature": 0, "seed": SEED},
            "think": False,
        }
    ]


def test_the_reply_comes_back_checked():
    llm = OllamaLLM(
        "qwen2.5:7b", client=FakeClient('{"player": "V Kohli", "balls": 6}')
    )
    assert llm.answer("s", "p", Reading) == Reading(player="V Kohli", balls=6)
    broken = OllamaLLM("qwen2.5:7b", client=FakeClient('{"player": "V Kohli"}'))
    with pytest.raises(LLMError, match="does not fit Reading"):
        broken.answer("s", "p", Reading)


def test_an_unknown_model_is_an_llm_error():
    client = FakeClient(error=ollama.ResponseError("model 'qwen9' not found", 404))
    with pytest.raises(LLMError, match=r"could not answer \(qwen9\)"):
        OllamaLLM("qwen9", client=client).answer("s", "p", Reading)


def test_no_server_is_an_llm_error():
    client = FakeClient(error=ConnectionError("Failed to connect to Ollama"))
    with pytest.raises(LLMError, match="could not answer"):
        OllamaLLM("qwen2.5:7b", client=client).answer("s", "p", Reading)
