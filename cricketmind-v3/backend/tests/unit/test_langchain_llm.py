"""Day 58, part 3: the LangChain adapter, with a fake chat model: what it asks for, and
that its reply is checked like OllamaLLM's. No model runs here."""

from types import SimpleNamespace

import httpx
import ollama
import pytest
from pydantic import BaseModel

from cricketmind_v3.application.ports import LLMError
from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.infrastructure.ollama_llm import SEED, TIMEOUT


class Reading(BaseModel):
    player: str | None
    balls: int | None


class FakeChat:
    """Stands in for ChatOllama: keeps what it was asked, sends back a set reply."""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.forms = []
        self.messages = []

    def with_structured_output(self, schema, method, include_raw):
        self.forms.append((schema, method, include_raw))
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        if self.error is not None:
            raise self.error
        return {"raw": SimpleNamespace(content=self.content), "parsed": None}


def test_it_asks_for_the_same_form_and_words():
    chat = FakeChat('{"player": "V Kohli", "balls": 6}')
    llm = LangChainLLM("qwen2.5:7b", chat=chat)
    llm.answer("You read cricket questions.", "Kohli out within 6 balls?", Reading)
    assert chat.forms == [(Reading.model_json_schema(), "json_schema", True)]
    assert chat.messages == [
        [
            ("system", "You read cricket questions."),
            ("human", "Kohli out within 6 balls?"),
        ]
    ]


def test_its_raw_reply_is_checked_like_ollamas():
    llm = LangChainLLM("qwen2.5:7b", chat=FakeChat('{"player": "V Kohli", "balls": 6}'))
    assert llm.answer("s", "p", Reading) == Reading(player="V Kohli", balls=6)
    broken = LangChainLLM("qwen2.5:7b", chat=FakeChat('{"player": "V Kohli"}'))
    with pytest.raises(LLMError, match="does not fit Reading"):
        broken.answer("s", "p", Reading)


def test_no_such_model_and_no_server_are_llm_errors():
    missing = FakeChat(error=ollama.ResponseError("model 'qwen9' not found", 404))
    with pytest.raises(LLMError, match=r"could not answer \(qwen9\)"):
        LangChainLLM("qwen9", chat=missing).answer("s", "p", Reading)
    down = FakeChat(error=ConnectionError("Failed to connect to Ollama"))
    with pytest.raises(LLMError, match="could not answer"):
        LangChainLLM("qwen2.5:7b", chat=down).answer("s", "p", Reading)


def test_a_refused_connection_while_streaming_is_an_llm_error():
    # found on Day 58: LangChain streams, and Ollama not running is httpx's ConnectError
    refused = FakeChat(error=httpx.ConnectError("actively refused"))
    with pytest.raises(LLMError, match="could not answer"):
        LangChainLLM("qwen2.5:7b", chat=refused).answer("s", "p", Reading)


def test_its_own_chat_model_has_ollamas_settings():
    llm = LangChainLLM("qwen2.5:7b")
    assert llm.chat.model == "qwen2.5:7b"
    assert (llm.chat.temperature, llm.chat.seed, llm.chat.reasoning) == (0, SEED, False)
    assert llm.chat.client_kwargs == {"timeout": TIMEOUT}
