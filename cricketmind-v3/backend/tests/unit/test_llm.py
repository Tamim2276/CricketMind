"""Day 57: the LLM port: how a reply is checked, and the fake model for tests."""

import pytest
from pydantic import BaseModel, ConfigDict

from cricketmind_v3.application.ports import FakeLLM, LLMError, read_reply


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a field nobody asked for is refused

    player: str | None
    balls: int | None


def test_a_reply_that_fits():
    assert read_reply('{"player": "V Kohli", "balls": 6}', Reading) == Reading(
        player="V Kohli", balls=6
    )
    assert read_reply('{"player": null, "balls": null}', Reading) == Reading(
        player=None, balls=None
    )


def test_a_reply_that_does_not_fit_is_refused():
    bad = [
        "",
        "Sure! Kohli, 6 balls.",  # words, not JSON
        '{"player": "V Kohli"}',  # a field missing
        '{"player": "V Kohli", "balls": "six"}',  # the wrong type
        '{"player": "V Kohli", "balls": 6, "chance": 0.4}',  # a number nobody asked for
    ]
    for reply in bad:
        with pytest.raises(LLMError, match="does not fit Reading"):
            read_reply(reply, Reading)


def test_the_fake_gives_its_replies_in_order():
    llm = FakeLLM(
        ['{"player": "V Kohli", "balls": 6}', '{"player": null, "balls": 12}']
    )
    assert llm.answer("You read questions.", "Kohli in 6?", Reading).balls == 6
    assert llm.answer("You read questions.", "12 balls?", Reading).balls == 12
    assert llm.asked == [
        ("You read questions.", "Kohli in 6?", "Reading"),
        ("You read questions.", "12 balls?", "Reading"),
    ]


def test_the_fake_checks_its_replies_like_a_real_model():
    llm = FakeLLM(['{"player": "V Kohli"}'])
    with pytest.raises(LLMError, match="does not fit"):
        llm.answer("system", "prompt", Reading)


def test_the_fake_can_fail_on_purpose():
    llm = FakeLLM([LLMError("Ollama is not running")])
    with pytest.raises(LLMError, match="not running"):
        llm.answer("system", "prompt", Reading)


def test_the_fake_runs_out():
    llm = FakeLLM([])
    with pytest.raises(LLMError, match="no reply left"):
        llm.answer("system", "prompt", Reading)
