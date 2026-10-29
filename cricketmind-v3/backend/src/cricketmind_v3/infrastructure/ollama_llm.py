"""The local language model, through Ollama (Day 57, D-024).

The question and the shape's JSON schema go in. Ollama makes the model write JSON that
fits the schema, and the reply is checked again here. Temperature 0 and a fixed seed, so a
question asked again should get the same answer. The schema fixes the reply's form, not
its truth: what the model says is checked after it (Day 58 on).
"""

import ollama
from pydantic import BaseModel

from cricketmind_v3.application.ports import LLMError, read_reply

SEED = 42
TIMEOUT = 300  # seconds: the first question also loads the model into memory


class OllamaLLM:
    def __init__(self, model: str, host: str | None = None, client=None) -> None:
        self.model = model  # "qwen2.5:7b", exactly as `ollama list` names it
        if client is None:
            client = ollama.Client(host=host, timeout=TIMEOUT)
        self.client = client  # a test can give a fake one

    def answer[Shape: BaseModel](
        self, system: str, prompt: str, shape: type[Shape]
    ) -> Shape:
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                format=shape.model_json_schema(),
                options={"temperature": 0, "seed": SEED},
                think=False,  # Qwen3 thinks aloud first; reading a question needs none
            )
        except (ollama.ResponseError, ConnectionError) as error:
            raise LLMError(
                f"Ollama could not answer ({self.model}): {error}"
            ) from error
        return read_reply(response.message.content, shape)
