"""The same local model, through LangChain (Day 58, part 3; D-024).

A second adapter behind the same `LLM` port, so that Day 59 can compare the two on the
same questions with the same model, prompt and form. LangChain's own way of asking for a
form (`with_structured_output`) sends Ollama the form's JSON schema, with the same
settings as OllamaLLM (checked on Day 58: temperature 0, seed 42, no thinking); its raw
reply is kept and checked here exactly as OllamaLLM's is, so neither can skip the check.
"""

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from cricketmind_v3.application.ports import LLMError, read_reply
from cricketmind_v3.infrastructure.ollama_llm import NO_ANSWER, SEED, TIMEOUT


class LangChainLLM:
    def __init__(self, model: str, host: str | None = None, chat=None) -> None:
        self.model = model  # "qwen2.5:7b", exactly as `ollama list` names it
        if chat is None:
            # the same settings as OllamaLLM: no randomness, the same seed, no thinking
            chat = ChatOllama(
                model=model,
                base_url=host,
                temperature=0,
                seed=SEED,
                reasoning=False,
                client_kwargs={"timeout": TIMEOUT},
            )
        self.chat = chat  # a test can give a fake one

    def answer[Shape: BaseModel](
        self, system: str, prompt: str, shape: type[Shape]
    ) -> Shape:
        asking = self.chat.with_structured_output(
            shape.model_json_schema(), method="json_schema", include_raw=True
        )
        try:
            result = asking.invoke([("system", system), ("human", prompt)])
        # LangChain streams, so a refused connection comes as httpx's own error
        except NO_ANSWER as error:
            raise LLMError(
                f"LangChain could not answer ({self.model}): {error}"
            ) from error
        return read_reply(result["raw"].content, shape)
