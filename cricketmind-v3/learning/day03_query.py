from pydantic import BaseModel, Field


class QueryIn(BaseModel):
    text: str = Field(min_length=5)
    player: str | None = None


if __name__ == "__main__":
    print(QueryIn(text="Kohli vs pace?", player="V Kohli"))
    print(QueryIn.model_validate_json('{"text": "Who bowls best at the death?"}'))
    QueryIn(text="hi")  # too short, so this line crashes
