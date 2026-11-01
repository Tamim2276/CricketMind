# Day 59: which door and which model read questions best (D-024, D-074)

90 practice wordings (30 questions, 6 of each kind, 3 ways each), each read once.
Exactly right: the kind and the whole subject (pre-registration, section 3).

| reader | exactly right | share |
|---|---|---|
| rules | 35 of 90 | 38.9% |
| OllamaLLM qwen2.5:7b | 58 of 90 | 64.4% |
| LangChainLLM qwen2.5:7b | 60 of 90 | 66.7% |
| LangChainLLM qwen3:8b | 43 of 90 | 47.8% |

- **The door that stays:** LangChainLLM (a tie keeps OllamaLLM).
- **The model that stays:** qwen2.5:7b (a tie keeps qwen2.5:7b), Ollama digest 845dbda0ea48.
- A gap of 9 wordings or fewer is within luck (section 5): the rule chose, but the
  paper claims no difference that small.

## What went wrong most often

- rules: player 44, kind 11, gender 9, bowler 8, k 6, team 6
- OllamaLLM qwen2.5:7b: kind 23, player 4, gender 3, opposition 3, since 3, k 2
- LangChainLLM qwen2.5:7b: kind 22, gender 3, opposition 3, player 3, since 3, k 2
- LangChainLLM qwen3:8b: kind 23, k 13, player 9, bowler 5, gender 3, opposition 3
