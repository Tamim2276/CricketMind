# Day 59: the reader's fixes on 90 new wordings (D-080)

| reader | exactly right | stats | prediction | video | rules | cant_check | declined, of 18 | read by rules |
|---|---|---|---|---|---|---|---|---|
| rules (before) | 17 of 90 | 2 | 1 | 9 | 5 | 0 | 6 | 90 |
| LangChainLLM qwen2.5:7b (before) | 49 of 90 | 6 | 12 | 14 | 17 | 0 | 1 | 4 |

## What went wrong most often

- rules (before): player 54, kind 28, bowler 9, gender 9, opposition 6, venue 6
- LangChainLLM qwen2.5:7b (before): kind 22, player 9, opposition 4, team 4, k 3, other_shot 3

## Why a model's reply was not used (the reader fell back to the rules)

- LangChainLLM qwen2.5:7b (before), S1-1: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 1 validation error for QuestionForm)
- LangChainLLM qwen2.5:7b (before), S1-3: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 1 validation error for QuestionForm)
- LangChainLLM qwen2.5:7b (before), S4-3: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 4 validation errors for QuestionForm)
- LangChainLLM qwen2.5:7b (before), S6-2: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 4 validation errors for QuestionForm)
