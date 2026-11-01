# Day 59: the reader's fixes on 90 new wordings (D-080)

| reader | exactly right | stats | prediction | video | rules | cant_check | declined, of 18 | read by rules |
|---|---|---|---|---|---|---|---|---|
| rules (after) | 25 of 90 | 3 | 2 | 12 | 8 | 0 | 6 | 90 |
| LangChainLLM qwen2.5:7b (after) | 43 of 90 | 6 | 0 | 13 | 16 | 8 | 15 | 4 |

## What went wrong most often

- rules (after): player 38, kind 25, bowler 9, gender 9, opposition 6, venue 6
- LangChainLLM qwen2.5:7b (after): kind 24, player 10, other_shot 5, opposition 4, team 4, match_format 3

## Why a model's reply was not used (the reader fell back to the rules)

- LangChainLLM qwen2.5:7b (after), S1-1: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 1 validation error for QuestionForm / season /   String should match pattern '^\d{4}(/\d{2})?$' [type=string_pattern_mismatch, input_value='since 2021', input_type=str] /     For further information visit https:
- LangChainLLM qwen2.5:7b (after), P6-2: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 4 validation errors for QuestionForm / season /   String should match pattern '^\d{4}(/\d{2})?$' [type=string_pattern_mismatch, input_value='null', input_type=str] /     For further information visit https://err
- LangChainLLM qwen2.5:7b (after), P6-3: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 4 validation errors for QuestionForm / season /   String should match pattern '^\d{4}(/\d{2})?$' [type=string_pattern_mismatch, input_value='null', input_type=str] /     For further information visit https://err
- LangChainLLM qwen2.5:7b (after), C5-1: read by rules: the model's reply did not fit twice (the reply does not fit QuestionForm: 2 validation errors for QuestionForm / since /   String should match pattern '^\d{4}(-\d{2}-\d{2})?$' [type=string_pattern_mismatch, input_value='null', input_type=str] /     For further information visit https:
