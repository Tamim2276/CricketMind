# Pre-registration: which door and which model read questions best (Day 59)

Written 2026-11-01, before any model reads the practice set. Decisions D-024 and D-074.
The scoring script checks this file's SHA-256 and refuses to run if it has changed.

## 1. The questions

`thesis/practice/day59_readings.jsonl`, SHA-256
`ff50793238791582acb04b299e3ba8ef9dec58cdcc30399d208733b3227f00d6`: 30 practice
questions, 6 of each kind (stats, prediction, video, rules, cant_check), each asked 3 ways:
90 wordings. Written by Claude for practice, never the locked exam's. Each has its
expected reading: the kind, and every field of the subject, with nothing else filled. A
perfect form gives every expected reading through the real reader and the real registers:
checked before this file was written, 90 of 90.

## 2. The readers and the order

1. **The adapters**, both with `qwen2.5:7b`: `OllamaLLM` and `LangChainLLM`.
2. **The models**, with the adapter that stays: `qwen2.5:7b` and `qwen3:8b`.

The same prompt, form, registers and checks for all (`read_question`, at the commit the
ledger names). Temperature 0, seed 42, no thinking. Each wording is read once. The rules
reader (no model) is scored beside them as a baseline; it decides nothing.

## 3. The score

The share of the 90 wordings read **exactly right**: the reading's kind equals the
expected kind, and its subject equals the expected subject (every field, nothing more). A
question asked back instead of read (several players, a misspelling) is not exactly right.

## 4. The rules

- **Adapters:** the higher score stays. A tie keeps `OllamaLLM`, the simpler one.
- **Models:** the higher score stays. A tie keeps `qwen2.5:7b`, the smaller one.
- No other comparison decides anything. The share of each kind read right, and which
  fields go wrong, are reported to explain the scores, not to choose.
- The model that stays is recorded with its Ollama digest. The other may then be removed.

## 5. What could make this unfair, said now

- Claude wrote the questions and knows the system: the expected readings are fixed here
  before any model reads them, and the set is not the exam.
- 90 wordings of 30 questions are not independent: wordings of one question fail together.
  With so few, a difference of a wording or two is within luck; the rule still decides,
  because some reader must be used, but the paper will not claim one reader is better
  unless the gap is wide (more than 9 wordings, a tenth).
