# Pre-registration: do the reader's fixes help, on questions they never saw? (Day 59)

Written 2026-11-01, before the fixes are coded and before either reader reads this set.
The scoring script checks this file's SHA-256 and refuses to run if it has changed.

## 1. Why

On Day 59's first set (N-044) the reader that stayed, LangChainLLM with qwen2.5:7b, read
60 of 90 wordings exactly right. Most misses were the kind: of 18 questions nobody can
check, it read 14 as stats and 3 as predictions (L-072). The fixes (D-080) were chosen
after seeing that set, so that set cannot judge them. This one can.

## 2. The questions

`thesis/practice/day59b_readings.jsonl`, SHA-256
`956994847959b1d51646b36b8196e8b61f9e910022e2448a22867156be5cafec`: 30 new questions, 6 of
each kind, 3 wordings each (90), none of them from the first set, written by Claude before
the fixes were coded, never the exam's. A perfect form gives all 90 expected readings
through the current reader and the real registers (checked before this file). One group
asks for a number no engine counts (Kohli's ODI centuries): its expected kind is
cant_check, because CricketMind cannot check it.

## 3. The two readers

1. **Before:** the reader at the commit that adds this file, run before any fix is coded.
2. **After:** the reader with the fixes (D-080), at the commit the ledger names.

Both are LangChainLLM with qwen2.5:7b (D-074), temperature 0, seed 42, no thinking, the
same registers. Each wording is read once. The rules reader is scored beside them,
deciding nothing.

## 4. The score and the rule

- **Score:** the share of the 90 read exactly right: the kind equals the expected kind,
  and the subject equals the expected subject (every field, nothing more), as on the
  first set.
- **Rule:** the fixed reader stays if it reads more wordings exactly right than the
  reader before; otherwise the reader before stays, and the fixes are taken out.
- **Luck:** as before, a gap of 9 wordings or fewer is not claimed as a difference, though
  the rule still decides.
- **Reported, deciding nothing:** the score on each kind; how many of the 18 cant_check
  wordings each reader declines; for the fixed reader, how often its measure is the
  expected one.

## 5. What could make this unfair, said now

- Claude wrote these questions knowing what the fixes will be. They are written in the
  same plain styles as the first set and fixed here before the fixes exist, and they
  include kinds and fields the fixes do not touch; the exam (Day 178), which nobody here
  has seen, remains the real test.
- The off-menu group (3 wordings) is expected to be declined, which the reader before was
  never built to do: it is reported on its own as well, so the comparison can be read
  with it and without it.
