# CricketMind v3 — journal

One entry per day, at the end of the day. Five minutes. Copy the template.

```markdown
## Day N — YYYY-MM-DD — <title from the plan>

**Learned:** one or two sentences, in your own words.
**Built:** what exists now that did not this morning.
**Tests:** how many pass; anything skipped and why.
**Surprise:** one thing that did not go as expected.

**For the paper** (leave a line out if there is nothing):
- Decision: D-0xx in thesis/decisions.md
- Number: N-0xx in thesis/numbers.md
- Evidence: E-0xx in thesis/evidence.md
- Limitation: L-0xx in thesis/limitations.md
- Section it feeds: e.g. Method §3.4
**Commit:** <short hash>
```

---

## Day 0 — 2026-10-15/16 — Designing v3 (done with Claude, before the build starts)

**Learned:** The v2 Faith Check checks *values*, not *claims*. A correct number attached to the
wrong player passes it. Several v2 agents also say things no engine ever computed.

**Built:** The v3 architecture document (`docs/CricketMind_Architecture_v3.pdf`, copied on
Day 1) and this build plan.

**Surprise:** Two worked examples on the v2 synthetic corpus changed conclusions, not just
formatting:
- The running example ("the cover drive is the stronger stroke") has a difference interval
  that contains zero, so v3 would make the agent say "no detectable difference" (N-001, N-002).
- v2's N9 interval only measures Monte Carlo error. With estimation uncertainty it is about
  20 times wider (N-003, N-004).

**For the paper:**
- Decisions: D-001 to D-013 in thesis/decisions.md
- Numbers: N-001 to N-008 in thesis/numbers.md
- Evidence: E-001 to E-003 verified by reading v2 code; E-004 onwards planned
- Limitations: L-001 to L-012
- A first literature search found no 2025–26 multi-agent LLM system for cricket besides
  FanCric (2024). It also found close work on verifying numbers (Proof-Carrying Numbers,
  tool receipts), which the paper must position against (related_work.md).
- Sections it feeds: Introduction (the gaps), Method (all), Related work

---

## Day 1 — 2026-10-16 — The workshop: worktree, environment, first test

**Learned:** A git worktree is a second folder for the same repository with its own branch,
so CricShot and v3 never share a checkout and nobody switches branches. `uv run` always uses
the project's own .venv, so nothing needs activating.
**Built:** The cricketmind-v3 worktree and branch; the backend project (uv, Python 3.12,
pytest, ruff); pytest settings so CricShot's pytest.ini is never used; a .gitignore for v3;
the architecture PDF in docs/.
**Tests:** 1 passed (test_first.py); ruff clean.
**Surprise:** My default Python is 3.10, so uv init needed --python 3.12. Without the
[tool.pytest.ini_options] block, pytest would have climbed up and used CricShot's settings.

**For the paper:** nothing today (setup is process, not method).

---

## Day 2 — 2026-10-16 — Type hints and frozen dataclasses

**Learned:** Type hints are notes, not checks: Python happily made a Ball with runs="four".
`@dataclass` writes `__init__`, printing and `==` for me, and `frozen=True` makes a ball
read-only, which is what a checked claim needs (Day 10).
**Built:** `learning/day02_ball.py` (a frozen Ball with batter, runs, is_wicket);
`learning/pytest.ini` so learning tests never use CricShot's settings; 4 tests.
**Tests:** 4 passed; ruff clean. With `frozen=True` removed, the frozen test failed with
"DID NOT RAISE FrozenInstanceError", so the test really guards the rule.
**Surprise:** `Ball("V Kohli", "four")` ran without any error.

**For the paper:** nothing today.

---

## Plan note — 2026-10-16 — frame picking is part of the model card

Read CricShot's preprocessing (read only). It drops repeated frames, crops, and only then
samples 15. The v3 plan said "15 evenly spaced frames", which would feed the classifier
inputs it never trained on, with no error. Days 100, 102 and 104 (new numbering; 112, 114
and 116 in the old plan) now follow the model card and check frame numbers exactly. D-012
updated.

---

## Day 3 — 2026-10-16 — Pydantic: checking data at the door

**Learned:** A Pydantic model checks data when it arrives; a dataclass does not. Its error
names three things: which field, which rule broke, and what was sent (text, at least 5
characters, 'hi').
**Built:** `learning/day03_query.py` (QueryIn: text with min_length 5, optional player);
pydantic 2.14.1 added to the backend; 6 tests in `learning/test_day03_query.py` (written
by Claude after the plan rewrite below).
**Tests:** 6 new pass (15 in `learning/` with Days 2 and 4); ruff clean. With
`min_length=2`, `test_short_text_is_rejected` failed with "DID NOT RAISE", so it guards
the rule.
**Surprise:** pytest ran only 4 tests at first: the test file was open in VS Code but never
saved, so it was not on disk. Also, printing a Pydantic model shows only its fields.

**For the paper:** nothing today.

---

## Day 4 — 2026-10-16 — Clean architecture, small: the strike-rate toy

**Learned:** The use case is handed its repository (dependency injection) and only knows
the port, so a fake, a CSV reader or DuckDB can stand behind it without changing the
domain.
**Built:** `learning/toy/` with domain (Ball, strike_rate), port (BallRepository), use case
(GetStrikeRate) and fake (FakeBallRepository); 5 tests. Written by Claude and explained.
**Tests:** 5 new pass, 15 in `learning/`; ruff clean; no test reads a file; `domain.py`
imports only `dataclasses`. Without the wide check the strike rate came out 275.0 instead
of 366.7 and two tests failed, so the rule is guarded.
**Surprise:** only 9 tests ran at first, not 15, because Day 3's test file did not exist
yet.

Arrows (who imports whom); every one ends at domain.py:
use_cases.py -> ports.py -> domain.py
use_cases.py -> domain.py
fakes.py     -> domain.py

**For the paper:** nothing today.

---

## Plan rewrite — 2026-10-16 — learn on the real project

From now on Claude writes the code one small piece at a time and explains every line; I
read it, ask, and say "next". No more practice exercises: every practice-only day is
dropped, and its lesson is taught on the first real day that needs it. The plan went from
177 to 165 days. CLAUDE.md, the plan and the thesis files (outline, evidence, numbers,
README) use the new numbers. Next is Day 5, the real skeleton.

Old day numbers (in entries above this one) map to new ones like this:

| Old days | New days |
|---|---|
| 1–4 | 1–4 (done) |
| 5–7 | dropped: FastAPI and LangGraph toys, now taught on Days 46, 47, 62, 63, 69, 70 |
| 8–27 | 5–24 |
| 28 | dropped: SQL basics, now on Day 25 |
| 29–37 | 25–33 |
| 38 | dropped: rates and sample size, now on Day 34 |
| 39–44 | 34–39 |
| 45 | dropped: Monte Carlo and the Beta curve, now on Day 40 |
| 46–49 | 40–43 |
| 50 | dropped: calibration, now on Day 44 |
| 51–86 | 44–79 |
| 87–89, 91–92 | dropped: JavaScript, TypeScript, fetch, React state, now on Days 80, 82, 84 |
| 93 | 80 (frontend setup) |
| 90 | 81 (the first components, moved after setup) |
| 94–177 | 82–165 |
