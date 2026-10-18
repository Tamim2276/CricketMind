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

---

## Day 5 — 2026-10-16 — The real skeleton, guarded by a test

**v3 in 10 lines** (written by Claude at my request, after we read the architecture
document together, section by section):

1. v2's Faith Check only asked "did an engine produce this number?". So a right number
   about the wrong player passed, typed numbers with no source passed, four agents said
   things no engine measures, predictions hid their real uncertainty, and all data was fake.
2. A claim is one fact as a form: kind, metric, engine, subject, value, sample size (and a
   range for predictions). As data, it can be checked by asking the engine again; a
   sentence can only be read.
3. M = measured from the clip (N6); D = descriptive, counted from past matches (N8);
   P = predictive, simulated (N9).
4. Subject-keyed: the claim says exactly who and what the number is about (Kohli, Cover
   Drive, powerplay, pace). If Kohli's 142.0 is written as Rohit's, the engine asked about
   Rohit gives a different number, so the lie is caught.
5. C1: does it fit the form, is the claim type on the menu, may this agent use it?
   C2: same value when the engine is asked again for this subject? C3: at least 30 balls,
   or does it say "small sample"? C4: do the words fit the chance, and does a comparison
   whose range includes 0 say "no detectable difference"? C5: no typed digits?
   C6: do the names in the sentence match the claim's subject?
6. F6, spurious difference: v2 let "the cover drive is the stronger stroke" through, though
   the difference range (−16 to +76) includes 0.
7. N8 counts the past (strike rates, venue, workload, DRS); N9 simulates the future (chance
   of being out, what-if matchups) with honest ranges.
8. An invariant is a rule that is always true, everywhere. Example: I3, agents never type
   numbers; they write {c1} and the renderer fills in the checked value.
9. It claims nothing about real players: 142 vs 112 and 0.405 come from synthetic data.
10. Least clear (Claude's guess, change it if something else is): why the honest range for
    0.405 (0.155 to 0.697) is about 20 times wider than v2's: v2 counted only simulation
    noise, not how little data 50 balls is.

**Learned:** The five layers are rooms with borrowing rules: `domain/` (the rules of v3)
may only use plain Python, and `application/` may never reach into the suppliers. A test
can read code as a tree (`ast`) without running it.
**Built:** `backend/src/cricketmind_v3/` with `domain/`, `application/`,
`infrastructure/`, `pipeline/`, `api/`; `tests/__init__.py`, `tests/unit/`;
`tests/unit/test_architecture.py` with three tests. Written by Claude and explained.
**Tests:** 4 passed (3 new + test_first); ruff clean; learning still 15 passed. Broken on
purpose twice: `import fastapi` in `domain/oops.py` failed with
`['oops.py imports fastapi']`; an `application/` file importing
`cricketmind_v3.infrastructure` failed too. Both passed again once removed.
**Surprise:** the test caught `import fastapi` although FastAPI is not even installed,
because it only reads the import lines and never runs them.

**For the paper:**
- Decision: D-014 in thesis/decisions.md (the layer rule, enforced by a test)
- Section it feeds: Implementation

---

## Day 6 — 2026-10-16 — Provenance and Subject

**Learned:** Every number needs two labels: how we know it (Provenance: M, D or P, a fixed
list, so "guessed" or a typo crashes at once) and who and what it is about (Subject: player,
shot, phase, bowler type; an empty field means "any"). Keyword-only fields stop player and
shot being swapped by accident.
**Built:** `domain/provenance.py` (the Provenance enum) and `domain/subject.py` (a frozen,
keyword-only Subject with four fields; Day 10 adds the rest). 9 tests in
`tests/unit/test_provenance.py` and `tests/unit/test_subject.py`. Written by Claude and
explained. The plan's Phase 1 was also written out day by day.
**Tests:** 13 passed (9 new); ruff clean; the architecture test still passes. With
`kw_only=True` removed, `Subject("Pull", "V Kohli")` made a subject with player "Pull" and
shot "V Kohli" with no error, and `test_fields_must_be_named` failed with "DID NOT RAISE
TypeError". Put back, all pass.
**Surprise:** ruff now flags two things the old defaults did not: `getattr` with a fixed
name, and a line that does nothing (it cannot know the line is meant to crash). Also, Python
rounds 0.405 to 40% but 0.155 to 16% (exact halves go to the even number), noted for Day 9.

**For the paper:** nothing today.

---

## Day 7 — 2026-10-16 — Claim: one fact as data

**Learned:** A claim is one fact as a form: id, kind, metric, subject, value, plus n for a
count, a range for a prediction, or a clip for a measurement. `__post_init__` runs right
after the object is made, so the form can check itself and refuse to exist if a rule is
broken. `dataclasses.replace` copies a claim with one field changed, which makes it easy to
break exactly one rule per test.
**Built:** `domain/claims.py`: `ClaimError` and a frozen, keyword-only `Claim` with four
rules (id like `c1`; D needs n ≥ 1; P needs a range that contains its value; M needs a
clip). 10 tests in `tests/unit/test_claims.py`, including the v3 document's c1 and c2.
Written by Claude and explained.
**Tests:** 23 passed (10 new); ruff clean; architecture test still passes. Before the rules
existed, a prediction with no range and the id "claim1" was accepted without complaint.
With the range check removed on purpose, a 0.9 prediction with range 0.155 to 0.697 was
accepted and `test_rule_3_the_value_must_sit_inside_its_range` failed; put back, all pass.
**Surprise:** without the rules, Python happily built a claim that contradicts its own range.
ruff asked to join nested `if`s with `and`.

**For the paper:**
- Decision: D-015 in thesis/decisions.md (a claim checks itself; the band is derived, not
  stored)
- Section it feeds: Method (claim contract)

---

## Day 8 — 2026-10-17 — Finding: claims plus a sentence with slots

**Learned:** What an agent hands in is a Finding: who wrote it, its claims, and one sentence
with slots like `{c1}` instead of numbers. Slots are found with plain string tools in a
loop (`find` the `{`, `find` its `}`, take the text between). A tuple cannot grow, so a
checked finding cannot have a claim slipped in later.
**Built:** `domain/findings.py`: `FindingError`, `references()` (the slots in order; an
unclosed `{` is refused) and a frozen, keyword-only `Finding` with three rules (claims are a
tuple; no two claims share an id; every slot names one of its claims). The shared example
claims moved to `tests/examples.py`, so `test_claims.py` and `test_findings.py` use one copy.
10 new tests. Written by Claude and explained.
**Tests:** 33 passed (10 new); ruff clean; learning 15. With rule 3 removed on purpose, a
sentence using `{c3}` with no claim c3 was accepted, and two tests failed: the missing-claim
test and the unclosed-slot test (without rule 3 the finding never reads its slots at all).
Put back, all pass.
**Surprise:** one removed rule broke two tests, because rule 3 was also the only place the
finding looked at its sentence.

**For the paper:** nothing new today (Day 8 builds part of D-004, the claim contract).

---

## Day 9 — 2026-10-17 — The renderer: from slots to sentences

**Learned:** The agent writes the words; plain code puts the numbers in, copied from the
checked claim, so the language model never gets the chance to make one up. Python rounds an
exact half to the even number (0.405 → 40%), and a computer stores 0.145 as slightly less,
so 0.145 × 100 = 14.4999…; the `decimal` module does the sum the way it is done on paper.
**Built:** `domain/render.py`: `fixed()` and `percent()` (halves always round up, in
decimal), a small table of words per metric, `render_claim()` (words plus the tag
`[D, 50 balls]`, `[P, range 16% to 70%]` or `[M, clip vid1_0]`) and `render()` (a finding's
sentence with every slot filled). 10 tests in `tests/unit/test_render.py`. The plan's Day 9
section now records the rounding decision. Written by Claude and explained.
**Tests:** 43 passed (10 new); ruff clean (after two automatic tidy-ups). Switched to
half-even on purpose: three tests failed and the user would have seen "a 40% chance";
switched back, all pass. The document's finding now renders as: "…returns a strike rate of
142.0 [D, 50 balls]. Six short balls carry a 41% chance [P, range 16% to 70%] of a dismissal."
**Surprise:** 0.145 × 100 is 14.499999999999998 in a computer, so the fix had to be decimal
arithmetic, not just a different rounding rule.

**For the paper:**
- Decision: D-016 in thesis/decisions.md (halves round up, in decimal; the claim tags)
- Section it feeds: Implementation (rendering)

---

## Plan extension — 2026-10-17 — every question gets an honest answer; tone words checked

Asked: can the tone gap (L-013) be closed, and can the agent answer any cricket question,
however it is worded? Two changes, eight new days (165 → 173):

- **C7 (D-017):** a tone word ("strong", "struggles") is a hidden comparison, so it needs a
  checked comparison claim pointing the same way. New lie type F8. L-013 narrowed.
- **Honest answers (D-018):** N3 sorts each question into stats, prediction, video, rules or
  can't-check, and every answer starts "I read this as: …". Stats questions become one engine
  query with any filters and one measure, so unlisted questions still get checked numbers.
  The Explainer answers rules in words only, labelled "not checked". The rest get "I can't
  check that". Phase 10 tests 230 questions, including 5 wordings of the same question.
  New limitations L-014 (explanations unchecked) and L-015 (N3 can misread a question).

Day numbers in entries above use the 165-day plan for days after 21. They map like this:

| 165-day plan | Now |
|---|---|
| 1–21 | 1–21 |
| — | 22 (new: C7, tone words) |
| 22–38 | 23–39 |
| — | 40–42 (new: any stats question) |
| 39–49 | 43–53 |
| — | 54 (new: sorting questions by kind) |
| 50–61 | 55–66 |
| — | 67 (new: the Explainer) |
| 62–137 | 68–143 |
| — | 144–145 (new: coverage) |
| 138–165 | 146–173 |

---

## Plan extension (journal) — 2026-10-19 — an evaluation built for a journal

The goal is a journal paper, so the evaluation must answer what reviewers ask (D-019). Ten
days added (173 → 183):

- **Days 13–15:** a systematic literature search; the journal chosen with my supervisor (and
  ethics approval asked for, if the human study needs it); the locked final exam: 100
  questions I write (ideally with friends) before the system can answer anything,
  fingerprinted with SHA-256, run once on Day 149. Claude never reads it before then.
- **Phase 10:** the ablation (each check off in turn) on the fabrication day; a practice set
  to improve the system, never reported; the final exam run once; two baselines (a plain AI,
  and an AI with the engine's facts but no checks); a small blind human study.
- New evidence rows E-014 (baselines), E-015 (human study), E-016 (ablation); E-013 now points
  at the locked exam; L-012 narrowed.

Day numbers in entries above use the 173-day plan for days after 12. They map like this:

| 173-day plan | Now |
|---|---|
| 1–12 | 1–12 |
| — | 13–15 (new: literature search, journal choice, locked final exam) |
| 13–145 | 16–148 |
| — | 149–155 (new: final exam run, baselines, human study) |
| 146–173 | 156–183 |
