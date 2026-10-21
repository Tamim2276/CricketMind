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

---

## Plan extension (Q1) — 2026-10-19 — aiming for a Q1 journal

Searched for more datasets and the newest related work. Two findings changed the plan:

- **The idea alone is no longer new.** In 2026, EG-VAR, VeriFin and tool receipts (NabaOS)
  check AI numbers against tool outputs. The paper's new parts must be: who a number is about
  (C2 + C6), honest uncertainty in words (C4), tone words (C7), the start-up menu check, and
  honest, measured coverage. Day 13 checks this against the new papers (added to
  `related_work.md`).
- **CricBench** (633 English cricket questions, built on Cricsheet) would be a strong outside
  test set, but its data is only promised "upon acceptance". Ask the authors on Day 14.

Nineteen days added (183 → 202), decisions D-020 to D-023:
- Receipts for every number, and a receipt checker (Days 47–48).
- False premises and trick attempts (Day 76; 20 such questions join the Day 15 exam).
- A football case study (Phase 9b, Days 143–149); `Subject` becomes sport-neutral on Day 10.
- AI checkers vs the Faith Check (Days 152–154); three models (Days 166–167); the tricky
  questions (Day 168); CricBench (Days 169–171).
- At no extra cost: all of Cricsheet, SportQA rules questions in the practice set, confidence
  ranges and significance tests for every result, the six rules as propositions in the paper.

Day numbers in entries above use the 183-day plan for days after 46. They map like this:

| 183-day plan | Now |
|---|---|
| 1–46 | 1–46 |
| — | 47–48 (new: receipts) |
| 47–73 | 49–75 |
| — | 76 (new: false premises and trick attempts) |
| 74–139 | 77–142 |
| — | 143–149 (new: Phase 9b, football) |
| 140–141 | 150–151 |
| — | 152–154 (new: AI checkers) |
| 142–152 | 155–165 |
| — | 166–171 (new: three models, tricky questions, CricBench) |
| 153–183 | 172–202 |

---

## Architecture v4 — 2026-10-19 — the design document, updated

Wrote `docs/CricketMind_Architecture_v4.pdf` (17 pages; LaTeX source in
`docs/architecture/`, built with pdflatex like v3). It records every design change since v3:
C7 and F8; the five kinds of question, the read-back, filters x measures, the Explainer and
the honest decline; false premises and trick attempts; receipts; the sport-neutral Subject
and the football case study; perception through an exported model whose card fixes frame
selection; self-checking claims and halves-up rounding (built); the journal-grade evaluation
(locked exams, baselines, ablation, AI checkers, three models, CricBench, human study); and
positioning against the 2026 work (EG-VAR, VeriFin, tool receipts). Nine invariants now
(I7 no unsupported judgment, I8 every number re-checkable, I9 an honest answer to every
question). No CricShot accuracy anywhere; every number labelled as a design parameter,
cited, or synthetic.

To confirm before citing: the ICCIT paper's final title, authors and year; EG-VAR's
authors; VeriFin's exact title (all marked in the .tex source).

---

## Plan note — 2026-10-21 — LangChain, compared on Day 59

Asked whether LangChain would read questions better. It would not make the model smarter:
the model is the same, only the wrapper changes. But because the LLM sits behind a port, a
`LangChainLLM` adapter costs nothing to add. Day 58 now builds it next to `OllamaLLM`, and
Day 59 scores both on the same 30 questions (each asked 3 ways); the better one stays (D-024).
No change to the number of days (202).

---

## Day 10 — 2026-10-21 — The registry: the menu of allowed claims

**Learned:** a registry is one list of every kind of claim the system may make, with the
engine behind each one; a dictionary from name to row makes it fast to look up. `("player")`
is plain text and `("player",)` is a tuple: the comma makes the tuple, not the brackets. A
child class (`CricketSubject(Subject)`) is a kind of its parent, so code written for the
parent accepts it, the way a car is a kind of vehicle.
**Built:** `domain/subject.py`: `Subject` is now an empty shared base; `CricketSubject` has the
4 old fields and 9 new ones (other_shot, bowler, other_bowler, venue, match_format, team,
dismissal_type, date, k). `domain/registry.py`: `ClaimType` (name, provenance, engine,
required fields, "at least one of" fields), the 10 rows, `index_by_name()` (a name listed
twice is refused), `lookup()` (None for a name not on the menu) and `missing_keys()` (lists
what the subject leaves empty; None or "" both count as empty). The document's c2 subject
now carries k=6 ("within the next 6 balls"). Written by Claude and explained, piece by piece.
**Tests:** 62 passed (19 new: 1 in test_subject.py, 18 in test_registry.py); ruff clean.
Broke it on purpose: with the "at least one of" rule switched off, the review-overturn test
failed; restored, all pass.
**Surprise:** the renderer has words for only 4 of the 10 claim types, so a claim of one of
the other 6 would fail at the very end, in front of a user. Agreed with the user: Day 11's
start-up check now also reports a claim type with no renderer words, and Day 11 adds the 6
missing phrases (plan updated; still 202 days).

**For the paper:**
- Decisions: D-007 (registry) and D-020 (sport-neutral Subject) now partly built
- Section it feeds: Method (claim contract, registry); Results (generality, later)

---

## Day 11 — 2026-10-21 — Fail at start-up, not in front of a user

**Learned:** fail fast: a mistake found before the system starts costs a minute; the same
mistake found by a user costs their trust. Report every problem at once, not only the first,
so one restart is enough. A difference needs its sign to say which way it points, and a gap
between two chances is in percentage points (20% to 32% is 12 points, not 12%).
**Built:** `check_registry()` in `domain/registry.py`: three loops, one per rule (an agent's
claim type not on the menu; a menu row with no engine; a menu row with no renderer words).
The parameter the plan called `engine_metrics` is `engine_sources` (it holds names like
"N8.M1"; plan updated). In `domain/render.py`: the six missing phrases, `signed()` and
`signed_points()`, a unit per phrase (balls, innings, reviews; "1 innings", "1 review"), a
counted comparison's range in its tag, and an unknown number style is an error. `ClaimType`
gains `needs_range`; a prediction without it is refused on the menu. Claim rule 4 now holds
for any range, D or P. Written by Claude and explained, piece by piece.
**Tests:** 83 passed (21 new); ruff clean. Broke it on purpose twice: removing the venue
phrase failed 4 tests including the start-up check; limiting the range rule to predictions
failed the counted-comparison test. Restored, all pass.
**Surprise:** every tag said "balls", so a venue average would have shown "[D, 38 balls]"
for 38 innings; and a counted comparison like +30.0 never showed its range (-16 to +76),
so it looked solid when it may be no difference at all. Plan notes added for C1 (Day 18:
a comparison with no range) and C3 (Day 20: floors for innings and reviews too).

**For the paper:**
- Evidence: E-023 (verified, unit test)
- Decisions: D-007 now built; D-025 (units, signs, percentage points, comparison ranges)
- Section it feeds: Method (registry, claim contract); Implementation (rendering)

---

## Day 13 — 2026-10-21 — The literature search

**Learned:** a systematic search fixes its words and databases first and logs every run, so
a reviewer could repeat it. "Already done" is not a disaster: it tells you what to cite and
how to say exactly what is yours.
**Did:** 16 fixed queries over arXiv, the ACL Anthology, Semantic Scholar and the open web
(Google Scholar cannot be searched by the tool), plus 6 targeted look-ups; about 30 papers
kept; abstract pages of the closest 16 read (Thomson and Reiter Sections 1–4 in full). Run
by Claude with the web-search tool at the user's request; the user reads the closest five
in full. Wrote the search log, a table of what each close paper checks, and a verdict for
each novelty claim of the v4 document, in `thesis/related_work.md`. Added 17 BibTeX
entries. Updated §13 of the v4 document and rebuilt the PDF; filled in EG-VAR's author
(Junyu Ren) and VeriFin's title and authors (Hall, Shome, Eiers).
**Verdicts:** uncertainty wording against an external interval: still new as a gate (the
problem is measured by Cerda-Mardini et al. 2026). Tone words as checked comparisons: still
new (Thomson and Reiter mark "strong" as an incorrect word, by hand). Subject binding:
partly done (VeriFin, RAGShield); what stays ours is the multi-filter re-query. Start-up
registry: new in this form but minor. Honest coverage: partly done (abstention). Slots and
renderer: already in PCN. Receipts: already in tool receipts and Thucy.
**Surprise:** PCN puts its verifier in the renderer, the same idea as our Day 9 renderer; and
Thomson and Reiter's own example of a wrong word is "strong", exactly the C7 case.
**Limits:** the hit counts are what the tool showed, not database totals, and verdicts rest
on abstracts. The user re-runs four queries on the sites' own pages and reads five papers in
full (checklist at the end of `related_work.md`).

**For the paper:**
- Evidence: E-024 (verified at abstract level)
- Numbers: N-009 (Thomson and Reiter's error counts)
- Decisions: D-026 (novelty claims narrowed to three headline contributions)
- Section it feeds: Introduction (contributions); Related work

---

## Day 14 (part 1) — 2026-10-21 — Getting ready to choose the journal

**Learned:** a journal is chosen early so that every test from now on is the kind its
reviewers expect. Quartiles and impact factors change every year and third-party sites
copy them loosely, so they are checked at the source (SCImago, JCR).
**Did:** prepared for the supervisor meeting, written by Claude: a one-page brief
(`thesis/drafts/day14_supervisor_brief.md`) with the project, its status, the three
contributions (D-026), the evaluation, and five decisions needed; a journal shortlist
(`drafts/day14_journal_shortlist.md`: IP&M, ESWA, KBS, with EAAI as backup; scope and article
types from the Elsevier pages, metrics third-party, length limits and fees to check in a
browser since ScienceDirect refused automated reading); and a draft email asking the
CricBench authors for early access (`drafts/day14_email_cricbench.md`). `outline.md` has a
"Target journal" table to fill after the meeting.
**Still to do (user):** meet the supervisor; choose the journal; read two recent papers from
the first choice; ask about ethics approval and apply if needed; send the CricBench email.
Day 14 is done when the journal is in `outline.md` and the ethics request is sent if needed.

**For the paper:**
- Section it feeds: all (the journal sets length and evaluation style)

---

## Plan note — 2026-10-21 — v3 or v4?

Asked whether the build follows v3 or v4. Both, in different senses: **CricketMind v3** is the
system (the code), and it follows **Architecture v4**, the fourth version of the design
document; **v2** is the ICCIT system. The plan now says so at the top. Phase 2 is renamed
"Faith Check, v4 design", and Day 26's table gains a middle row: the v3 design (C1–C6,
no C7) should catch F1–F7 and miss F8, which shows exactly what C7 adds. E-004, E-005,
Day 151 and the outline use the same three names. No change to the number of days (202).

---

## Phase 2 note — 2026-10-21 — built ahead of Days 12, 14 and 15

At the user's request, Claude built Phase 2 (Days 16–27) in one go, testing each day before
the next, and then explained it day by day. Days 12 (tidy, teach-back, draft), 14 (the
supervisor meeting) and 15 (the locked exam) are still open. Phase 2 answers no questions
(no data, no language model), so the exam is still written before the system can answer
anything; it must be locked before Phase 5 at the latest.

---

## Day 16 — 2026-10-21 — Words for probability

**Learned:** Kent's words of estimative probability turn a chance into words: below 13% is
"almost certainly not", 40–60% "about even", 87% and above "almost certain". Each edge belongs
to the outer band, so the bands mirror each other around 50%.
**Built:** `domain/wording.py`: five `Band`s (name, the words that fit, the words that do
not), `NEVER` (will, certain, guaranteed, always, never, definitely, surely), digit-free
`HEDGES`, `band_for(p)`, `needs_hedge(range)`, `fits(phrase, band)` and `vocabulary()`.
**Tests:** 98 passed (15 new: the four edges, 0 and 1, outside 0–1 refused, the mirror, the
document's 0.405 "about even" with a range across three bands, which words fit). Broke it:
`p <= 0.13` changed to `p < 0.13` failed the 0.13 edge test.
**Surprise:** the v4 table's own hedge, "on 4 dismissals in 50 balls", has digits, which C5
forbids: the hedges had to be digit-free (D-029).

**For the paper:**
- Decisions: D-006 (now built), D-015 (the band is computed, Day 16), D-029
- Section it feeds: Method (C4)

---

## Day 17 — 2026-10-21 — A pipeline of checks

**Learned:** functions are values: a list of functions can be run one by one. Every check has
the same shape (finding and context in, failures out), so the gate does not care which
check it runs.
**Built:** `domain/faith/gate.py`: `Failure` (agent, claim, check, reason), `Context`, the
`Check` type and `run_checks()`, which runs every check in order and keeps every failure.
**Tests:** 104 passed (6 new, with two dummy checks). Broke it: stopping at the first failure
failed `test_every_check_runs_even_after_a_failure`. Ruff asked for `extend` instead of an
append loop.

**For the paper:**
- Decision: D-027
- Section it feeds: Method (Faith Check)

---

## Day 18 — 2026-10-21 — C1: is this claim allowed?

**Learned:** the first check is the cheapest: look things up on the menu before asking any
engine.
**Built:** a claim now names its `source` (the engine the agent says it used). `Context`
gains the roster (`allowed`). `domain/faith/c1_registry.py`: the agent is on the roster; the
claim type is on the menu (if not, nothing else is checked for that claim); the agent may
assert it; the claim names the right engine; its kind (M, D, P) matches the menu; its
subject has every field the type needs; a comparison carries its range.
**Tests:** 115 passed (11 new: unknown type reported once, agent not allowed, agent not on the
roster, no source, wrong source, a count posing as a prediction, missing player, every
missing field named, a comparison with no range, three problems at once). Broke it twice:
without the stop after an unknown type the test failed; with the range rule off the
comparison test failed.

**For the paper:**
- Decisions: D-007, D-027
- Section it feeds: Method (C1)

---

## Day 19 — 2026-10-21 — C2: ask the engine again

**Learned:** a port again: the gate asks "an engine" without knowing which. The asking step
has input and output, so it lives in `application/`; the comparison is pure, so it lives in
`domain/`.
**Built:** `domain/facts.py` (`Fact`: value, n, range; refuses a value outside its own
range); `render.shown()` (a number as the reader sees it); `Context.facts`;
`domain/faith/c2_values.py`; `application/ports.py` (`FactSource`, `FakeFactSource`);
`application/faith_check.py` (`recompute()` asks with the claim's metric and subject, never
its value; `faith_check()` asks, then runs the checks).
**Tests:** 136 passed (21 new). Broke it twice: a gate that always asks about Kohli let Kohli's
number on Rohit through, and the F2 test failed; with the sample-size comparison off, two
tests failed.
**Surprise:** Kohli's 142.0 given to Rohit fails twice: the value (Rohit's is 118.0) and the
sample (44 balls, not 50).

**For the paper:**
- Decisions: D-004 (now built), D-027, D-028 (within rounding = the same number on screen)
- Section it feeds: Method (C2)

---

## Day 20 — 2026-10-21 — C3: is the evidence big enough?

**Learned:** six balls are like six coin flips: anything can happen, so a number on six balls
must say so.
**Built:** the registry now owns each type's `unit` (balls, innings, reviews), and the
renderer reads it from there; `render.counted()`. `domain/faith/text.py`, one reader for the
sentence checks: sentences, words, phrases found longest first with each word used once,
the sentence of a claim, the claims in a sentence. `domain/faith/c3_evidence.py`: floors of
30 balls, 10 innings, 20 reviews; below the floor, the claim's own sentence must warn the
reader, in words without digits.
**Tests:** 164 passed (28 new). One mistake of mine: a test used the document's two-slot
sentence with one claim, which `Finding` refused. Broke it twice: with the floor never
applying, six C3 tests failed; searching shortest phrases first failed the reader's test.
**Surprise:** the plan's example warning, "on only 6 balls", would fail C5: the warnings are
digit-free ("on only a few balls"). No list of "generalising words" is needed: every small
count needs the warning, and a word list would leak (D-030).

**For the paper:**
- Decisions: D-010 (floors widened), D-030
- Limitations: L-010 widened, L-019, L-021
- Section it feeds: Method (C3)

---

## Day 21 — 2026-10-21 — C4 part 1: words must fit the chance

**Learned:** "will probably get him out" at 41% is two lies at once: "will" is too sure for
any model, and "probably" belongs to the 60–87% band.
**Built:** the registry marks `comparison` types (a difference: zero means none).
`domain/faith/c4_wording.py`, part 1: in each sentence with a predicted chance, every word
about a chance must fit its band (with two chances, either); "never" words always fail; a
range across bands needs a hedge. A sentence with no such words passes: the number and its
range speak for themselves.
**Tests:** 175 passed (11 new). Broke it twice: with the hedge rule off its test failed; with
every word fitting, three tests failed.

**For the paper:**
- Decisions: D-006, D-029
- Section it feeds: Method (C4)

---

## Day 22 — 2026-10-21 — C4 part 2: "no detectable difference"

**Learned:** a range from −16 to +76 includes zero, so the cover drive and the pull may not
differ at all; "the stronger stroke" claims more than the data shows.
**Built:** part 2: a comparison whose range includes zero (touching counts) must say "no
detectable difference" (or a close phrase) and may use no verdict word ("stronger",
"better"); a range that excludes zero may not say there is no difference. A matchup gap is
judged as a difference, never as a chance.
**Tests:** 183 passed (8 new, including the comparison flags in the registry). Broke it
twice: `<` for `<=` let a range touching zero through; judging differences as chances failed
the matchup test.

**For the paper:**
- Decisions: D-006, D-029
- Section it feeds: Method (C4); Results (F6)

---

## Day 23 — 2026-10-21 — C5: no typed numbers

**Learned:** the slots `{c1}` contain a digit but are not numbers, so they are taken out
before scanning.
**Built:** `domain/faith/c5_numbers.py`: no digit in any script (so Bengali "১৫০" is caught),
apart from T20, T20I, U-19 and their forms.
**Tests:** 192 passed (9 new). A known gap is kept as a test: "one hundred and fifty" in words
passes (L-020). Broke it twice: judging the slots as numbers failed nine tests; accepting
only Western digits missed "১৫০".

**For the paper:**
- Decision: D-031
- Limitation: L-020
- Section it feeds: Method (C5); Results (F3)

---

## Day 24 — 2026-10-21 — C6: same player, same shot

**Learned:** a gazetteer is a list of known names and what each stands for. "Kohli" fills a
player field; "pull" a shot field; "at the death" the phase.
**Built:** `domain/faith/gazetteer.py`: the fifteen CricShot10k shots (names read from the v2
config, not changed) with plurals and synonyms, phases, bowler types, and players passed in
(the full register comes on Day 33); a surname may belong to two players. `Context.names`.
`domain/faith/c6_names.py`: in a sentence with a claim, every known name must fill a matching
field of a claim there; an empty field does not match, so the sentence cannot narrow a
number ("cover drive" beside an all-shots number fails).
**Tests:** 209 passed (17 new: the Rohit/Kohli swap, wrong shot, wrong phase and bowler type,
narrowing, the second side of a comparison, both bowlers of a matchup, reverse sweep not
read as sweep, a shared surname, case and possessives, and a name not on the list, a known
gap). Broke it twice: letting an empty field fit any name failed three tests; letting names
only bat failed the matchup test.

**For the paper:**
- Decision: D-031
- Limitation: L-019
- Section it feeds: Method (C6); Results (F2)

---

## Day 25 — 2026-10-21 — C7: tone words are hidden comparisons

**Learned:** "struggles" means "does worse than someone", so it needs a checked comparison,
and the whole range must sit on its side of zero.
**Built:** `domain/faith/c7_tone.py`: tone words pointing better ("strong", "dominates",
"better") or worse ("struggles", "weak", "worse"); "not" or "never" one or two words before
turns them around; every tone word in every sentence needs a comparison in that sentence
whose whole range is on its side. The reasons say which way the numbers point. The labels
"(F7)" and "(F3)" were taken out of C1 and C5's reasons: agents do not know our lie classes.
**Tests:** 220 passed (11 new: the plan's four, a comparison in another sentence, a judgment
with no number, one backing comparison is enough, "not strong", "never struggles", no tone
words, "out of form" read once). Broke it twice: ignoring "not" failed two tests; letting
part of the range count failed the zero test.

**For the paper:**
- Decisions: D-017 (now built), D-031
- Limitations: L-013, L-022
- Section it feeds: Method (C7); Results (F8)

---

## Day 26 — 2026-10-21 — Breaking things on purpose: the fabrication harness

**Learned:** to prove a gate works, break a correct finding in every way it could lie, and
also show it lets correct findings through (no false alarms). A result is recorded with the
code version that made it.
**Built:** `Context.surface`; `FakeFactSource.numbers()`; `domain/faith/v2_mode.py` (v2's rule,
exact equality on values, read from v2's code without changing it); `domain/faith/checks.py`
(the full gate C1–C7, the v3 design C1–C6, v2 mode); `faith_check` uses the full gate by
default; `application/harness.py` (a synthetic world, a correct finding with four claims in
three sentences, a second correct one with a warned small sample, nine broken versions: F1,
F2 in the claim, F2 in the sentence, F3–F8; a change that does not apply exactly once is
refused); `thesis/scripts/fabrication_synthetic.py` (the CSV, written safely, and a ledger
line with the commit, the changed-files flag and the output's SHA-256).
**Tests:** 230 passed (10 new). Every broken version was stopped by exactly the checks meant
for it (F6 by C4 and C7). One weak test of mine (it allowed digits 1–4 anywhere) was made
strict. Broke it twice: leaving C7 out of the gate failed two harness tests; a broken
version with a typo went unrefused and its test failed.
**Result (synthetic):** v2 mode catches 1 of 8 lie classes (F1), the v3 design 7 of 8 (it
misses F8, the tone lie), the v4 gate 8 of 8; all three let both correct findings through.

**For the paper:**
- Evidence: E-004 (verified, synthetic)
- Number: N-010 (synthetic)
- Decision: D-031
- Limitation: L-023
- Section it feeds: Results (containment)

---

## Day 27 — 2026-10-21 — Review day — M1

**Did:** read every Phase 2 file again. Two real problems found and fixed: (1) a "chance" above
100% made C4 crash instead of failing; it is now reported; (2) any check that crashes now holds
the finding back with a reason instead of crashing the gate (fail closed). Formatted the two
older files ruff had flagged since Day 9 (line breaks only). Wrote the plan's "Built" notes for
Phase 2 and a Day 39 note (a difference's n is its thinner side).
**Tests:** 232 passed (2 new); `ruff check` and `ruff format --check` clean. Broke both fixes
on purpose: each test failed, then passed again.
**M1:** the editor works on fake data: seven checks, 8 of 8 lie classes, no false alarm on
the two correct findings (E-004, synthetic).
**Still to do:** the user's teach-back (explain the Faith Check in their own words); re-run
the harness after committing, so the ledger has a line made from committed code; correct the
v4 document's hedge example at its next revision.

**For the paper:**
- Decisions: D-027 (fail closed)
- Section it feeds: Method (Faith Check); Implementation
---

## Day 28 — 2026-10-21 — Reading Cricsheet by hand

**Learned:** a Cricsheet match is three parts: `meta` (the format version), `info` (who, where,
when, the result) and `innings` (overs, each a list of balls). Overs count from 0. A wide or a
no-ball is bowled again under the same number ("9.4" twice), so a ball's position in the list,
not its number, identifies it. A wide is not a ball faced; a no-ball is, and the batter can
score off it. A review names its own batter, who need not be the striker.
**Did:** the user downloaded `all_json.zip` (23,132 matches, 148 MB; checked: no corrupt entry;
SHA-256 recorded in D-032). Claude picked five recent matches, each with a wide, a no-ball, a
wicket and a review (a men's T20I, an IPL game, a men's ODI, a men's Test, a women's T20I),
read the zip as data only, and walked through one ball of each kind. Notes in
`notes/cricsheet.md`.
**The diagram of one ball** (the wide at 9.4, Pakistan v Sri Lanka, 2026-10-11):

```
 E Malinga ──bowls──► Maaz Sadaqat (on strike)        Sahibzada Farhan (other end)
 "actual_delivery": "9.4"     over 9 = the 10th over, ball 4
 "extras": {"wides": 1}       a wide: bowled again, also numbered 9.4
 "runs": batter 0 + extras 1 = total 1
```

**Surprise:** two format versions (1.2.0 and 1.3.0) in one zip; a T20 over with eight
entries; and Cricsheet withholds 380 matches (Afghanistan men, the Afghanistan Premier League).

**For the paper:**
- Decision: D-032 (one pinned snapshot)
- Limitation: L-024 (withheld matches)
- Number: N-011
- Section it feeds: Data
---

## Day 29 — 2026-10-21 — The deliveries table

**Learned:** four SQL words: `CREATE` a table, `INSERT` a row, `SELECT` with a condition,
`GROUP BY` to add things up per batter. A schema is the list of columns and their types. A
primary key stops the same ball being stored twice.
**Built:** the user installed DuckDB 1.5.6 (`uv add duckdb`). `pipeline/schema.py`: `phase()`
(T20 and IT20: overs 0–5, 6–14, 15 on; ODI and ODM: 0–9, 10–39, 40 on; Test and first-class:
none; The Hundred: none), `Match` (what every ball shares), `Place` (innings, team, super over,
over, position), `Delivery` (31 columns), `make_delivery()`, the `CREATE TABLE` in SQL,
`insert_sql()` and `row_values()`. Written by Claude and explained.
**Tests:** 250 passed (18 new: 13 in `tests/unit/test_schema.py`, 5 in a new
`tests/integration/test_deliveries_table.py` that uses a real DuckDB database in memory). The
plan's test: the Day 28 wide at 9.4 becomes exactly the right row; the ball bowled again is a
different row. Broke it twice: `<` for `<=` lost the last powerplay over (two tests failed);
an extra SQL column failed the column-match test and every SQL test.
**Surprise:** a survey of the whole snapshot found The Hundred (389 matches labelled T20 with
five-ball overs: a T20 phase rule would be wrong there) and 228 super-over innings. DuckDB
accepts `over` as a column name even though it is an SQL word.

**For the paper:**
- Decision: D-033
- Limitation: L-025
- Number: N-012
- Section it feeds: Data
---

## Phase 3 note — 2026-10-21 — built and tested on real data; the full build is the user's

At the user's request Claude built Days 30–37 in one go, testing each day (and breaking it on
purpose) before the next. Each day was also tried on 1,000 real matches in a scratch
database, which found four problems the fixtures could not. The full build (about six
minutes) and the download of the player file are the user's to run (CLAUDE.md); Day 37's
numbers are written in from its reports.

---

## Day 30 — 2026-10-21 — A whole match into rows

**Learned:** a match file is read with three loops (innings, overs, balls); a ball's place is
its position in the over's list. A reader must fail with a message that names the match.
**Built:** `pipeline/parse_match.py` (`read_match`, `parse_match`, `ParsedMatch`); the schema
gains three small tables: `matches` (the result), `innings` (target, opening powerplay,
penalties) and `people` (each name's register id and team). Three real matches became test
fixtures (both format versions, men's and women's; Cricsheet's licence noted beside them).
**Tests:** 265 passed (15 new), checked against counts taken from the files independently:
Pakistan 215 minus Sri Lanka 133 is the "won by 82 runs" in the file, and the target 216 is
215 plus one. Broke it: the file's own ball number instead of the position, and the wrong
powerplay; both caught.
**Surprise:** a scan of all 11,689,170 balls (44 seconds) found 14 forfeited innings with no
overs, 466 "retired hurt" entries that are not dismissals, 17 balls with two wickets, and a
repeated `over` inside 3,068 balls.

---

## Day 31 — 2026-10-21 — Loading everything, safe from power cuts

**Learned:** an atomic write: a transaction either saves all of a batch or none of it, so a
power cut cannot leave half a batch. Copying a CSV file is far faster than inserting rows one
by one.
**Built:** `pipeline/ingest.py`: reads the zip directly, batches of 500, one transaction each
(COPY per table, then the `loaded` rows), unreadable files written to `problems`, the zip's
SHA-256 stored, and a database refuses a second snapshot.
**Tests:** 274 passed (9 new): everything loaded; NULL and false survive; a second run adds
nothing; stop after one match and restart; a simulated power cut inside a batch leaves
nothing behind; a broken file is written down; another snapshot is refused.
**Surprise, found by a test:** one player was missing. DuckDB, left to guess, took the first
row of the people file ("A Capsey") for a header and dropped it; it would have dropped the
first row of every batch. Fixed with `HEADER false`, and a test now guards it. On 1,000 real
matches: 643,494 balls in 20 seconds, about 6 minutes for everything.

---

## Day 32 — 2026-10-21 — Checking the data against itself

**Learned:** Cricsheet has no innings totals, but each file says who won by how much and what
the target was, so the balls can be checked against those.
**Built:** `pipeline/quality.py`: twelve checks, nine that must find nothing and three that
count real oddities; a Markdown report.
**Tests:** 283 passed (9 new): each rule caught on a spoiled copy (one wrong ball total is
caught three ways: the ball, the target, the margin).
**Surprise:** on 1,000 real matches the checks found a bug of mine: an ODI lists three
"mandatory" powerplays and the reader kept the last (530 "odd" powerplays; 92 after the fix,
all shortened matches). They also found a gap in a rule: two chases reset for rain to 20 and
33 overs, in "no result" matches that name no rain rule. After both fixes every must-hold
rule finds nothing.

---

## Day 33 — 2026-10-21 — Matching names

**Learned:** the register id is the person; names are how a match wrote them.
**Built:** `pipeline/players.py` (the players table: the name used most, matches played, and
every other name) and `domain/names.py` (`find_players`: whole name or alias, then initial and
surname, then surname or first name, most-played first).
**Tests:** 302 passed (19 new). The plan's test: "Kohli" finds "V Kohli". Broke it: least-played
first; caught.
**Surprise:** on real data "Shakib" found nobody; many players are known by their first name
(Shakib Al Hasan, Tamim Iqbal, Mushfiqur Rahim). Once first names were added, "Babar" gave
Zulfiqar Babar before Babar Azam, so surname and first name became one level, most-played
first.

---

## Day 34 — 2026-10-21 — Batting hand and bowler type

**Learned:** the plan's Route A needed R, but the same table, cricketdata's `player_meta`
(16,101 players), is published as a CSV.
**Built:** `pipeline/attributes.py`: style words to hand (right, left) and type (pace, spin);
reading the downloaded file and a hand-made one (Route B wins); full names become search
aliases. A ten-row sample file with real ids and made-up styles covers every kind of mess.
**Tests:** 311 passed (9 new). Broke it twice; the second time (reading both styles instead of
the first) no test noticed, because no example told the two apart. Added "Right arm
Offbreak, Right arm Medium" (spin first); now it is caught.

---

## Day 35 — 2026-10-21 — The attribute join

**Built:** the `deliveries_full` view (every ball with the batter's and bowler's ids, hand and
type, by LEFT JOINs) and the coverage report (share of balls with a type, by format, and the
unknown bowlers with most balls, to hand-label first).
**Tests:** 319 passed (8 new). The plan's test: the join keeps every delivery (734 of 734).
Coverage is counted again from the files, not the database. Broke it: an inner join (four
tests failed) and the download winning over a hand label (caught). The join over 643,494 real
balls takes 0.09 seconds.

---

## Day 36 — 2026-10-21 — Derived tables

**Built:** `pipeline/derived.py`: `innings_scores`, `venues` (with `match_venues`),
`bowler_days` and `reviews`, for the engines M2, M3 and M4.
**Tests:** 330 passed (11 new), checked against counts from the files (E Malinga's legal balls
on each of his two match days; every review once; Wyatt's upheld review out caught).
**Surprise:** "the name before the comma" merged nine different County Grounds (127 matches)
into one. The rule now keeps the place when a ground name comes with several: 208 names
become 181 grounds. Broke it twice (the greedy rule; retired hurt counted as out); caught.

---

## Day 37 — 2026-10-21 — Review day

**Did:** one command for the whole build, `thesis/scripts/build_data.py` (load, check,
players, attributes and coverage, derived tables, summary; three reports with ledger lines),
tried end to end on the fixtures in a scratch folder. A shared `thesis/scripts/ledger.py`
(the E-004 script now uses it and makes byte-for-byte the same CSV). `pipeline/summary.py`.
331 tests, `ruff check` and `ruff format --check` clean, no comprehensions.
**Still to do (user):** download `player_meta.csv`, run the build, commit; then the row
counts, the check results and the coverage are written into `numbers.md`, E-025 and this
journal.

**For the paper (Days 30–37):**
- Decisions: D-032 and D-033 now built; D-034 to D-038
- Limitations: L-026 to L-029
- Evidence: E-025 (planned: the full build)
- Section it feeds: Data; Method (N3 names, engines M2–M4)
