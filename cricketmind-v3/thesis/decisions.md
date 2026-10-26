# Design decisions

One entry per decision. Keep the reason honest: "because I tried X and it failed" is better
than a reason invented afterwards. Status moves from **designed** to **built** to
**validated**.

Template:

```markdown
## D-0xx — <short title>
- Date:
- Decision:
- Why:
- Alternatives considered:
- Status: designed | built | validated
- Paper section:
```

---

## D-001 — Rebuild v3 from scratch as a website
- Date: 2026-10-16
- Decision: build CricketMind v3 from an empty folder as a FastAPI backend plus a React
  frontend, following clean architecture. Nothing is copied from the v2 notebook or package.
- Why: the author is learning by building, and a working website is a stronger thesis
  deliverable than a notebook.
- Alternatives considered: extend the v2 notebook/package (faster, but teaches little and
  inherits v2's structure).
- Status: designed
- Paper section: Implementation

## D-002 — Separate folder, branch and package name
- Date: 2026-10-16
- Decision: v3 lives in its own git worktree, `D:\Cricket\cricketmind-v3-work\cricketmind-v3\`,
  on the git branch `cricketmind-v3`; its Python package is `cricketmind_v3`. CricShot keeps
  `D:\Cricket\cricshot-improve\` on its own branch, and neither folder ever switches branch.
- Why: keep CricShot10k (classifier, separate paper) and CricketMind apart. The old
  `cricketmind/` prototype is still imported by CricShot's tests, so it keeps its name.
- Alternatives considered: a separate repository; renaming the old prototype (it would break
  CricShot's tests).
- Status: designed
- Paper section: none (process)

## D-003 — Frontend stack
- Date: 2026-10-16
- Decision: React + TypeScript + Vite + Tailwind CSS + shadcn/ui + TanStack Query + React
  Router, with TypeScript types generated from FastAPI's OpenAPI schema.
- Why: the most widely used stack (help and jobs); real interactivity is needed (upload with
  progress, a live node timeline); generated types keep frontend and backend in agreement.
- Alternatives considered: Next.js (a second server, not needed); Streamlit/Gradio (demos,
  not a website; blur the layers); HTMX + Jinja (simpler, kept as the fallback).
- Status: designed
- Paper section: Implementation

## D-004 — Subject-keyed claim contract
- Date: 2026-10-16
- Decision: agents emit claims as data (metric, subject tuple, value, interval, provenance)
  plus a prose template with references like `{c1}`. A deterministic renderer inserts
  verified values. Agents never type numbers.
- Why: the v2 evaluation showed the gate passes a correct figure attached to the wrong subject,
  and an uncited figure (E-001). Its conclusion named a subject-keyed contract as a requirement.
- Alternatives considered: checking prose with a second LLM (not auditable: an LLM judging an
  LLM); keeping v2's value-only check (known blind spots).
- Status: built (claims and findings Days 7–8, the renderer Day 9; the subject-bound re-query,
  C2, Day 19)
- Paper section: Method (claim contract)

## D-005 — Three provenance classes, still two engines
- Date: 2026-10-16
- Decision: numbers come only from N6 (measured, from the clip), N8 (descriptive) and N9
  (predictive). N6 is a measurement, not a third engine.
- Why: a count of the past and a simulated future need different verification rules.
- Alternatives considered: treating all numbers alike (v2), which cannot check uncertainty.
- Status: designed
- Paper section: Method (provenance)

## D-006 — Uncertainty-faithful wording (check C4)
- Date: 2026-10-16
- Decision: a predictive claim's wording must fall in the probability band that contains its
  point estimate (bands after Kent, 1964: boundaries 0.13, 0.40, 0.60, 0.87). If the 95%
  interval crosses a boundary, a hedge naming the evidence is required. A comparison whose
  interval contains zero must say "no detectable difference".
- Why: v2's N9 interval measures Monte Carlo error only (E-003). With estimation uncertainty
  the interval is about 20 times wider (N-003, N-004). Without a wording check, an agent can
  sound certain about a coin flip.
- Alternatives considered: numbers only, no prose (rejected: the stakeholders need prose);
  no wording check (the v2 behaviour).
- Status: built (the bands Day 16; C4 Days 21–22; refinements in D-029)
- Paper section: Method (Faith Check C4)

## D-007 — Claim-type registry, checked at build time
- Date: 2026-10-16
- Decision: every claim type an agent may assert must resolve to an engine metric. This is
  checked when the graph is built, so a hollow agent is a build failure.
- Why: four v2 agents (Stadium, Compliance, Injury, Sponsorship) assert claims no engine
  computes, and pass the gate because the number they cite is correct (E-002).
- Alternatives considered: catching these at run time with an LLM judge (not auditable).
- Status: built. Day 10: the registry (`domain/registry.py`), 10 claim types, each with its
  engine and the Subject fields it needs; `lookup()` and `missing_keys()`; a name listed
  twice, or keys written without the tuple comma, are refused. Day 11: `check_registry()`
  reports every problem at once: an agent's claim type not on the menu, a menu row with no
  engine, and (added 2026-10-21) a menu row with no words in the renderer, since on Day 10
  only 4 of the 10 had them (E-023). Runs on the real graph from Day 71.
- Paper section: Method (registry)

## D-008 — Roster changes
- Date: 2026-10-16
- Decision: add Bowling Performance (N11g, replacing Sponsorship); reframe Compliance as
  Match Officials (N11i, backed by DRS review data); merge Sponsorship into Content Production.
  Club Operations and Player Representation lose valuation claims until a market engine exists.
- Why: no public data can back compliance or sponsorship claims; every v2 agent reasons about
  the batter, while Cricsheet records the bowler of every delivery.
- Alternatives considered: keeping 12 roles unchanged (hollow agents remain); adding agents
  to raise the count (more routing confusion, no new evidence).
- Status: designed
- Paper section: Method (agents)

## D-009 — N9 scope and uncertainty
- Date: 2026-10-16
- Decision: N9 has Tier 1 (delivery risk) and Tier 2 (matchup counterfactuals); Tier 3
  (batting order, chases) is future work. Intervals include estimation uncertainty (Beta
  posterior, shrinkage toward population rates), and counterfactuals are reported as a
  difference against a baseline.
- Why: Tier 3 costs about four times as much (N9 design note); honest intervals are the
  point of check C4.
- Alternatives considered: Tier 3 now (schedule risk); Monte Carlo error only (v2, misleading).
- Status: designed
- Paper section: Method (engines)

## D-010 — Design parameters (to revisit with data)
- Date: 2026-10-16
- Decision: evidence floor n_min = 30 balls (10 innings and 20 reviews added on Day 20,
  D-030; 30 wickets, and bowler types known for 80% of the wickets in a share, on Day 40, D-043; workload cut-offs 1.5 and
  0.8 on Day 41, D-044); bridge acceptance κ ≥ 0.61 ("substantial", Landis
  and Koch 1977); posture metric acceptance AUC ≥ 0.75; at most 4 active agents; at most 2
  retries and 2 regenerations.
- Why: reasonable starting points. The paper must call them design parameters, not tuned values.
- Alternatives considered: tuning them now (no data yet).
- Status: designed
- Paper section: Method; Limitations (L-010)

## D-011 — Commentary-to-shot bridge, with a go/no-go gate
- Date: 2026-10-16
- Decision: label commentary lines with the 15 shot classes (rules first, then a constrained
  LLM), validate against 500 hand labels, join to Cricsheet deliveries. Go/no-go on licence
  and agreement; the fallback is a hand-labelled evaluation set only.
- Why: Cricsheet has no shot type, and no labelled commentary dataset was found.
- Alternatives considered: aligning broadcast video to deliveries (weeks of fiddly work);
  dropping shot-level statistics.
- Status: designed
- Paper section: Method (evidence); Results (bridge)

## D-012 — Perception through an exported model and a model card
- Date: 2026-10-16
- Decision: CricShot exports the final classifier as TorchScript, plus `classes.json` and a
  model card. CricketMind loads the file and never imports CricShot code. The model card
  states every input step, including which frames are picked (CricShot drops repeated
  frames and crops before sampling, so "15 evenly spaced frames" would be wrong). A parity
  test checks the frame numbers exactly and the crops within a tolerance.
- Why: separation of the two projects; the classifier's accuracy is reported in the CricShot
  paper, not here.
- Alternatives considered: importing CricShot's model code (couples the projects).
- Status: designed
- Paper section: Method (perception) — with no accuracy figures

## D-013 — DuckDB for evidence, SQLite for the app
- Date: 2026-10-16
- Decision: the ball-by-ball evidence is a read-only DuckDB file built offline; the website's
  own data (saved reports) lives in SQLite through SQLModel.
- Why: DuckDB is fast at the engines' "filter, count, group" queries; SQLite handles the app's
  many small writes.
- Alternatives considered: one SQLite database for everything (simpler; slower aggregations).
- Status: designed
- Paper section: Implementation

## D-014 — The layer rule is enforced by a test
- Date: 2026-10-16
- Decision: the backend has five layers (`domain`, `application`, `infrastructure`,
  `pipeline`, `api`). `domain/` imports only the standard library; `application/` never
  imports `infrastructure`, `pipeline` or `api`; imports are absolute.
  `tests/unit/test_architecture.py` reads every file's imports with `ast`, without running
  the code, and fails naming the file and the import.
- Why: the claim contract, the registry and the Faith Check live in `domain/`. Keeping it
  free of frameworks keeps those rules fast to test and independent of FastAPI, DuckDB and
  the language model. A wrong import is caught the same minute, not months later.
- Alternatives considered: a written convention with no test (breaks silently); a
  dedicated import-linting tool (one more dependency for two rules).
- Status: built (both rules seen to fail on a planted import, then pass)
- Paper section: Implementation

## D-015 — A claim checks itself when it is made; its wording band is derived, not stored
- Date: 2026-10-16
- Decision: `Claim` (`domain/claims.py`) is frozen and refuses to exist if it breaks one of
  four rules: the id is `c` plus digits; a descriptive claim carries its sample size
  (n ≥ 1); a predictive claim carries a 95% range that contains its value; a measured claim
  names its clip. The wording band shown in the v3 document's example claim
  (`"band": "about even"`) is not stored; it is computed from the value when needed (built
  on Day 16: `band_for` in `domain/wording.py`).
- Why: a malformed claim is stopped where it is made, before it reaches the Faith Check, the
  report or a user. A stored band could disagree with the value it describes.
- Alternatives considered: leaving all validation to the Faith Check (a malformed claim
  travels further and fails further from its cause); storing the band, as in the document's
  example (two sources of truth).
- Status: built (each rule broken once in a test; removing the range check let a 0.9
  prediction with range 0.155 to 0.697 through, and the test failed)
- Paper section: Method (claim contract)

## D-016 — Numbers shown to people round halves up, in decimal arithmetic
- Date: 2026-10-17
- Decision: the renderer (`domain/render.py`) rounds every displayed number with exact
  halves going up, using Python's `decimal` module on the number's shortest written form:
  0.405 → 41%, 0.155 → 16%, 0.145 → 15%, 0.125 → 0.13. Each claim is shown with its tag:
  `[D, 50 balls]`, `[P, range 16% to 70%]`, `[M, clip vid1_0]`.
- Why: people expect halves to round up. Python's default rounds an exact half to the even
  number (0.405 → 40%), and float arithmetic can push a half just below (0.145 × 100 =
  14.4999…, shown as 14%). A reader comparing the report with the engine's value would see
  an unexplained one-point difference.
- Alternatives considered: Python's default formatting (half to even, plus float error);
  `round()` (the same half-to-even rule).
- Status: built (switching to half-even made three tests fail; 0.405 showed as 40%)
- Paper section: Implementation (rendering)

## D-017 — Check C7: a tone word needs a checked comparison
- Date: 2026-10-17
- Decision: a new Faith Check, C7, beyond the architecture document's C1–C6. Tone words
  carry a direction ("strong", "dominates" up; "struggles", "weak" down). A tone word about a
  player needs a comparison claim in the same sentence (`sr_vs_average`, the player against
  the average batter with the same filters, or `sr_difference`) whose whole 95% range is on
  that side of zero. A new fabrication class, F8 (a tone word the numbers do not support),
  joins the harness.
- Why: C1–C6 check every number and its subject, not the words around it: "Kohli struggles
  with the cover drive: {c1}" passes with a strong 142.0 (L-013). A tone word is a hidden
  comparison, and v3 already checks comparisons (C4, "no detectable difference"), so the
  same deterministic idea extends to it.
- Alternatives considered: a second language model judging tone (not auditable, so not a
  gate, as in D-004; it may still help measure); fixed sentence templates only (removes the
  agents' voice); leaving tone unchecked.
- Status: built (C7 on Day 25; F8 in the harness on Day 26: the v3 design without C7 misses
  it, the full gate catches it; E-004, synthetic)
- Paper section: Method (Faith Check); Results (F8)

## D-018 — Every question gets an honest answer, however it is worded
- Date: 2026-10-17
- Decision: N3 sorts each question into one of five kinds (stats, prediction, video, rules,
  can't-check), and every answer starts "I read this as: …". A stats question is one engine
  query: any combination of filters (player, bowler, opposition, venue, season, format,
  innings, phase, bowler type, batting hand) with one measure, so questions nobody listed are
  still answered with checked numbers. Rules questions go to an Explainer (N11m): words only,
  no digits, no named players, labelled "general explanation, not checked against data".
  Questions no data can answer get "I can't check that" and what can be asked instead.
  Phase 10 measures coverage on 230 questions (40 written 5 ways each, plus 30 outside the
  plan's lists), including whether every wording gets the same checked facts.
- Why: people word the same question in many ways and ask things no list foresaw. Answering
  everything confidently is the failure v3 exists to prevent; answering honestly, and
  measuring how much could be proved, is a claim the paper can defend.
- Alternatives considered: the language model writing SQL over the delivery table (most
  LLM-generated cricket SQL runs, but under 29% is semantically correct, N-007, and nothing
  would check it); keeping only the nine claim types (most questions unanswered); answering
  rules questions from the model's memory with numbers (unchecked numbers, against I1).
- Status: designed (Days 40–42, 54, 67, 146–147)
- Paper section: Method (question handling); Evaluation (coverage)

## D-019 — An evaluation built for a journal
- Date: 2026-10-19
- Decision: the evaluation answers the five questions journal reviewers ask. Is it new? A
  systematic literature search on Day 13, and the journal chosen with the supervisor on Day
  14 (not at the end). Is it better than other approaches? Two baselines on the same
  questions: a plain AI with no data or checks, and an AI given the engine's facts but no
  Faith Check (Days 150–152). Does each part matter? An ablation that turns each check off in
  turn (Day 141). Was it tested fairly? A locked 100-question final exam, written on Day 15
  before the system can answer anything, fingerprinted with SHA-256 and run once on Day 149;
  improvements are made only on a separate practice set. Do people find it useful? A small
  blind human study, about 10 raters (Days 153–155), with ethics approval asked for on Day 14.
- Why: a strong idea alone does not get a journal paper accepted; reviewers ask what it is
  compared with, whether each part is needed, whether the test was fair, and whether anyone
  finds it useful. The previous plan compared only with v2 and had no locked test set and no
  human study.
- Alternatives considered: comparing only with v2 (weak: v2 is our own earlier system);
  reporting results on the same questions used to improve the system (overfitting); a large
  field study (beyond a thesis).
- Status: designed
- Paper section: Evaluation; Results

## D-020 — A sport-neutral Subject, and a football case study
- Date: 2026-10-19
- Decision: `Subject` becomes a small shared base with no fields; `CricketSubject` holds the
  cricket filters, and `FootballSubject` (Day 147) the football ones. The checks, the claims,
  the findings and the renderer only ever see a `Subject`. Phase 9b runs the method on open
  football event data with about 10 measures, three agents, the 8-lie test and a locked
  40-question football exam; video is not ported.
- Why: a method shown in two sports with zero changed lines in its checks is evidence that it
  is general, not a cricket trick; Q1 reviewers ask for this. Making `Subject` neutral on Day
  10 costs little; doing it later would mean rewriting checks that already work.
- Alternatives considered: cricket only (weaker generality claim); a generic `Subject` as a
  dictionary of filters (loses the keyword-only, typo-crashes-at-once safety of Day 6); a full
  second product (beyond a thesis).
- Status: partly built. Day 10: `Subject` is an empty base and `CricketSubject(Subject)` has
  13 fields; `Claim` and `missing_keys()` only see a `Subject`, and a subject without a field
  counts that field as missing (tested with a plain `Subject`). `FootballSubject` on Day 147.
- Paper section: Method; Results (generality)

## D-021 — A receipt for every number
- Date: 2026-10-19
- Decision: every claim from an engine carries a receipt: the data snapshot's checksum, the
  exact engine query and the engine's version. A receipt checker re-runs any receipt and
  confirms the number; Phase 10 re-checks every number in the final exam this way.
- Why: anyone, including a reviewer, can re-check any number in any answer, and receipts
  make the comparison with tool-receipt work (NabaOS) and Proof-Carrying Numbers concrete:
  ours carry the statistical query, not only the fact that a tool was called.
- Alternatives considered: logging only (cannot be re-run); signing receipts cryptographically
  (adds little when the engine and data are local and pinned).
- Status: designed (Days 47–48)
- Paper section: Method; Implementation

## D-022 — False premises and trick attempts
- Date: 2026-10-19
- Decision: N3 marks a judgment hidden in a question ("Why is Kohli bad against spin?"); the
  answer checks it like a tone word (C7) and says plainly when the premise is wrong. Text
  inside a question is only ever treated as a question, never as an instruction. The locked
  exam includes 20 such questions.
- Why: a system that answers "why is he bad?" by explaining a weakness that does not exist
  repeats the user's error with apparent authority; answering only checked facts is not
  enough if the question itself is false.
- Alternatives considered: answering the question as asked (repeats false premises);
  refusing every question that contains a judgment (unhelpful).
- Status: designed (Day 76; evaluated on Day 184)
- Paper section: Method; Results

## D-023 — Stronger comparisons: AI checkers, three models, an outside test set
- Date: 2026-10-19
- Decision: the Faith Check is compared with two AI-based checkers (an AI judge and
  Chain-of-Verification) on the 8-lie test; the main experiments are repeated with three
  language models (two local, one online, version and date recorded); and, if released,
  CricBench's English questions are used as a test set written by other researchers.
- Why: the obvious reviewer questions are "why not let another AI check?", "does this depend
  on one model?" and "would it hold on questions you did not write?". New 2026 work on
  checking AI numbers (EG-VAR, VeriFin, tool receipts) makes these comparisons necessary.
- Alternatives considered: one local model only (results could be model-specific); only
  self-written questions (author bias, L-009).
- Status: designed (Days 163–167, 182–187)
- Paper section: Evaluation; Results

## D-024 — The language model sits behind our own port; LangChain is compared, not assumed
- Date: 2026-10-21
- Decision: N3 and every other LLM call go through our own `LLM` port. Day 57 builds the
  `OllamaLLM` adapter (about 30 lines: question plus JSON schema in, JSON out). Day 58 adds a
  second adapter, `LangChainLLM` (`langchain-ollama`), behind the same port. Day 59 runs both
  on the same questions with the same model, prompt and schema; the better score stays, and
  a tie keeps `OllamaLLM` because it is simpler.
- Why: LangChain does not change the model; both adapters send Ollama the same kind of
  request, so any difference should be small. Accuracy comes from the schema, allowed
  values, the checks after the model, a retry with the error, and the read-back. Measuring
  both settles it with numbers instead of opinion. The port keeps the inner layers free of
  LLM libraries and makes the three-model experiment (D-023) one adapter per model.
- Alternatives considered: LangChain only (more layers to understand and fake in tests);
  our adapter only, never compared (a reviewer may ask).
- Status: designed (Days 57–59)
- Paper section: Implementation (only if the scores differ enough to matter; numbers.md row first)

## D-025 — Every shown number says what it counts, which way it points, and how sure it is
- Date: 2026-10-21
- Decision: each renderer phrase names the unit of its sample size (`[D, 38 innings]`,
  `[D, 120 reviews]`, `[D, 50 balls]`); a difference shows its sign ("+30.0"), and zero has
  none; a gap between two chances is given in percentage points, not percent; a counted
  comparison shows its range like a prediction does (`[D, 100 balls, range -16.0 to +76.0]`).
  `ClaimType.needs_range` marks the claim types that must carry a range (`sr_difference`,
  `sr_vs_average`, and every prediction); C1 enforces it on Day 18. Any range, counted or
  predicted, must hold its own value (Claim rule 4).
- Why: found while writing the six missing phrases on Day 11. Every tag said "balls", so a
  venue average over 38 innings would have read "[D, 38 balls]". "12% more" and "12 points
  more" are different claims (20% × 1.12 = 22.4%, not 32%). And invariant I4 says
  comparisons carry their uncertainty: "+30.0" alone reads as solid, but its range -16 to
  +76 includes zero, which C4 (Day 22) and C7 (Day 25) need to see.
- Alternatives considered: one unit for every claim (wrong for venue and review claims);
  showing ranges only for predictions (a comparison would hide its uncertainty); leaving
  "which claims need a range" to each check (a second list that could drift from the menu).
- Status: built (a phrase removed: 4 tests and the start-up check failed; the range rule
  limited to predictions again: the counted-comparison test failed)
- Paper section: Method (claim contract); Implementation (rendering)

## D-026 — Novelty claims narrowed after the literature search
- Date: 2026-10-21
- Decision: the paper's headline contributions are (1) a subject-keyed re-query for
  statistics whose subject has several filters at once, with the prose's names checked
  against it; (2) uncertainty wording gated against an external engine's interval; and
  (3) tone words and question premises gated by a checked comparison pointing the same way;
  all backed by the journal-grade evaluation. The start-up registry and honest, measured
  coverage are presented as parts of the method, not as new ideas. The slots-and-renderer
  design (PCN), receipts (tool receipts; Thucy) and the fix-retry-withhold loop (GSAR) are
  cited, not claimed.
- Why: the Day 13 search (16 fixed queries, `related_work.md`) found subject binding partly
  done (VeriFin binds operands to period, unit and line item; RAGShield links numbers to
  entities), a tool registry checked at call time (Closed-World Resolution), and a large
  abstention literature. It found the two other problems measured but not gated: models pick
  miscalibrated words for external predictions (Cerda-Mardini et al. 2026), and human
  annotators mark "strong" as an incorrect word in sports summaries (Thomson and Reiter
  2020). Claiming less, and exactly, is safer with reviewers than claiming five things.
- Alternatives considered: keeping the five claims of the v4 document (two are partly done,
  one is minor); dropping subject binding entirely (the multi-filter re-query is still not
  found).
- Status: designed (verdicts at abstract level; confirmed after reading the full texts)
- Paper section: Introduction (contributions); Related work

## D-027 — How the Faith Check is built: pure checks, one asking step, fail closed
- Date: 2026-10-21
- Decision: the seven checks are pure functions in `domain/faith/`, each reading a finding and
  a `Context` (the roster, the engines' answers, the name list) and returning `Failure`s
  (agent, claim, check, reason). `run_checks` runs them in order, cheapest first, and keeps
  every failure, so an agent hears all its mistakes at once. Asking the engines again is the
  one step with input and output; it lives in `application/faith_check.py`, behind the
  `FactSource` port. The gate fails closed: a number nobody re-checked, an engine with no
  answer, an impossible chance, or a check that crashes all hold the finding back with a
  reason. To make the checks possible, a claim names its `source`, and a claim type its
  `unit` and whether it is a `comparison`.
- Why: pure checks are tested in milliseconds without engines, and the architecture test
  keeps `domain/` free of outside code. Collecting every failure saves regeneration rounds.
  A safety gate that crashes, or passes a number by default, fails at its one job.
- Alternatives considered: C2 inside `domain/` holding a port (domain would depend on outer
  code); stopping at the first failure (more rounds of regeneration); letting exceptions
  through (the whole question fails, or worse, a finding passes unchecked).
- Status: built (Days 17–19, 27)
- Paper section: Method (Faith Check); Implementation

## D-028 — C2: "within rounding" means the reader would see the same number
- Date: 2026-10-21
- Decision: C2 passes a value when it shows exactly as the engine's value shows (the
  renderer's own rounding: 142.04 and 142.0 both show "142.0"; 0.4049 shows "40%" and the
  engine's 0.405 "41%", so it fails). A count's sample size must match exactly, and a range
  must match end for end, or both be absent.
- Why: what matters is what the reader sees. A fixed tolerance would either pass numbers
  that display differently or reject ones that display the same. An inflated sample size
  would let a small-sample claim dodge C3.
- Alternatives considered: an absolute tolerance (arbitrary, unit-dependent); a relative
  tolerance (passes visibly different numbers at large values).
- Status: built (Day 19)
- Paper section: Method (C2)

## D-029 — C4 refinements: the word lists, optional words, digit-free hedges
- Date: 2026-10-21
- Decision: (1) The bands keep Kent's edges (0.13, 0.40, 0.60, 0.87; each edge goes to the
  outer band) and the v4 table's words, plus close variants ("very unlikely", "probably",
  "even chance", "highly likely"); "definitely" and "surely" join the words too strong for
  any band. (2) Words are optional: the number and its range speak for themselves, but any
  word about a chance must fit its band, and a word of another band fails. (3) With two
  chances in one sentence, a word may describe either. (4) A hedge for a range that spans
  bands has no digits ("the estimate is very uncertain"), so it passes C5; the numbers are
  in the tag. (5) A difference (`comparison` in the registry) is judged by part 2 only,
  never by bands: a gap of 12 points is not a 12% chance.
- Why: forcing a band word into every sentence would reject honest findings; leaving a
  phrase like "probably" unrecognised would let it through at 20%. The v4 document's hedge
  "on 4 dismissals in 50 balls" would itself fail C5: the design contradicted itself.
- Alternatives considered: a required band word in every sentence (many false alarms);
  hedges with digits (they fail C5).
- Status: built (Days 16, 21, 22); the v4 document's example is corrected at its next revision
- Paper section: Method (C4)

## D-030 — C3: a floor per unit, and a warning whatever the words
- Date: 2026-10-21
- Decision: a count under its floor (30 balls, 10 innings, 20 reviews; design parameters,
  D-010) must carry a warning in its own sentence ("on only a few balls", "a small sample"),
  whatever words surround it. The unit lives in the registry, so C3 and the renderer read
  the same one. The warnings have no digits. For a difference, the engine reports n as the
  smaller of its two samples, so the floor protects the thinner side (Day 39).
- Why: a list of "generalising words" would never be complete ("a liability against spin");
  requiring the warning for every small count does not depend on any list. The plan's own
  example, "on only 6 balls", would have failed C5.
- Alternatives considered: a warning only beside generalising words (leaks); a floor on the
  total of both samples of a difference (hides a thin side).
- Status: built (Day 20; the difference rule on Day 39)
- Paper section: Method (C3); Limitations (L-010)

## D-031 — C5, C6, C7 rules, and the fabrication harness
- Date: 2026-10-21
- Decision: one shared reader splits sentences and finds phrases, longest first, each word
  used once ("as likely as not" is not also "likely"). C5: apart from claim slots and a short
  allowlist (T20, T20I, U-19), no digit in any script. C6: in a sentence with a claim, every
  known name must fill a matching field of at least one claim there; an empty field does not
  match, so the prose cannot narrow a claim ("cover drive" beside an all-shots number); a
  person may fill player, bowler or other bowler. C7: every tone word, in any sentence, needs
  a comparison in its own sentence whose whole range is on its side of zero; "not" or "never"
  one or two words before it turns it around. The harness (Day 26) builds one correct
  synthetic finding, a second correct one with a warned small sample, and nine broken
  versions (F1–F8, F2 two ways), runs v2 mode (exact equality on values, like v2's code), the
  v3 design (no C7) and the full gate, and writes a CSV plus a ledger line (commit, changed
  files, the output's SHA-256).
- Why: one reader for every check means one tokenisation to test. Two correct findings
  measure false alarms; two versions of F2 separate C2 (the claim lies) from C6 (the sentence
  lies). A broken version that silently fails to apply is refused, so a typo cannot pass
  for a caught lie.
- Alternatives considered: separate regular expressions per check; one broken version per
  class (it would hide that C6 is needed when the claim is right but the prose is wrong).
- Status: built (Days 20, 23–26)
- Paper section: Method (C5–C7); Results (containment, E-004)
## D-032 — One pinned Cricsheet snapshot
- Date: 2026-10-21
- Decision: all real data comes from one Cricsheet download, `all_json.zip`, fetched on
  2026-10-21 (23,132 matches; SHA-256 `2c0b9c7d4f86da1d1ba939539d40bb435917c360ecf71661daf6f0a2bd5f8573`),
  kept unchanged in `data/raw/` (git-ignored). Every engine number and every receipt (D-021)
  refers to this snapshot; a newer download would be a new, recorded snapshot.
- Why: the same question must give the same number months later, and a reviewer must be able
  to fetch the same data. Cricsheet updates the zip weekly.
- Alternatives considered: downloading on every run (numbers would drift); the CSV or YAML
  files (JSON is the format Cricsheet documents most fully).
- Status: built (downloaded Day 28; the ingest stores the zip's SHA-256 and refuses a
  database built from another snapshot, Day 31)
- Paper section: Data; Reproducibility
## D-033 — The deliveries table: one row per ball, keyed by position
- Date: 2026-10-21
- Decision: one table, `deliveries`, one row per ball as bowled (wides and no-balls included),
  with the match's facts (format, gender, team type, competition, date, venue) repeated on
  every row so a filter never needs a join. A ball is identified by match, innings, over and
  its position in the over's list (`ball_index`), never by Cricsheet's "9.4", which repeats
  when a ball is bowled again; the database refuses a second row with the same key. Overs
  keep Cricsheet's count from 0. Phases come from a rule per format: T20 and IT20 overs 0–5
  powerplay, 6–14 middle, 15 on death; ODI and ODM 0–9, 10–39, 40 on; Test and first-class
  none. The Hundred (five-ball overs) and super overs get no phase. A ball with two wickets
  keeps the count and the first wicket's details.
- Why: the engines filter and count; repeated match facts cost little in a column store and
  keep every query to one table. Day 28 showed "9.4" twice in one over. The survey of the
  snapshot found 389 matches of The Hundred labelled T20 and 228 super-over innings.
- Alternatives considered: a separate matches table (a join on every query); the ball number
  as the key (not unique); the file's own `powerplays` ranges (not in every file; checked
  against the rule on Day 32).
- Status: built (Day 29: `pipeline/schema.py`; filled by the reader and the load, Days 30–31)
- Paper section: Data
## D-034 — The load: from the zip, in batches, each batch all or nothing
- Date: 2026-10-21
- Decision: the load reads the match files straight from the zip (never unpacked), parses a
  batch of 500, writes one CSV per table and copies them into DuckDB inside one transaction,
  together with a `loaded` row per match and a `problems` row per unreadable file. A power
  cut leaves no half-batch, and a rerun skips every match already loaded or recorded as a
  problem. The CSV files declare `HEADER false`: left to guess, DuckDB took a first row of
  names for a header and dropped it (a test found the missing player on Day 31). Next to the
  balls, three small tables keep each match's result, each innings' target and opening
  powerplay, and each person's register id.
- Why: CLAUDE.md asks that every long job survive a power cut. One transaction per batch is
  the simplest atomic write; COPY from CSV loads about 32,000 balls a second (about 6 minutes
  for the snapshot) where row-by-row inserts would take far longer.
- Alternatives considered: unpacking 23,132 files first (slow, untidy); one transaction for
  everything (a power cut near the end loses it all); a manifest file beside the database
  (two places that can disagree).
- Status: built (Days 30–31)
- Paper section: Data; Reproducibility

## D-035 — The data is checked against itself, with two kinds of rule
- Date: 2026-10-21
- Decision: twelve checks, each a rule and a query for the rows that break it. Nine must
  find nothing (a ball's total is its parts; extras add up; a chase's target is the first
  innings plus one; "won by N runs" is the difference; every loaded match has all its balls
  and a match row; no empty names; every batter and bowler has a register id; every file was
  readable). Three only count real but unusual things for the paper (overs with more legal
  balls than allowed, shortened powerplays, balls with two wickets). Every check is tested by
  spoiling a copy of the fixtures.
- Why: Cricsheet has no innings totals, so the data can only be checked against itself: the
  target and the margin in each file are independent of the balls. Running the checks on
  1,000 real matches found a bug in the reader (an ODI lists three "mandatory" powerplays;
  the reader kept the last, not the opening one) and a gap in a rule (a target reset for
  fewer overs is a rain target even when no rain rule is named).
- Alternatives considered: trusting the source (a bug like the powerplay one would pass
  silently); an outside source of scorecards (licensing; a second pipeline to check).
- Status: built (Day 32); on the full data (Day 37) the pipeline breaks no rule, and 7 of
  25,611 target and margin checks find the source disagreeing with itself (N-015, L-030)
- Paper section: Data (quality)

## D-036 — Finding a player from a name in a question
- Date: 2026-10-21
- Decision: a name is matched in three steps, the closest kind found winning: the whole name
  or an alias ("V. Kohli", "Virat Kohli"); a first name's initial with the surname
  ("Rohit Sharma" -> "RG Sharma"); one part of the name, surname or first name, most-played
  first ("Kohli"; "Mendis" gives both; "Babar" gives Babar Azam before Zulfiqar Babar).
  Dots, case, hyphens and Cricsheet's "(1)" for same-named people are ignored. Full names
  from the player file become aliases. Every candidate is returned in order; N3 shows its
  choice in the read-back.
- Why: on 1,000 real matches, "Shakib" first found nobody: many players are known by their
  first name (Shakib Al Hasan, Tamim Iqbal, Mushfiqur Rahim), which matters for a
  Bangladeshi user base. A strict surname-first order then put Zulfiqar Babar before Babar
  Azam.
- Alternatives considered: fuzzy string distance (unexplainable matches); surname only
  (misses first-name players).
- Status: built (Day 33)
- Paper section: Method (N3, names)

## D-037 — Batting hand and bowler type: from cricketdata, and by hand
- Date: 2026-10-21
- Decision: Route A without R: the cricketdata package's `player_meta` table (16,101
  players, from ESPNcricinfo's player pages) is downloaded once as a CSV from r-universe and
  joined on the Cricsheet id. Its style words are mapped to right or left hand and pace or
  spin (pace words first, so "slow medium" is pace; of two styles, the first); unknown words
  give no label. Route B: a short hand-made file fills gaps and wins over the download. A
  view, `deliveries_full`, joins hand and type to every ball, and the coverage report lists
  the unknown bowlers with the most balls, to label first.
- Why: the engines' filters need bowler type and batting hand; Cricsheet has neither. The
  plan's Route A needed R; the same table is published as a CSV.
- Alternatives considered: installing R; scraping player pages ourselves (terms of use);
  guessing from names (no).
- Status: built (Days 34–35); on the full data a bowler type is known for 96.4% of balls
  and a batting hand for 96.8% (N-014)
- Paper section: Data

## D-038 — Derived tables, and one name per ground
- Date: 2026-10-21
- Decision: four tables for the engines of Phase 4: `innings_scores` (runs with penalties,
  wickets without "retired hurt" or "retired not out", legal balls, won, chasing; M2),
  `venues` (M2), `bowler_days` (legal balls per bowler per match date; M3) and `reviews`
  (who asked, upheld means overturned, the dismissal; M4). A ground is named by the part of
  its name before the first comma, unless that part comes with several different places
  after the comma, when the place is kept: "Lord's" and "Lord's, London" are one ground;
  "County Ground, Bristol" and "County Ground, Taunton" are two.
- Why: on 1,000 real matches, "the part before the comma" alone merged nine County Grounds
  into one; the refined rule turns 208 names into 181 grounds without that error.
- Alternatives considered: the raw venue strings (one ground counted twice); a hand-made
  ground list (does not scale; may come later for the grounds N3 is asked about).
- Status: built (Day 36)
- Paper section: Data; Method (engines M2–M4)

## D-039 — How M1 counts a strike rate and a dismissal rate
- Date: 2026-10-22
- Decision: a ball faced is every ball to the batter except a wide; a no-ball counts (the
  batter could hit it). Strike rate = 100 × runs off the bat ÷ balls faced. Dismissal rate =
  the batter's own dismissals ÷ balls faced, where any wicket of the striker counts except
  "retired hurt" and "retired not out"; an out on a wide (stumped, run out) counts; a run out
  at the non-striker's end is not on a ball the batter faced, so it is not in the rate. n is
  the balls faced. Super overs are left out. No ball faced: no answer. The engine does not
  hide a small sample: it gives n, and C3's floor decides.
- Why: these are the scorers' conventions, so a number can be set against published records:
  Kohli's Test runs come out at exactly 9,230, and his dismissals at 197 (196 as striker, one
  run out as non-striker). The rules are pure functions (`domain/balls.py`,
  `domain/measures.py`), so Day 39's bootstrap can re-use them on resampled balls.
- Alternatives considered: counting non-striker run outs (that is the batting average's
  count, a different measure, and it would add outs on balls the batter did not face);
  dropping small samples inside the engine (C3 does it once, for every claim).
- Status: built (Day 38)
- Paper section: Method (N8 M1)

## D-040 — The delivery repository: SQL picks the balls, Python counts them
- Date: 2026-10-22
- Decision: the engines ask a `DeliveryRepository` port for the balls a subject is about.
  The DuckDB adapter picks them with SQL on `deliveries_full`; the engine counts them in
  plain Python. People are matched on the register id, never on the name. A name that two or
  more players share is refused, not guessed. A subject field the adapter cannot filter by is
  refused, never left out (today it filters by player, bowler, phase and bowler type; shots
  wait for the bridge). Every value goes into the SQL as a `?`. Balls come back in match
  order (date, match, innings, over, ball), so one question always gives one list. Raw SQL,
  not an ORM.
- Why: every counting rule sits in one place and is tested without a database (a fake with
  five balls). On the real data a player's question takes 0.06–0.4 s (Kohli: 42,341 balls).
  A left-out filter would answer a different question with a true-looking number. 161 names
  belong to more than one player (five "Imran Khan"s). Day 39's seeded bootstrap needs the
  same balls in the same order. An ORM is built for editing one row at a time, adds a layer
  over DuckDB, and hides the exact query, which Day 47's receipt must show.
- Alternatives considered: counting in SQL (fast, but the rules would live in SQL, out of
  reach of the fake and the unit tests); matching on names (one player under two names is
  split, two players with one name are merged); an ORM such as SQLAlchemy.
- Status: built (Day 38). The average batter of Day 39 is millions of balls: counted in SQL,
  decided then.
- Paper section: Method (N8; architecture)

## D-041 — The bootstrap: how a range is made
- Date: 2026-10-22
- Decision: a difference's 95% range comes from 20,000 replays with seed 42. A replay has as
  many balls as the sample, each picked at random, repeats allowed. It is drawn as how many of
  each kind of ball come up (one binomial draw per kind, from what is left), which gives the
  same replays as picking one ball at a time. The range keeps the middle 95%: the lowest and
  highest 2.5% are dropped. The two sides are replayed separately, and n is the smaller side
  (D-030). A replay with no ball faced, or a range that misses its own estimate, gives no
  answer.
- Why: the same balls must always give the same range (receipts, Day 47). Picking one ball at
  a time would take about 5 minutes for Kohli's 42,988 balls; counting kinds (48 of them)
  takes about a second. Adding up every possible replay of the v2 example gives exactly
  N-002's [−16, +76]; 20,000 replays land within one run of it (seed 42: [−16, +78]).
- Alternatives considered: one ball at a time (minutes); numpy (a new dependency, and still
  random); a normal-curve range (wrong for small samples, the ones that matter); the BCa
  bootstrap (better for lopsided samples, harder to explain; later, if Day 53's calibration
  asks for it).
- Status: built (Day 39)
- Paper section: Method (N8 M1; C4)

## D-042 — A player against everyone else, like with like
- Date: 2026-10-22
- Decision: `sr_vs_average` sets a player's strike rate against every other batter's under
  the same filters; the player is left out, so the two sides share no ball. Like with like:
  everyone else's strike rate is taken in each of the player's formats and mixed in his
  shares (half his balls in T20s and half in ODIs: half of each). DuckDB counts everyone
  else's balls, grouped by format and kind with the names replaced (`THE_BATTER`,
  `SOMEONE_ELSE`), so the Python counting rules still apply. Their average is taken as fixed
  and only the player's balls are replayed; that needs 20 times the player's balls in each
  of his formats, or there is no answer.
- Why: on the real data, Kohli faced Bumrah only in T20s (108 balls, 147.2), but everyone
  else's balls against Bumrah include 10,146 Test balls (44.1). Pooled, the difference came
  out +77.8; like with like it is +35.8 (N-016). With 20 times the balls, the average's own
  wobble is about a twentieth of the player's (when the spread per ball is alike), so
  leaving it out narrows the range by about 2.5%.
- Alternatives considered: all balls pooled (sets T20 against Test); the same matches only
  (too few balls for the 20 times rule); replaying everyone else's millions of balls as well
  (slow, and changes almost nothing at 20 times the balls); relying on a format filter alone
  (Day 43; a question that names no format would still mix).
- Status: built (Day 39); like with like is by format and gender since Day 42 (D-046)
- Paper section: Method (N8 M1; C7)

## D-043 — M2, a ground's profile: which matches, innings and wickets count
- Date: 2026-10-22
- Decision: M2 gives three numbers for a ground in one format, each a menu row:
  `venue_first_innings_avg`, `venue_chasing_win_share` (new) and `venue_wicket_share`
  (new; pace or spin). Only standard matches count: the format's usual length (120 balls an
  innings in a T20, 300 in an ODI; any length in a Test), so The Hundred and matches cut
  short before the start are left out. A first innings counts when it ran its course: all
  out, every scheduled ball batted, or declared. The chasing share is over limited-overs
  matches with a winner, rain-rule results included; ties and no results are left out, and
  Tests have none. The wicket share is over the bowlers' wickets (bowled, caught, caught and
  bowled, lbw, stumped, hit wicket) whose bowler type is known, and needs that type for 80%
  of them. A new unit, wickets, has a C3 floor of 30. The ground is named by its key from
  D-038 and the format by Cricsheet's match type, until Day 43; a subject with any other
  field filled (a team, a player) gets no answer.
- Why: at Lord's, 61 of 127 T20 first innings are The Hundred's 100 balls; across the data,
  about 5% of T20 and 8% of ODI first innings were cut short (rain, reduced matches). At
  Mirpur the bowler type is known for 99.7% of the wickets. On real grounds the numbers look
  right: spin takes 68% of the Test wickets at Galle and 19% at Lord's.
- Alternatives considered: every innings (rain pulls the average down); a separate rain
  flag (Cricsheet does not mark a reduced match before the start; the length rule finds it);
  a share over all wickets (run outs are nobody's bowling); unknown types counted as a third
  group (the share would shrink as coverage falls). The 30-wicket floor and the 80% rule are
  design parameters (D-010).
- Status: built (Day 40); a ground's men's and women's matches kept apart since Day 42
  (D-046): at Lord's the men's T20 first innings are 90, 29 of them The Hundred's
- Paper section: Method (N8 M2)

## D-044 — M3, a bowler's workload: what counts, and what the flag may say
- Date: 2026-10-22
- Decision: a bowler's load is every ball he bowled, wides, no-balls and super overs
  included. A multi-day match's balls are spread over its days in the order they were
  bowled (a fifth to each day of a five-day Test), because Cricsheet gives a match's dates,
  not each ball's. This week is the 7 days up to and including the day asked about; his
  usual week is a quarter of the 28 days up to and including it, as in Hulin et al. (2014).
  The ratio is this week over his usual week, and n is the 28 days' balls. The flag words
  come from the number, in the renderer: above 1.5 "a spike above his usual week", below 0.8
  "well below his usual week", otherwise "close to his usual week" (design parameters,
  D-010). No answer when he bowled nothing in the 28 days, when the data has no ball of his
  before them (a newcomer's usual week is unknown, not zero), when the day is after the data
  ends, when several players share the name, or when anything but a bowler and a date is
  asked. Never an injury chance (L-003).
- Why: on the real data, putting a Test's balls on its first day (as Day 36's `bowler_days`
  does) hid Bumrah's 327 balls at Melbourne from the week before the Sydney Test: his ratio
  on 4 January 2025, the day he broke down, came out 0.40, "an underload"; spread, it is
  1.32 (N-018). The body bowls a wide, so it is load. A day after the snapshot ends would make
  the week look empty. "A normal load" was dropped as words: the ratio compares him only
  with himself (L-036).
- Alternatives considered: all balls on a match's first day (wrong by up to four days in a
  Test); legal balls only (a wide is still bowled); the uncoupled ratio (the four weeks before
  this one: avoids counting this week twice, but leaves the studies' form); a weighted
  average over the days (proposed in the critiques, not standard in cricket); an absolute-load
  flag (no cut-off with evidence behind it yet).
- Status: built (Day 41). M3 reads the deliveries, not `bowler_days`.
- Paper section: Method (N8 M3); Limitations (L-003, L-036, L-037)

## D-045 — M4, DRS reviews: by team, and never by dismissal type
- Date: 2026-10-22
- Decision: M4 gives a team's review overturn rate, men's or women's: the reviews whose
  decision was changed ("upheld") over those decided ("upheld" or "struck down"). Umpire's
  call is a decision that stood. A technical failure decided nothing and is left out. A
  team's batting and fielding reviews are counted together. There is no rate by dismissal
  type: the menu row now needs a team and a gender, and a dismissal type is an extra field
  that gets no answer.
- Why: Cricsheet keeps no appeal kind with a review; the dismissal is known only when the
  batter ended up out. A batting side's review that succeeded, and a fielding side's that
  failed, leave no dismissal, so a rate by kind would see only the batters' failures and
  the fielders' successes: 0% and 100%. The archive holds 17,153 reviews: 4,704 upheld,
  12,447 struck down (2,094 of them on umpire's call) and 2 technical failures.
- Alternatives considered: a rate by kind over the reviews whose kind is known (lopsided,
  as above); umpire's call as its own outcome (Cricsheet records it as struck down); one rate
  for each side (batters' reviews change the decision more often, 35.0% against 21.4% in
  men's cricket; a filter for Day 43).
- Status: built (Day 42). The load of Day 30 did not keep the review's umpire (recorded for
  16,851 reviews) or its type; a rate by umpire, for N11i, needs a re-load.
- Paper section: Method (N8 M4); Data
- The v4 document's "overturn rates by team and by dismissal type" changes at its next
  revision.

## D-046 — Men's and women's cricket are never mixed
- Date: 2026-10-22
- Decision: a subject has a gender ("male" or "female", Cricsheet's words). A ground or a
  team must name it: the M2 and M4 menu rows need it. A ball carries its gender, and "like
  with like" (D-042) is by format and gender: a man is set against other men's balls only.
  M1 takes gender as a filter. N3 reads a team or ground question that names no gender as
  men's, and says so in its read-back (Day 58), so the engines never guess.
- Why: Cricsheet names a country's men's and women's teams alike ("India"). At Lord's the
  men's T20 first-innings average is 171.1 and the women's 146.4; mixed, it read 167.7. T20
  strike rates are 125.8 for men and 96.9 for women, and women's balls (1.39 of the 11.69
  million) pulled "everyone else" down: Kohli's +10.9 against everyone else is +8.6 against
  other men (N-016).
- Alternatives considered: a default to men's cricket inside the engines (a hidden choice
  nobody sees); separate team names (Cricsheet uses one); a gender filter on Day 43 only
  (the Day 39 and Day 40 numbers would have stayed mixed until then).
- Status: built (Day 42); D-042 and D-043 corrected with it
- Paper section: Method (subjects; N8)

## D-047 — What a format's name means
- Date: 2026-10-22
- Decision: a subject's `match_format` is one of: "Test"; "ODI"; "T20I" (Cricsheet's T20
  between international sides); "T20" (every twenty-over match, club and international,
  with Cricsheet's "IT20" too); or Cricsheet's own codes "IT20", "ODM" (one-day, not ODIs)
  and "MDM" (multi-day, not Tests). Any other name gets no answer.
- Why: Cricsheet files a T20I and an IPL match alike as "T20" (team type tells them apart),
  and keeps "IT20" for 320 other internationals (regional qualifiers). A reader's "T20I"
  means internationals only: Kohli's T20 strike rate mixes in the IPL; his T20I one is
  136.0 on 2,915 balls (N-020).
- Alternatives considered: Cricsheet's codes alone (a reader's "T20I" is not one of them);
  "T20I" with "IT20" in it (they are not the official internationals a reader means).
- Status: built (Day 43). The Hundred's 100-ball matches are "T20" too: fine for rates per
  ball, and M2 leaves them out by length (D-043).
- Paper section: Data; Method (N8 filters)

## D-048 — Filters times measures: one engine, many questions
- Date: 2026-10-22
- Decision: every filter is a Subject field. New fields: `opposition`, `season` ("2024" or
  "2024/25", as Cricsheet writes it), `since` (a year or a day, that day included),
  `innings`, `batting_hand` and `side` (who asked for a review). Each menu row lists the
  fields it may be narrowed by (`filters`); a filled field it neither needs nor lists gets
  no answer, never a quietly different question. A batter's measures take bowler,
  opposition, ground, season, since, format, innings, phase, bowler type and gender; a
  bowler's take the batter (head to head), opposition, ground, season, since, format,
  innings, phase, batting hand and gender. "Opposition" is seen from the side the subject
  is on: a batter's opponents bowled, a bowler's batted. The match filters (format,
  gender, season, since, ground) are one shared piece of SQL for every engine; M2 takes
  season and since, M4 format, season, since and side. Every value goes in as a ?.
- Why: the v4 document's example, "Kohli's batting average against Australia since 2020",
  needs four filters at once; a fixed list of questions would never be finished. A
  question the engine cannot narrow must not be answered as if it had been.
- Alternatives considered: one claim type per question (endless); filters in free text
  (unchecked); language-model SQL (most such SQL runs and is wrong, CricBench).
- Status: built (Day 43)
- Paper section: Method (N8; the registry)

## D-049 — The batting and bowling measures
- Date: 2026-10-22
- Decision: batting (n is the balls faced): runs off the bat; batting average, runs per
  time out, where a run out at the non-striker's end counts as out (matched on the
  player's register id through the names of that match) and a player never out has no
  average; boundary share (balls faced hit for four or six); dot-ball share (balls faced
  with nothing off the bat, so a ball with byes is a dot). Bowling (n is the legal balls):
  wickets the bowler is credited with (bowled, caught, caught and bowled, lbw, stumped,
  hit wicket); economy, runs off the bat plus wides plus no-balls per six legal balls
  (byes, leg byes and penalty runs are not the bowler's); bowling strike rate, legal
  balls per wicket, none without a wicket. Each is a menu row, an engine function and
  renderer words.
- Why: these are the scorers' conventions, so published records can check them: Kohli's
  Test average comes out at 46.85 (9,230 runs, 196 outs on strike and one run out at the
  other end), as published (N-021).
- Alternatives considered: an average over the outs on strike only (misses the run outs a
  batting average counts); economy per ball (cricket reads it per over).
- Status: built (Day 44)
- Paper section: Method (N8 M1)

## D-050 — M5, match results
- Date: 2026-10-22
- Decision: a new engine, N8 M5, reads the matches table. `team_win_share`: of a team's
  matches that someone won, the share it won (ties, draws and no results left out; a new
  unit, matches, with a C3 floor of 10). `match_won_by_runs` and `match_won_by_wickets`:
  the margin of one match, named by its winner, its opponent, its day and its gender. If
  that team did not win it, there is no answer, so a false "X beat Y" fails C2. An
  innings win is not a win "by runs". One match's result is a fact, not a sample: its
  menu row says `sample=False`, and C3's floor skips it.
- Why: the plan's "match results (winner, margin)"; a result names a winner, which is
  text, so the winner goes in the subject and C2 checks it by asking. Real checks:
  Pakistan beat India by 10 wickets on 24 October 2021, Bangladesh beat India by 7
  wickets on 3 November 2019, and the opposite claims get no answer (N-022).
- Alternatives considered: the winner as the claim's value (C2 compares numbers);
  margins inside M1 (they are not counted from balls); a C3 floor of one match.
- Status: built (Day 45). The v4 document lists M1-M4; M5 joins it at its next revision.
- Paper section: Method (N8 M5; C3)

## D-051 — The real FactSource
- Date: 2026-10-22
- Decision: `EngineFactSource` maps each menu metric to its engine function and the
  repository it reads; `engines_on(db)` wires the five DuckDB adapters to one connection.
  A metric with no engine yet, or a clip's number, gets no answer, so the Faith Check
  fails closed on it. Its `sources()` feed the registry check, which now names only the
  rows still to build: weight_transfer (N6) and the two N9 rows.
- Why: the Faith Check had only spoken to a fake. Now C2 asks the real engines: on the
  fixtures a true number passes, and a wrong number, the right number for the wrong
  player, an inflated sample and a player the data does not know each fail (E-027).
- Alternatives considered: each agent calling the engines itself (the gate's own
  re-query is the point: I8); one adapter for all engines (five small ones are easier to
  test).
- Status: built (Day 46)
- Paper section: Method (the Faith Check; N8)

## D-052 — A receipt for every number
- Date: 2026-10-22
- Decision: the real FactSource stamps a receipt on every answer it gives the gate: the
  data's fingerprint, the engine and its version, the exact question as one line of
  canonical JSON (the metric, the subject's filled fields, the clip), and the number with
  its n and range. The fingerprint is one SHA-256 over the Cricsheet zip's own SHA-256 and
  every player's batting hand and bowler type (the inputs from outside the zip). An
  engine's version is a number in `ENGINE_VERSIONS`, raised when one of its counting rules
  changes. A receipt is not part of the number: two facts with the same number are equal
  whatever their receipts. Receipts come from the gate's own re-query, never from an agent.
- Why: invariant I8; anyone can re-make a number. On the real data the fingerprint takes
  0.04 s, and the same question twice gives the same receipt.
- Alternatives considered: a checksum of the DuckDB file (its bytes change when it is
  opened and written, not only when the data does); the git commit as the engine's version
  (the running app may have no git; a hand-raised version says what changed); the SQL text
  in the receipt (an adapter's detail; the question plus the engine's version decide it).
- Status: built (Day 47)
- Paper section: Method (receipts, I8)

## D-053 — The receipt checker
- Date: 2026-10-22
- Decision: `check_receipt` re-asks the engines the receipt's question and compares,
  exactly: the number, n, the range and the engine's name. A receipt from other data is
  not re-run ("other data"); a receipt from an older engine version is re-run and says so.
  `python -m cricketmind_v3.pipeline.check_receipts <database> <receipts file>` reads one
  receipt a line, opens the database read-only, says one line per receipt, and exits 1 if
  any is not confirmed.
- Why: exact comparison works because every engine is deterministic, the bootstrap
  included (seed 42): on the real data, 9 receipts across the five engines were confirmed,
  and a tampered one (Kohli's Test average as 50) was caught (N-023).
- Alternatives considered: comparing within rounding (a tampered 46.86 would pass);
  re-running receipts from other data (a different number would look like tampering).
- Status: built (Day 48); the final exam's receipts are checked on Day 178 (E-022)
- Paper section: Method (receipts); Evaluation

## D-054 — What-if pictures: drawn from the engine, in forms people read correctly
- Date: 2026-10-25
- Decision: a prediction is shown as its sentence first, then pictures drawn only from a
  small summary the engine keeps (with the receipt), never computed in the browser and
  never by the language model: 100 simulated overs as dots, with the range (a quantile
  dotplot in cricket form); the chance of still batting after each ball, with its band (a
  survival "worm"); and for a matchup, the two sides' simulated overs animated side by
  side (hypothetical outcome plots). Labelled "simulated"; shown only once Day 53's
  calibration passes. Built with visx (SVG) and Motion (animation). For a clip, the
  measured skeleton is drawn over the real frames in 2D; a 3D skeleton is a stretch goal
  only. Ball tracking, pitch maps and wagon wheels are never drawn: the data has no line,
  length or direction. Whether the human study compares sentence alone with sentence plus
  dotplot is decided on Day 188.
- Why: people read chances more accurately from discrete outcomes than from curves (Kay
  et al. 2016); text and quantile dotplots kept trust highest over time ("In Dice We
  Trust", CHI 2024); FiveThirtyEight showed 100 dots drawn from 40,000 simulations;
  hypothetical outcome plots beat error bars when comparing two things (Hullman et al.
  2015; Kale et al. 2019). Cricket fans already read CricViz's win-probability worm.
- Alternatives considered: a density curve (harder to read correctly); 3D ball paths
  (would be invented data); a general chart library such as Recharts (too rigid for the
  overs grid) or ECharts (built for large data); pictures drawn by the language model
  (unchecked).
- Status: planned (Days 49, 51-52, 100, 108, 127; optional on Day 188)
- Paper section: System (the report page); Evaluation (the human study, if chosen)

## D-055 — Agentic retrieval, where it cannot touch a number
- Date: 2026-10-25
- Decision: agentic retrieval-augmented generation (an LLM that retrieves, judges what it
  found, and retrieves again) is used in three places, none of which makes a number.
  (1) A third baseline in the evaluation (Days 180-181): the same local model writes SQL on
  the same DuckDB, grades the result, rewrites and retries, at most three rounds, with no
  Faith Check; its numbers are scored like the other systems'. (2) N3's names and grounds
  (Day 58): retrieve candidates from the registers, weigh the question's clues, keep one
  register id or ask which one; at most three look-ups. (3) Optional: the Explainer's
  rules answers (Day 72), with corrective retrieval over the Laws of Cricket and a check
  that every cited clause exists, only if MCC permits storing the Laws. The engines, the
  numbers, the Faith Check and the receipts stay as they are.
- Why: a 2026 reviewer will ask why not agentic RAG; the fair answer is to run it on the
  same exam. On CricBench, models write correct cricket SQL only about a third to a half of
  the time (GPT-4o 33.7%, DeepSeek R1 50.6%). For names, the loop fixes known gaps (L-028,
  L-029, L-031), and research on entity-linking agents and clarifying questions supports
  asking only when candidates stay close. The Laws are words, where retrieval is the right
  tool; MCC reserves their copyright.
- Alternatives considered: agentic RAG at the core (the model would see raw data and make
  numbers, and the cost would no longer be bounded); no agentic retrieval at all (the
  obvious modern baseline missing from the paper); web search for "can't check" questions
  (unverified text in a system built on checked facts).
- Status: planned (Days 58, 72 optional, 164-165)
- Paper section: Evaluation (Baseline 3); Method (N3; the Explainer)

## D-056 — A stronger out-risk model only if the test says so
- Date: 2026-10-25
- Decision: N9.T1 starts simple (Day 49: one out-rate per ball from the batter's record,
  with its Beta uncertainty). Day 50 adds shrinkage, and puts one extra on trial: the risk
  per ball depends on how long the batter has been in (balls faced so far: 1-5, 6-15,
  16-30, after 30), each stage shrunk toward the average the same way. Day 53 decides it:
  fitted on older seasons and tested on the newest, the stages stay only if the Brier score
  is better and the calibration no worse, a rule written down before the run. Other extras
  (recent form, the match situation) are not tried now; line, length, speed, field and
  pitch are not in the data and are never pretended.
- Why: the user asked how to make the simple model stronger; "stronger" must mean "predicts
  better when tested", not "more complicated". Being out early in an innings is a commonly
  held cricket belief, and the balls faced so far can be counted from the data. A model that
  fails calibration is weaker, however detailed.
- Alternatives considered: a full ball-by-ball model with every feature (much more to build
  and test; harder to explain; kept for future work); no extra at all (leaves an obvious,
  testable improvement untried).
- Status: built on Day 50 (D-058); not kept on Day 53 (D-066)
- Paper section: Method (N9 T1); Evaluation (E-007)

## D-057 — N9 T1: the chance of being out within k balls, with an honest range
- Date: 2026-10-25
- Decision: the batter's outs and balls faced under the question's filters (counted as the
  dismissal rate counts them, D-039) give a Beta curve of believable out-rates per ball,
  Beta(0.5 + outs, 0.5 + balls not out on): the Jeffreys prior. 20,000 rates are drawn
  (seed 42); at each, the chance of being out within his next k balls faced is
  1 - (1 - p)^k, what endless replays at that rate would give. The answer is the middle of
  those chances, the range their middle 95%, and n the balls faced. The menu row needs only
  the batter and k; shot, phase, bowler type and the batting filters narrow his past; a shot
  gets no answer until the bridge. The fact keeps a summary for the pictures (D-054): the
  chance of still batting after each ball with its band, and 100 overs played ball by ball
  that stand for the replays; like a receipt, it is not part of the number.
- Why: v2 gave 0.394 with a range of 0.387-0.400, luck only (E-003). The rate itself is
  uncertain: on the same 4 outs in 50 balls the honest answer is 0.405 with a range of
  0.155-0.697 (N-004), about forty times wider, and the replays land within 0.01 of the exact
  answer found by adding up the curve. On real data, Kohli's T20 powerplay against pace
  (2,855 balls) gives 16% (13-20%), Bumrah batting in T20s (107 balls) 51% (32-70%): little
  evidence, wide range. The shot, required before, has no data until Day 129, and Day 53's
  calibration needs real questions.
- Alternatives considered: the point estimate alone (v2's mistake); a flat prior, Beta(1, 1)
  (pulls tiny samples harder); playing balls inside each draw for the answer too (adds luck
  without adding anything); pulling small samples toward other batters (Day 50).
- Status: built (Day 49)
- Paper section: Method (N9 T1)

## D-058 — N9 T1 pulled toward normal (shrinkage); how long he has been in, on trial
- Date: 2026-10-26
- Decision: the prior for a batter's out-rate comes from the other batters under the
  question's filters, in his format and his gender: their normal rate, and how much they
  really differ, found by the method of moments (the spread of their rates, less the share
  luck alone would make). It is a Beta curve worth so many balls, Beta(normal x worth,
  (1 - normal) x worth), and it takes the place of Day 49's Jeffreys prior. Fewer than 30
  other batters (FEW_BATTERS, a design parameter), or no real difference seen between them,
  gives no answer. The menu row now needs the format as well as the batter and k: normal is
  for one kind of cricket (like with like, D-042). On trial (D-056): a question may say how
  many balls he has faced (faced_so_far); then each of his next k balls takes the risk of its
  stage (balls 1-5, 6-15, 16-30, after 30), each stage with its own prior from the others in
  that stage. The stage of a ball is counted on the whole innings (a new derived table,
  ball_stages), so a filter such as the middle overs does not start him again. N9.T1 is
  version 2 (D-052).
- Why: on his own record, 1 out in 2 balls is a 50% risk a ball. Pulled toward normal, a real
  batter out once in his two balls in men's T20s gets 5.9% (2.8-10.5%) for his next ball.
  Big records barely move: Kohli in the T20 powerplay against pace goes from 16.4% to 16.8%
  for his next 6 balls (2,855 balls). Small ones move a lot: Bumrah batting in T20s, from 51%
  to 39% (107 balls; see L-045). The stages matter in the data: the risk a ball falls as the
  batter settles in Tests (2.3% to 1.4%) but rises in T20s (5.0% to 6.4%).
- Alternatives considered: a maximum-likelihood fit of the same curve (harder to explain,
  and it fails in the same cases); a fixed worth (a guess); a wider group of batters when the
  narrow one is too small (a second path that mixes situations); the format left open and
  his own mix of formats used (D-042's way for strike rates; a prediction is about the next
  balls, which are in one format); a stage's prior set relative to his own level in the other
  stages (borrows strength; more to build and explain; future work, not a second try if the
  trial fails, since D-056's rule is fixed before the run).
- Status: built (Day 50); the stages are decided on Day 53
- Paper section: Method (N9 T1)

## D-059 — Normal is for batters who bat where he bats
- Date: 2026-10-26
- Decision: the prior of D-058 comes only from the other batters' balls faced batting in
  the same part of the order as him: the top (places 1-3), the middle (4-7) or the tail (8
  on). His place is where he faced the most balls under the question's filters (the top
  first, if two are level); a nightwatchman's few balls do not move him. A batter's place in
  an innings is the order he came in, from the first ball he was seen on at either end (a
  derived table, batting_slots; the striker before the non-striker on the first ball). His
  own record stays all his balls. On trial with the stages: Day 53 compares this prior with
  the prior from every batter, by the same rule (D-056), written down before the run.
- Why: found the same day (L-045). Pulled toward every batter, a tailender looks safer than
  he is: the tail is out 9.22% a ball in men's T20s, the top order 4.82%. Bumrah batting in
  T20s (107 balls) went from 51% on his own record to 39% pulled toward everyone; pulled
  toward the tail he is 48% (33-63%). A tailender out once in his only two balls gets 9.9%
  for his next, not 5.9%. Top-order batters barely change (Kohli 16.8%). Batters in one part
  of the order also differ less from each other: the top's prior is worth 281 balls, every
  batter's 139.
- Alternatives considered: his place in each innings, not his usual one (the same batter
  would be pulled toward different groups ball by ball); exact places (too few batters at
  each); a prior that also leans on his batting average (borrows from other numbers; more to
  explain); leaving it as a limitation (a known, large bias, and the data to fix it exists).
- Status: built (Day 50); kept on Day 53 (D-066)
- Paper section: Method (N9 T1)

## D-060 — The prior is fitted by maximum likelihood, not the method of moments
- Date: 2026-10-26
- Decision: the prior of D-058 is fitted by maximum likelihood: the normal rate and the
  worth under which the other batters' records were the most likely (a Beta curve behind
  each batter's rate, the beta-binomial), found by golden-section search, the worth then
  the normal rate, five times over. No prior, so no answer, when there are fewer than 30
  others, when every one faced a single ball, or when the fit finds no sign at all that
  batters differ (the worth at its ceiling of a million balls): pulled all the way to
  normal, the range would be falsely sure. N9.T1 is version 3 (D-052).
- Why: found on Day 51, while fitting how much meetings matter. A batter bats until he is
  out, so his balls depend on his outs; the method of moments assumes the balls were fixed
  beforehand, the likelihood does not mind (the likelihood principle). On made-up innings
  with a known truth, the moments found 5.26% and worth 163 for 5.5% and 200, and 1.47% for
  a Test-like 1.7% (too low: the good batters last longest, so a balls-weighted rate leans
  to them); the likelihood found 5.51% and 182, and 1.70%. For meetings the moments were far
  off: worth 7 when meetings add nothing, 6 when they are worth 50; the likelihood found
  1,167 and 53. On real data, top-order men in T20s: 5.22% and worth 683, where the moments
  said 4.82% and 277. In the fixtures, 11 batters showed no difference, and the fit would
  have pulled Farhan to a range of 27.6-28.0%: hence the refusal at the ceiling.
- Alternatives considered: keeping the moments (biased, and badly so for meetings); a full
  Bayesian fit that also carries the doubt in the normal rate and the worth (more honest for
  small groups; slower; future work, L-049); a fixed worth (a guess).
- Status: built (Day 51)
- Paper section: Method (N9)

## D-061 — N9 T2: what if this bowler bowled instead of that one?
- Date: 2026-10-26
- Decision: the question names the batter, the bowler of the what-if, the bowler he would
  replace, k and the format; the phase, venue, season, since, innings and gender narrow it
  (a bowler type or an opposition does not apply to two named bowlers). Every ball under
  the filters, in his kind of cricket and his part of the order (D-059), gives three
  records: his against every bowler, pulled toward batters like him; each bowler's against
  batters in that part of the order, pulled toward normal for bowlers; and their meetings.
  What to expect when they meet is the odds rule (Bill James's log5): his odds of being out
  times the bowler's, over the normal odds (every ball's rate). The meetings nudge that
  only as far as meetings are worth, fitted by maximum likelihood over every pair (D-060).
  Each of 20,000 replays draws one batter and one of each bowler, so the difference is the
  bowler alone, not luck: the answer is the chance against the bowler minus the chance
  against the other, with the middle 95% of the differences; a range across zero is "no
  detectable difference" (C4). n is the thinnest record the answer rests on (his balls,
  each bowler's balls to batters like him), not the meetings, which may be none. For the
  pictures (D-054), both sides keep their 100 overs, paired over for over with the same
  luck. N9.T2 is version 1. The menu row changes: the phase is a filter, not a need, and the
  format is needed (D-058).
- Why: meetings are few (Kohli and Bumrah: 5 outs in 102 balls in T20s) and, the data
  shows, add almost nothing beyond the two records: in men's T20s the fit puts them at its
  ceiling, in Tests at about 8,000 to 25,000 balls. Baseball found the same for batter
  against pitcher (Tango, Lichtman and Dolphin, 2007). So most real what-ifs come out as
  "no detectable difference": Kohli in T20s, Bumrah instead of Shami, +0.0 points (-2.5 to
  +2.5). A made-up field where meetings do matter shows the engine then listens to them.
- Alternatives considered: the meetings alone, pulled toward normal (5 in 102 is mostly
  luck, and it ignores who they are); T1 with the bowler as a filter (pulls toward other
  batters against that bowler, ignoring the batter); his record alone (ignores the bowler);
  two independent simulations (the difference would carry his doubt twice); n as the
  meetings (would refuse most what-ifs that the two records answer well).
- Status: built (Days 51-52); calibrated on Day 53
- Paper section: Method (N9 T2)

## D-062 — An ML baseline for the out-risk engine, judged by the same test
- Date: 2026-10-26
- Decision: after Day 53's calibration, on its exact split (fitted on the older seasons,
  tested on the newest) and the same questions, two machine-learning models predict the
  same thing as N9 T1, the chance of being out within the next k balls: logistic
  regression, and gradient-boosted trees (scikit-learn). Their inputs are only what the
  data holds at the moment of the ball, from matches before that day: the batter's and the
  bowler's records, the format, phase, balls faced so far, place in the order, innings,
  wickets down, and the chase (runs needed, balls left). No line, length, speed, field or
  pitch: the data has none (L-044). The rule, written down on Day 53 before any model is
  fitted: an ML model is taken up only if its Brier score is better than N9 T1's by more
  than the score's own uncertainty (a paired bootstrap over the test balls), its
  calibration is no worse, and it can give an honest range; otherwise it is reported as a
  baseline. Taken up, it would go through the same receipts and Faith Check: seeded,
  versioned, run again to check. The code stays apart from the engines (an evaluation
  script), so the domain stays plain Python. The result goes in the paper either way.
- Why: the user asked whether the predictions should come from an ML or DL model run again
  and again. N9 already runs a model 20,000 times; the honest test of ML is whether it
  predicts the newest season better (D-056's rule: stronger only if tested). On these
  inputs, who faced whom and the match situation, a learned model may add the situation,
  or may add nothing; a paper that tried it and reports the outcome is stronger.
- Alternatives considered: a deep network (needs far more data, harder to calibrate, and on
  table data gradient-boosted trees usually match or beat deep networks: Grinsztajn et al.,
  NeurIPS 2022); ML instead of N9 from the start (no honest range, small samples learned as
  noise, and hard to check by receipt); no baseline (leaves the obvious question open).
- Status: planned (rule and features written on Day 53; run on Day 54; joined by tabular
  transformers and the ball model on Day 157; in Day 161's evaluation plan)
- Paper section: Evaluation (N9 against ML baselines)

## D-063 — Learned models wherever they can be tested fairly
- Date: 2026-10-26
- Decision: machine learning, deep learning and transformers are tried wherever a fair test
  exists, as an engine, as a helper that never makes a number, or as a comparison:
  (1) the ball model, a small transformer trained on every ball, against N9 T1 and T2
  (Phase 9c, D-064); (2) plain ML (logistic regression, gradient-boosted trees; Day 54,
  D-062) and tabular transformers (TabPFN-2.5, TabICL; Day 157); (3) "batters like him" from
  learned player vectors, against batting place (Day 158); (4) a third AI checker, an open
  hallucination detector (LettuceDetect, MiniCheck or HHEM-2.1; Day 165); (5) a small
  fine-tuned router (SetFit) beside the keyword and LLM routers (Day 173); (6) small models
  for reading a question: GLiNER for names, Qwen3-Embedding to match them to the register,
  SetFit for the kind (Day 176); (7) prompts tuned by DSPy beside hand-written ones (Day
  177); (8) a transformer pose model (ViTPose) beside the first (Day 125); (9) the bridge's
  labels distilled into a small fine-tuned transformer (ModernBERT) beside the rules and the
  LLM (Day 137); (10) with MCC's permission, hybrid retrieval over the Laws (Day 72). Each
  has a rule written before its run. A learned model makes a number a user sees only behind
  the engine port, with receipts the Faith Check can ask again (rules 6 and 10); none is
  trained or tuned on the locked exams; GPU work follows the device rule and is resumable;
  the CricShot classifier's accuracy stays out, and video classifiers are compared only in
  CricShot's paper.
- Why: the user asked for ML, DL and transformers wherever they can be used, at least for
  comparison. A thesis that tries the modern methods and reports where they help and where
  they do not is stronger than one that never tries. The search (2026-10-26) found no
  transformer trained on ball-by-ball cricket, while football has event transformers
  (Mendes-Neves et al., 2024; EventGPT and ScoutGPT, 2025-26): a gap the ball model fills. A
  small encoder beat a zero-shot LLM at a fixed sorting task in one test (SetFit 91% against
  69%); hallucination detectors are small and fast (LettuceDetect, 2025); tabular
  transformers are now strong (TabPFN-2.5; TabICL, ICML 2025); transformer pose models are
  usually more accurate than MediaPipe, which is faster.
- Alternatives considered: a transformer instead of N9 from the start (no honest range of its
  own, hard to check, small samples learned as noise); video classifiers compared here (that
  is CricShot's paper); large models only through an API (cost, and the locked exam must run
  locally too).
- Status: planned (Days 54, 72, 125, 137, 152-160, 165, 173, 176, 177)
- Paper section: Method; Evaluation (learned models against the simpler ones)

## D-064 — The ball model: a small transformer on every ball, and its rule
- Date: 2026-10-26
- Decision: each innings is a sequence of balls, each ball a few tokens: the match state
  (format, gender, innings, over, wickets down, runs and the chase), the batter, the bowler,
  his place in the order, his balls faced so far, and the outcome (a dot, runs, extras, the
  kind of wicket). A small decoder-only transformer, about ten million weights, learns to
  predict each ball from those before it, trained only on the seasons Day 53 fits on. Its
  chance of a wicket on each ball, rolled forward 20,000 times, gives the chance of being out
  within k balls; swapping the bowler gives the what-if. Five copies trained with different
  seeds (a deep ensemble) give the range; temperature scaling on a held-out season
  calibrates it; if its 95% ranges do not hold about 95% of the time, Venn-Abers or conformal
  calibration is tried, and said. The size, inputs and training budget are written down
  before the first run. The rule, written now, before any training: it is taken up as an
  engine only if, on the newest season, its Brier score beats N9's by more than the score's
  own uncertainty (a paired bootstrap over the test balls), its calibration is no worse, and
  its ranges hold. Then it runs behind the engine port with receipts: its weights' SHA-256 as
  its version, inference on the CPU, seeded rollouts. Otherwise it is a reported baseline.
  Its batter and bowler vectors are tried as "batters like him" against batting place
  (D-059), by the same rule. Training resumes after a power cut and never shares the GPU
  with CricShot.
- Why: N9 knows a batter, a bowler and a few situations; a sequence model could also learn
  the match situation, the moment in the innings and how players meet, from 11.7 million
  balls. No cricket work found trains a transformer on ball-by-ball sequences; football's
  Large Event Models do it for events (Mendes-Neves et al., 2024), and (batter|pitcher)2vec
  learned players from baseball's at-bats (Alcorn, 2018). The fair test decides; either
  outcome is a result.
- Alternatives considered: an LSTM (used in cricket win-probability work; weaker over long
  sequences); a large pretrained LLM reading balls as text (slow, costly, hard to
  calibrate); a table model on single balls (no sequence; tried as a baseline on Day 157).
- Status: planned (Phase 9c, Days 152-160)
- Paper section: Method (the ball model); Evaluation

## D-065 — The Faith Check's word lists against models that judge meaning
- Date: 2026-10-26
- Decision: on Day 166, the word lists of C3 (warnings), C4 (hedges and band words), C6
  (names) and C7 (tone words) are set against small transformers that judge meaning: a
  DeBERTa-v3 model trained for natural-language inference, for hedges, certainty and tone,
  and GLiNER for names. The test set is written first: each of the 8 lies and each correct
  finding rewritten five ways ("struggles against spin", "is all at sea against spin"),
  labelled before running, never taken from a locked exam. Both judge every version; for
  each check, the lies missed and the honest findings wrongly stopped are counted. The rule,
  written now: a model is taken into the gate only if it catches lies the lists miss without
  stopping more honest findings than the lists do, and runs the same way every time on the
  CPU. Even then it is a second reader: it can stop a finding, never let one through that
  the lists stop, and its reason names the sentence it judged. Otherwise the lists stay,
  and the measured misses go in the paper.
- Why: the user asked whether finished work needs ML or a comparison. The counting engines
  and the exact checks (C1, C2, C5) do not: they have one right answer. The word lists do:
  their known weakness is that a word not on a list slips past (L-019), and a model that
  reads meaning may catch paraphrases. Measuring both turns a written weakness into a
  result, whichever way it goes.
- Alternatives considered: replacing the lists with a model (a gate whose reasons cannot be
  read, and that can let a lie through); growing the lists by hand only (never measured
  against paraphrase); the hallucination detector of Day 165 alone (it judges a whole
  finding, not which word failed which check).
- Status: planned (Day 166)
- Paper section: Evaluation (the word lists against meaning); Limitations (L-019)

## D-066 — Day 53's verdicts: batting place kept, stages out, and how calibrated it is
- Date: 2026-10-26
- Decision: by the rules written before the run (`thesis/prereg/day53_calibration.md`,
  SHA-256 1a7070c9...ff34), on every test ball of men's and women's T20s, men's ODIs and
  men's Tests after 2025-10-14, fitted only on the matches before: the prior by batting
  place stays (B over A, D-059); the stages do not (C over B, D-056): they leave the engine
  and the menu (`faced_so_far`), and stay only as a baseline for Day 157. The engine's
  chances are called calibrated, per ball and within 6 balls, on men's and women's T20s and
  men's ODIs, but not on men's Tests, where the gap is 13% of the mean chance, above rule
  3's 10%: the paper says so, and any recalibration is a later decision, fitted on the
  validation year, never on the test year. T2 adds nothing measurable per ball over T1;
  it stays for what-ifs, whose differences then come out as "no detectable difference",
  as they should.
- Why: per ball, Brier(B) - Brier(A) has a 95% range of -0.0000317 to -0.0000125, wholly
  below zero, and B's calibration gap is the smallest (0.00103 against A's 0.00166);
  Brier(C) - Brier(B) ranges from -0.0000180 to +0.0000013, so C's edge may be luck. The
  shrinkage itself earns its place: his record alone (Day 49) scores 0.047443 with a gap of
  0.00999; pulled toward normal, 0.045655 and 0.00103, ten times better calibrated. The
  engine, asked itself on 30 moments, agreed with the run to 0.075 points at most.
- Alternatives considered: deciding per kind of cricket (C was better on Tests; but the rule
  was pooled and set before the run, and a new per-kind rule now would be fitted to the
  test year); a looser rule (better Brier alone, with no allowance for luck: written before
  as stricter, and kept so).
- Status: decided (Day 53); the stages left the engine and the menu in the commit after
  033b4b7, Day 53's own
- Paper section: Evaluation (calibration; the two trials)

## D-067 — Day 54's ML verdict, and what "taking N9's place" can mean today
- Date: 2026-10-26
- Decision: by the pre-registered rule (section 7): logistic regression is a baseline (its
  Brier score is lower by more than luck, but its calibration is clearly worse); the
  gradient-boosted trees pass the rule's first two parts (Brier lower by more than luck,
  calibration not clearly worse), and its third, an honest range from 20 refits on
  resampled training matches, is run as its own resumable step (`--ranges`). But the trees
  learn from the match itself (the score, the wickets, the balls left, the bowler), and a
  question to the engine carries none of it. An exploratory run, not pre-registered and
  deciding nothing, gave the same trees only what a question carries: they still beat N9,
  by about half as much. So the menu's questions stay N9's for now: the pre-registered trees
  cannot answer a question that does not state the match, and the question-only trees were
  not pre-registered. The trees become the bar Phase 9c's ball model (which reads the match
  too) must clear; a menu that can carry the match situation, with receipts for a learned
  engine, is taken up there (Days 156, 159); and the question-only trees join Day 157's
  table, pre-registered before it runs.
- Why: on 487,381 test balls the trees score a Brier of 0.045590 against N9's 0.045882 (95%
  range of the difference -0.00032 to -0.00026), and on men's Tests they are calibrated (gap
  8% of the mean) where N9 is not (13%). A model is only as useful as the questions it can
  be asked: filling a missing match situation with an average would be an invented input.
- Alternatives considered: replacing N9 with the trees now, guessing the match situation when
  a question lacks it (invented inputs); calling the trees a baseline whatever the rule said
  (ignores the result); counting the exploratory run as a verdict (fitted after seeing the
  pre-registered result).
- Status: decided (Day 54). The honest-range step ran on 2026-10-27 (N-043): the trees pass
  all three parts of the rule, and the verdict stands for the reasons above, since the
  rule's passing model still cannot be asked a question that does not state the match.
  Its range is the doubt of the fit, not of the batter's rate (L-064). Phase 9c takes the
  rest up
- Paper section: Evaluation (N9 against ML baselines)

## D-068 — The engine report, and receipts that hold to the last digit
- Date: 2026-10-26
- Decision: the engine report (M2) is a script: it asks the real engines 29 questions from
  Days 39 to 52, sets M1 against five published career records, counts L-034's near-zero
  ranges over 150 real ranges, and copies in the calibration and the ML baseline; every
  engine number carries a receipt, and all are checked again on the same data before the
  report is written. Every query whose rows an engine sums over is ordered: the pairs
  (T2), and the others' tally (M1), with a test pinning the pairs' order.
- Why: the first run confirmed 192 of 194 receipts. The two failures were T2's: DuckDB's
  GROUP BY handed the pairs back in a different order each run, the sums came out in
  another order, and the fitted prior moved in its eighth digit. Receipts are checked to the
  last digit, so the engine must give the very same number every time. After the fix: 194
  of 194, and T2 the same on two runs.
- Alternatives considered: checking receipts to a tolerance (a tolerance hides real changes
  too); rounding engine answers (the same, earlier).
- Status: built (Day 54)
- Paper section: Method (receipts); Evaluation (the engine report)

## D-069 — N9 mixed with ML: three ensembles, on Day 157
- Date: 2026-10-27
- Decision: Day 157 adds three mixes of N9's chance with the question-only trees (the
  trees given only what a question carries, D-067), so that every mix can answer the
  menu's questions: (1) the two chances averaged, nothing learned; (2) stacking, a logistic
  regression on the two chances, its weights learned on the validation year from N9 and
  trees fitted on the years before it, then used with both fitted on every train year;
  (3) boosted trees that start from N9's chance and learn only a correction (LightGBM's
  starting score), whose range is N9's Beta curve moved by the correction, so it carries
  the batter's doubt as well as the fit's (L-064). The three are fixed today, written into
  Day 157's pre-registration before the run, judged by Day 54's rule (D-062), and all
  reported; none is picked after the run. A mix that passes the rule could answer the
  menu's questions, so it would need receipts the way Day 159 gives them to the ball model.
- Why: the user asked whether the statistical model and ML could be used together. A mix
  helps when two models make different mistakes; N9 is sure where data is thick and
  careful where it is thin, and the trees find patterns N9 has no slot for. The trees
  already take N9's batter rate as an input, so the gain may be small; only the test can
  say. Mixing only the question-only trees keeps the mix askable, which the full trees are
  not.
- Alternatives considered: trying mixes on the test year now (it has been looked at, so a
  mix picked after seeing it could win by luck); mixing the full trees (they need the match
  situation no question carries); more than three mixes (each one more is one more chance
  to win by luck); a new plan day (Day 157 already holds the table every model joins).
- Status: planned (Day 157)
- Paper section: Evaluation (N9 against ML)
