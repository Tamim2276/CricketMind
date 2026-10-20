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
- Status: designed
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
- Status: designed
- Paper section: Method (Faith Check C4)

## D-007 — Claim-type registry, checked at build time
- Date: 2026-10-16
- Decision: every claim type an agent may assert must resolve to an engine metric. This is
  checked when the graph is built, so a hollow agent is a build failure.
- Why: four v2 agents (Stadium, Compliance, Injury, Sponsorship) assert claims no engine
  computes, and pass the gate because the number they cite is correct (E-002).
- Alternatives considered: catching these at run time with an LLM judge (not auditable).
- Status: partly built. Day 10: the registry (`domain/registry.py`), 10 claim types, each
  with its engine and the Subject fields it needs; `lookup()` and `missing_keys()`; a name
  listed twice, or keys written without the tuple comma, are refused. The build-time check
  itself comes on Day 11.
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
- Decision: evidence floor n_min = 30 balls; bridge acceptance κ ≥ 0.61 ("substantial", Landis
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
  (`"band": "about even"`) is not stored; it is computed from the value when needed (Day 13).
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
- Status: designed (built on Day 22; F8 in the harness on Day 23)
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
- Status: designed (Days 40–42, 54, 67, 144–145)
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
  cricket filters, and `FootballSubject` (Day 145) the football ones. The checks, the claims,
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
  counts that field as missing (tested with a plain `Subject`). `FootballSubject` on Day 145.
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
- Status: designed (Day 76; evaluated on Day 168)
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
- Status: designed (Days 152–154, 166–171)
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
