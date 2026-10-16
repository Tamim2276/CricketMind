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
- Status: designed
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
