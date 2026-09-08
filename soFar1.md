# soFar1 — Session Record

A record of the CricketMind working session: what was decided, what was built, what
was learned, and what is still open.

---

## 1. What this session produced

| Artifact | State |
|---|---|
| `notebooks/cricketmind_langgraph_demo.ipynb` | **119 cells, 30 nodes, runs end to end with 0 failures** |
| `paper/CricketMind_Architecture_v2.tex` | 8 pages, compiles clean — 0 errors, 0 overfull boxes |
| `paper/CricketMind_Architecture_v2.pdf` | Compiled output with the full TikZ architecture diagram |
| `CricketMind_LangGraph_Build_Guide.md` | The 11-step build plan the notebook follows |
| `CricketMind_Completion_Estimate.md` | How long the full system takes (21–35 focused days) |
| `CricketMind_Simulation_Paper_Plan.md` | Whether/how to publish on simulation results |
| `CricketMind_Results_Plan.md` | What the results section contains, in plain terms |
| `CricketMind_5Day_Plan.md` | Day-by-day plan for methodology + results |

---

## 2. Architecture decisions

### The 14-agent roster

The original architecture PDF specified six generic "workers". A stakeholder table
listed twelve agents. Reconciling the two by set difference gave the missing pair:
**Stats Analysis** and **What-If Simulator** — the two workers with no stakeholder
of their own.

That produced the Version 2 structure:

- **12 stakeholder-facing agents** (N11a–N11l), one per cricket stakeholder, each
  with its own vocabulary and output format.
- **2 shared analytical services** (N8, N9) — computational engines with no
  audience, which are the **only components permitted to emit a number**.

This split is the central design claim: numeric truth is produced once by two
auditable engines, then narrated twelve different ways.

### Node numbering

The v1 figure jumped from N8f straight to N12, leaving N9/N10/N11 blank. The new
roster fills that gap exactly, so N1–N7 and N12–N13 keep their original meanings:

```
N8  = Statistical Analysis Engine
N9  = What-If Simulator
N10 = Supervisor Router
N11a–N11l = the twelve stakeholder agents
```

### Corrections to the original spec

| Item | Was | Now | Source |
|---|---|---|---|
| N4 temporal stage | GRU | **Temporal Transformer** (2 layers, 4 heads, learnable positional embeddings, mean pooling) | User decision; `src/models/temporal_transformer.py` |
| Frame count | 16 | **15** | `configs/transformer_local.yaml` |
| Shot class list | Invented placeholder names | **Real `CLASS_NAMES`** pulled from git history, cross-checked against `data/splits/train.csv` | `src/datasets/cricshot_dataset.py` |

The real 15 classes, in the order the trained head predicts:

```
Cover Drive, Defensive, Down The Wicket, Flick, Hook, Late Cut,
Lofted Legside, Lofted Offside, Pull, Reverse Sweep, Scoop,
Square Cut, Straight Drive, Sweep, Upper Cut
```

---

## 3. The notebook — built in ten steps

Each step ended with a checkpoint that had to pass before moving on.

| Step | Built | Checkpoint result |
|---|---|---|
| 0 | Installed langgraph 1.2.11, langchain-core, pydantic, ipykernel | Device resolves to `xpu` (Arc B580), bf16 autocast, no GradScaler |
| 1 | `CricketMindState` TypedDict, reducers, `new_state`, `describe_state` | Parallel writes merge; regeneration replaces rather than duplicates |
| 2 | N1, N2, N4, N5 stubs; linear graph | First run: `Cover Drive` at 0.92, shapes `(15,1280)` → `(1280,)` |
| 3 | Conf. Check gate + bounded retry loop | Three cases: accept / recover on retry / degrade at budget |
| 4 | N3 text encoder (parallel branch) + N6 Shot DNA join | Accept path joins; degrade path correctly skips N6 |
| 5 | N7a/N7b retrieval fan-out | 100 of 150 deliveries retrieved; SR 142/112 computed from rows |
| 6 | N8 Stats Engine, N9 What-If Simulator | 39.4% simulated vs 39.36% analytic; CI brackets the closed form |
| 7 | N10 Supervisor Router | Wakes 2 of 12 on the running example; caps at 4 under a query matching all twelve |
| 8 | Twelve agents from one factory; dynamic fan-out | Only routed agents run; citations verified against sanctioned facts |
| 9 | Faith Check + regenerate loop | Fabrication caught, agent re-run, corrected; persistent liar withheld |
| 10 | N12 synthesis, N13 output, full graph | 30 nodes, complete report generated |

### The final output

```
CRICKETMIND REPORT -- Kohli
========================================================================
Query      : Compare Kohli's cover drive vs his pull shot against fast bowlers
             in the powerplay
Vision     : Cover Drive at 92% confidence (1 attempt(s))
Evidence   : 100 deliveries -- Cover Drive SR 142.0, Pull SR 112.0
Simulation : 6 consecutive deliveries targeting the pull -> 39.4%
             (95% CI 38.7%-40.1%, 20,000 trials)

FINDINGS (2 agent(s))
------------------------------------------------------------------------
[Players]
  Weight transfer on the Cover Drive measures 0.88, and it is the stronger
  stroke at SR 142.0. The Pull lags at SR 112.0 -- shot selection, not
  technique, is the gap to close.

[Coaches]
  Bowl short at Kohli. The Pull returns SR 112.0 against the Cover Drive's
  SR 142.0; 6 consecutive back-of-a-length deliveries carry a 39.4% chance
  of dismissal.
```

Every figure traces to N8 or N9. The strike rates are counted from rows; the 39.4%
is Monte Carlo agreeing with `1 − 0.92⁶`.

---

## 4. LangGraph lessons learned the hard way

These cost real debugging time and are worth keeping.

### An edge is a trigger, not a data dependency

Figure 1 draws an arrow from N3 into N6, so the obvious wiring was
`add_edge("n3_text_encoder", "n6_shot_dna")`. That broke: N6 fired the moment N3
finished, while the vision branch was still at N2.

**LangGraph has no all-parents barrier.** A node fires as soon as *any* incoming
edge is satisfied. Four wirings were tested; only one worked — N6 triggered solely
by the gate's `accept` route, reading N3's output from **state** rather than through
an edge.

### When `defer=True` is needed, and when it is harmful

| Situation | Needs defer? |
|---|---|
| Parents at **different depths** (N6, N8, N9, N10) | **Yes** — the shallow branch would trigger early |
| **Siblings from one fan-out** (the twelve agents → faith_check) | **No** — one superstep completes as a unit |

Adding `defer=True` to `faith_check` was not merely redundant: it made
`get_graph()` draw a phantom `n13_final_output → faith_check` edge and drop
`n13_final_output → END` entirely. The runtime was correct throughout; only the
rendered diagram was wrong — and the diagram is what goes in the thesis.

### Reducers are for parallel writers only

`faith_failures` was declared `Annotated[list, operator.add]`. Wrong: only one node
writes it, and on the second pass through the regenerate loop the reducer appended
new failures to stale ones, so a corrected agent still looked broken. Changed to a
plain `list`.

`agent_findings` needs a **custom** reducer — plain `operator.add` would keep a
rejected finding alongside its replacement, and the hallucination would reach the
report anyway. `upsert_findings` keys on agent id so a regeneration replaces.

### Other gotchas

- A conditional function returns a **route name only** — it cannot write state.
  That is why the `retry` and `regen_prep` nodes exist: an edge has no body to put
  a counter in.
- A conditional function returning a **list** is LangGraph's fan-out mechanism.
- Return `[END]`, never `[]` — an empty list leaves the graph unable to finish.
- Every loop needs an explicit counter and cap. `recursion_limit` will eventually
  stop a runaway graph, but by raising `GraphRecursionError` — an accident, not a
  designed outcome.

---

## 5. LaTeX lessons

### `out` is a reserved TikZ key

The architecture diagram rendered as **nothing** — caption present, picture absent.
Cause: a style named `out`, which collides with TikZ's outgoing-angle key for curved
`to` paths. TikZ read `[out]` as the built-in key with no value and aborted the
entire picture. `loop` is reserved too. Both renamed.

Two earlier guesses (float placement, then figure size) were wrong. The fix came
from installing a compiler and reading the actual error.

### Large floats are silently dropped

`\begin{figure}[p]` with an oversized float means LaTeX cannot place it and drops it
without an error the author will notice. The same happened to the 14-row roster
table with `[h]` — it deferred to the last page, leaving Section 4 as a bare
heading. Both are now non-floating (`\captionof` via `capt-of`).

### Tooling

No LaTeX was installed. **Tectonic 0.17** was downloaded into the scratchpad, which
made it possible to compile, read errors, and verify renders instead of guessing.

---

## 6. Current project state

### Working

- The notebook runs top to bottom with zero failures.
- The architecture PDF compiles with zero errors and zero overfull boxes.
- Trained checkpoints exist for all five model variants in
  `experiments/*/checkpoints/best.pt`, including the transformer.

### Open issues

| Issue | Detail |
|---|---|
| **`src/` and `configs/` are gone** | Deleted from the working tree and the deletion is now committed. The class list and frame count were recovered from git history, but the training code is not present on disk. `git checkout <earlier-commit> -- src configs` would restore it. |
| **511 MB file blocks `git push`** | `CricShoot10kModels/Efficientnetv2-s_GRU_128_...keras` is tracked. GitHub rejects files over 100 MB. `.gitignore` covers `*.pt` but not `*.keras`. Fix: `git rm --cached`, add `CricShoot10kModels/` and `*.keras` to `.gitignore`, amend, push. |
| **Cricsheet has no shot labels** | The biggest one. See below. |

### The data problem

The premise is bridging *what a shot looks like* to *what its outcome is*.
Cricsheet ball-by-ball data records runs, wickets, bowler and phase — **not shot
type**. So N7a can retrieve `Kohli + Powerplay + Pace` but not
`Kohli + Cover Drive + Powerplay + Pace`.

Three routes, costed:

| Route | Approach | Added time | Total to finish |
|---|---|---|---|
| (a) | Reframe — the shot label becomes context, not a retrieval key | ~1 day | ≈ 6 weeks |
| (c) | Reframe + hand-label a few hundred deliveries as an evaluation set | ~1 week | ≈ 7–8 weeks |
| (b) | Full labelled join via video/record alignment | 3–6 weeks | ≈ 3–4 months |

**Recommendation: (c).** Decide before building the real data layer, since it
determines what N7a's index looks like.

---

## 7. The conference paper

**Constraint:** one week, 6 pages, and the CricShot10k classification results are
reserved for a different submission.

**Consequence:** the paper rests entirely on the agentic layer, evaluated in
simulation. That is a legitimate framework/short paper, and simulation is the
*correct* method here — with a live LLM there is no ground truth about what was
fabricated, so containment cannot be measured; in simulation the fabrication is
injected and therefore known exactly.

### Claim

> If two auditable engines are the only components allowed to produce numbers, and
> every agent must cite them, fabricated figures cannot reach the user.

### Three results plus a walkthrough

1. **Containment** — inject fabrications of controlled size (0.1% → 600%), count how
   many reach the report, with the gate on versus off. Detection is independent of
   error size because verification is exact-match, not plausibility judgement.
2. **Routing efficiency** — precision/recall against ~60 hand-labelled queries, plus
   the reduction in agent invocations versus waking all twelve.
3. **Termination** — bounded loops always stop, degrading to an explicit status.
4. **Walkthrough** — one query traced stage by stage, with two agents producing
   different advice from identical evidence and zero disagreement about facts.

A fourth result comparing Version 1 (each worker computes its own statistics)
against Version 2 was designed and then **removed at the user's request**.

### Honesty requirements

Two paragraphs must appear, or the paper is misleading rather than merely limited:

- **Error model** — fabrication is modelled as corruption of one cited value by
  relative magnitude ε; this simplifies real LLM error.
- **Implementation status** — all node bodies are deterministic; no LLM is invoked;
  the corpus is synthetic; worked-example figures are not real match data.

The simulated statistics must never be presented as facts about real cricket.

### Novelty, honestly assessed

| Element | Novelty |
|---|---|
| Multi-agent LLM system with a supervisor | Low |
| Self-correction loops | Low |
| Centralising numeric authority + citation validation | Moderate |
| Twelve stakeholder agents over shared evidence | Moderate as application |
| Simulation testbed for hallucination containment | Moderate–good |

Realistic target: workshop, short paper, or applied/WIP track. Its strength is being
complete, executable and measurable — most architecture papers are none of those.

---

## 8. Tooling decisions

**Rejected for the paper's experiments**, with reasons:

| Tool | Why not |
|---|---|
| Langfuse, LangWatch, Maxim AI, Future AGI | Observability/evaluation platforms. They observe a running LLM app; the system makes zero LLM calls, so there is nothing to observe that is not already known. |
| GEM (*A Gym for Agentic LLMs*, ICLR 2026) | An RL **training** environment suite. Needs a policy, rewards and RL infrastructure. Nothing here is being trained. |
| SimPy | Discrete-event simulation — LangGraph already provides an execution model. |
| Mesa | Spatial/social agent-based modelling — wrong domain. |

**The stub graph is the simulator.** The only thing missing is an experiment harness
around it: a sweep driver, pandas, matplotlib and a fixed seed.

GEM is worth **citing in future work**: wrapping CricketMind as a GEM environment and
learning the supervisor router as an RL policy is a natural follow-on.

Reproducibility argument: a reviewer can rerun a seeded notebook; they cannot rerun a
hosted SaaS dashboard.

---

## 9. Working preferences established

- **One definition per cell** in notebooks — each function or class gets its own
  cell. Import blocks and constant groups may stay together.
- **Node names follow function names** — `n5_classifier_head`, not `N5`. The
  rendered diagram then labels boxes with names findable by searching the notebook.
  Architecture numbers survive as the function-name prefix.
- **Claude writes the files directly**, rather than handing over snippets to paste.
  VS Code caches `.ipynb` in memory, so after any notebook edit: *Developer: Reload
  Window*, answer **Don't Save**, then re-run.
- **Code must run on both machines** — Intel Arc B580 (XPU, bf16, no GradScaler) and
  NVIDIA RTX 3070 (CUDA, fp16 + GradScaler). Never hardcode `cuda`.

---

## 10. Next actions

1. **Decide the data route** (a/c/b) — blocks the real retrieval layer.
2. **Fix the git push** — untrack the 511 MB `.keras`, extend `.gitignore`, amend.
3. **Restore `src/` and `configs/`** from git history if the training code is needed.
4. **Day 1 of the 5-day plan** — build the experiment harness: corruption-size knob,
   gate-disable switch, trial runner, sweep driver.

The harness is the only thing standing between the current state and measured
results.
