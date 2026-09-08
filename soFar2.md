# soFar2 — Session Record

Continues from `soFar1.md`. That file recorded building the 30-node LangGraph
stub. This one records the session that **rethought what the paper measures** and
**made the graph accept queries with no video**.

Two things dominate this session: a correction to the results plan that changes
what the paper argues, and one topology change to the graph.

---

## 1. What this session produced

| Artifact | Status |
|---|---|
| `CricketMind_Results_Plan.md` | **rewritten twice** — new results, then rewritten in plain English |
| `CricketMind_5Day_Plan.md` | **Days 1–3 rebuilt** to produce the new results |
| `CricketMind_N9_Design.md` | **new** — design for the real What-If Simulator |
| `notebooks/cricketmind_langgraph_demo.ipynb` | **130 cells**, video-optional entry branch, runs clean |
| `paper/CricketMind_Architecture_v2.tex` / `.pdf` | entry branch added to diagram and prose, recompiled |
| `soFar2.md` | this file |

---

## 2. The correction that changed the paper

This is the most important thing in the session.

### The old plan was measuring a tautology

The planned headline result was *"we injected fabrications; zero reached the
report."* The plan even annotated the column **"(by design)"**.

That annotation is fatal. The Faith Check compares a cited value against the
engine value. Injecting `142.1` where the engine holds `142.0` and then finding a
mismatch is **an equality check, not a discovery**. No number of trials makes it
informative.

The sentence a reviewer would write:

> *"The central result is an assertion of equality testing, presented as an
> empirical finding."*

The same flaw applied to the termination result. Loops bounded by `MAX_REGENS = 2`
stop for the same reason a `for` loop stops. Running 500 adversarial trials to
confirm it reads as uncertainty about one's own design.

**You cannot fix this by running more trials. You fix it by measuring different
things.**

### The reframe

> Containment is a **structural property, not an empirical one**. Do not evaluate
> whether it holds — it holds by construction. Evaluate the **conditions under
> which alternatives fail**, the **cost** the property imposes, and the
> **boundary** beyond which it stops protecting the user.

### The three results that replaced the old ones

**R1 — Where the obvious alternative fails.** Three gates on identical injected
errors: no gate (control), a *plausibility* gate that rejects deviations beyond a
threshold `k` (standing in for LLM-judge and self-consistency verification), and
the exact-match citation gate. The plausibility gate misses every fabrication
below `k`; lower `k` and it starts rejecting truthful values. **No threshold has a
viable operating point.** Exact matching is not on that curve at all. This is the
only result whose outcome is not fixed by construction, so it is the headline.

**R2 — What containment costs.** Containment is purchased by *withholding*: an
agent that will not correct itself has its finding dropped, so the report gets
shorter. Sweep the fabrication rate from 0 to 100% and plot two lines — false
claims reaching the user (flat at zero) and findings surviving (decaying). Nobody
has published that exchange rate. `MAX_REGENS` turns out to be a **utility** knob,
not a safety knob: containment holds even at 0.

**R3 — The boundary.** The gate verifies *values*, not the claims wrapped around
them. An agent can write *"Kohli's pull shot is his strongest, at SR 142.0"*,
cite `best_sr = 142.0` correctly, and **pass** — while 142.0 belongs to the cover
drive. Needs no experiment: two hand-written templates and two traces.

**Demoted:** routing became a two-sentence invocation-reduction note (the labels
are the author's own, so precision against them is close to circular); termination
became a stated guarantee rather than a sweep, which is strictly stronger.

### Two supporting rules added

- **Do not fake variance.** With deterministic nodes and a fixed seed, many cells
  have sd = 0 by construction. Reporting "mean ± sd over 100 runs" manufactures
  statistical texture that is not there.
- **Citations must carry full engine precision.** If the engine holds `142.037`
  and the agent cites `142.0`, exact matching rejects a truthful claim. Rounding
  is a presentation concern applied *after* verification.

---

## 3. The architecture change: video is now optional

### The problem, found by accident

A question about the What-If Simulator surfaced it. *"What if Rohit had batted at
number two instead of Kohli?"* is a legitimate query with **no clip**. But every
path in the graph ran `N1 → N2 → N4 → N5 → N6`, `n6_shot_dna` asserted a shot
label, and the confidence gate assumed the classifier had run.

**The graph assumed every query carries a video.** It would have crashed.

### Why it was fixed now rather than in month two

- Topology changes are the expensive kind. Adding an agent is absorbed by the
  factory; adding a branch is not.
- The paper's Figure 1 shows the topology, and papers are permanent. Publishing a
  diagram that is about to change means publishing the wrong architecture.
- Retrofitting later means touching the confidence gate, the retry loop and the
  sanctioned-fact surface at the same time.

### What changed

```python
g.add_conditional_edges(START, has_video, {
    "video":     "n1_video_input",
    "text_only": "n6_shot_dna",
})
g.add_edge(START, "n3_text_encoder")     # every query has text
```

| Cell | Change |
|---|---|
| `new_state` | `video_path` may be `None` |
| **`has_video`** (new) | START router returning `"video"` / `"text_only"` |
| `n6_shot_dna` | tolerates a missing classification; records `modality` |
| `n7a_stats_retrieval` | asserts a text-only query names at least one shot |
| **`agents_missing_facts`** (new) | agents whose citations this query cannot satisfy |
| `n10_supervisor_router` | removes those agents before ranking |
| `sanctioned_facts` | omits `shot` / `weight_transfer` when there is no clip |
| `n12_supervisor_synthesis` | prints "no clip supplied" instead of crashing |
| `_score_agents` | fixed a `None` comparison (below) |
| `build_cricketmind_graph` | conditional edge at `START` |
| 13A–13F (new) | six checkpoint tests |

Only `build_cricketmind_graph` was changed. The five step-graphs stay as
historical snapshots of how the system was built.

### The design decision worth remembering

On the text-only path, posture is left as **`None`, never defaulted**. Two agents
— Personal Performance (N11c) and Talent Development (N11j) — must cite
`weight_transfer`, so the router **declines to wake them**.

Filling the gap with `DEFAULT_POSTURE` would place an invented measurement into
the sanctioned facts, where the Faith Check would then certify it as ground
truth — a fabrication laundered through the mechanism built to stop fabrications.

The report is shorter instead. That is exactly the trade R2 measures.

### The bug this exposed

```python
if dna.get("confidence", 1.0) < 0.80:      # TypeError on the text path
```

**`.get(key, default)` returns the default only when the key is absent — not when
its value is `None`.** On the text-only path `confidence` exists and is `None`, so
this became `None < 0.80`. Fixed with an explicit `is not None` check.

Invisible until a second path existed. A good argument for building the branch
early rather than late.

---

## 4. Audit of the finished notebook

| Check | Result |
|---|---|
| Cells run top to bottom | **130/130, zero failures** |
| Checkpoint assertions | **20 pass** |
| Nodes vs Architecture v2 | **30 / 30 — exact match** |
| Unexpected nodes | none |
| Edges | 61 |
| Retry loop, regenerate loop | both intact |
| `n13 → END`, `degrade → END` | both intact |
| Entry branch | `START → [n1_video_input, n3_text_encoder, n6_shot_dna]` |
| Build guide Steps 0–10 | all 11 covered |
| Node naming convention | no violations |
| One definition per cell | no violations |

Constants: `NUM_FRAMES=15`, `NUM_CLASSES=15`, `CONF_THRESHOLD=0.7`,
`MAX_RETRIES=2`, `MAX_REGENS=2`, `MAX_ACTIVE_AGENTS=4`.

### Outstanding cosmetic issues

1. Cell 12 — the seed line appears **three times** (harmless, idempotent, but noise)
2. Cells 20, 36, 122 are empty
3. Seven lines carry trailing whitespace
4. **42 code cells carry saved output** — this makes every git diff enormous

---

## 5. The gap that matters

**Build guide plan: 100% done. Results plan: 0% done.**

Nothing in the notebook measures anything. Every Day-1 symbol is missing:

| Symbol | Status |
|---|---|
| `CORRUPTION_EPS` | missing |
| `GATE_MODE`, `PLAUSIBILITY_K`, `gate_accepts` | missing |
| `FABRICATION_CLASS` | missing |
| `CITE_PRECISION` | missing |
| `run_trial`, `sweep` | missing |
| `n_findings_intended`, `n_findings_surviving`, `agent_invocations` | missing |

Fabrication is still hardcoded to `999.0`, there is only one gate mode, and no run
writes a CSV row. **The notebook can demonstrate the architecture but cannot
produce a single number for R1, R2 or R3.**

---

## 6. N9 — designed, deliberately not built

`CricketMind_N9_Design.md` specifies the real What-If Simulator. Not for the
one-week paper; for the two-month system.

### Three tiers

| Tier | Question | Cost |
|---|---|---|
| 1 (exists) | "Six short balls at Kohli — chance of a wicket?" | built |
| 2 | "What if Bumrah bowled instead of Shami?" | 4–5 days |
| 3 | "What if Rohit batted at #2 instead of Kohli?" | 13–19 days |

Today's N9 is a coin flipped six times — one batter, one shot, one probability.
Tier 3 needs a whole-innings ball-by-ball simulator with batting order, strike
rotation and per-player scoring distributions.

### The four points that matter

1. **Report the difference, never the bare counterfactual.** "Mean score 187" is
   meaningless; "+8.7 runs, 95% CI [+1.2, +16.3]" is the claim. If the interval
   contains zero, N9 must say *"no detectable difference"*.
2. **The interval is part of the fact.** Citing a point estimate without its CI
   must fail verification — otherwise an agent cites `8.7` exactly, passes, and
   says something statistically empty with the system's stamp on it.
3. **Calibrate before wiring it in.** N9's output *is* ground truth to the Faith
   Check. A miscalibrated N9 means the architecture faithfully verifies wrong
   numbers. The containment guarantee protects against agents inventing figures;
   it offers **no protection against an engine that is simply wrong**.
4. **Sparsity will break naive estimation.** A tail-ender may have thirty balls on
   record. Without shrinkage toward population averages the simulator will report
   that the number nine is a better opener than Rohit.

### Good news and bad news

**Good:** this needs only ball-by-ball outcomes, which Cricsheet has. It
**completely sidesteps the shot-label problem**.

**Bad:** it does not use the video branch at all. The bridge — video-observed
shot quality adjusting the probability tables — is the strong version of the
thesis, and also the least defensible part unless validated. Make it optional,
off by default, and report both.

### Timeline impact

The completion estimate budgets 2–3 days for the whole data layer. This does not
fit inside that. It is a **new block** competing with Blocks B, B′ and D.

---

## 7. The Q1 conference question

Asked whether the completed system, roughly two months out, could go to a Q1
venue. Honest answer: **not with the paper as scoped.**

Top venues reward methodological novelty, strong results against strong
baselines, a new dataset, or a surprising finding. A complete working system with
simulation results is none of those, and a reviewer will write *"the components
are known; the contribution is integration."*

### Three routes that could reach it

- **A — dataset contribution.** Build the shot-labelled outcome join. It does not
  exist, and it is what the whole premise needs. Hardest, and can fail on data
  availability.
- **B — real LLMs plus real baselines.** Measured hallucination reduction against
  self-consistency, CoT verification, AutoGen. Feasible in two months; novelty
  stays moderate.
- **C — applied/industry track.** KDD ADS, CIKM applied, ACM MM industry. Lower
  novelty bar, higher system-completeness bar. **Most realistic target.**

### Strategy recommended

**Submit the one-week short paper now, then extend it.** The short paper is very
likely accepted somewhere reasonable, gives feedback and a citation, and most
venues permit an extended version with ~30%+ new content. **Check the workshop's
archival status first** — that determines whether extension is allowed.

**Dual-submission warning:** the CricShot10k classification results are going
elsewhere. This paper cannot rest on them, and overlap must be disclosed.

---

## 8. Tooling: LangSmith and friends cannot produce the results

Asked whether LangGraph Studio, LangGraph Visualizer, LangSmith Multi-Turn, or the
No-Code Builder could simulate the model for results. **No — all four are
development and observability tools, not experiment harnesses.**

What the results need is a sweep driver: run the graph thousands of times varying
*configuration* and write CSV rows. None of them does that.

Three reasons LangSmith specifically is wrong here:

1. It needs an LLM. Every node body is a deterministic template.
2. The system is single-turn. Multi-turn evals score conversations; there are none.
3. **It would contradict the thesis.** LangSmith's multi-turn evals score with
   LLM-as-a-judge, and R1 argues that judgement-based verification fails exactly
   where exact matching succeeds. Evaluating an anti-judgement architecture with
   an LLM judge is incoherent, and a reviewer would notice.

**Worth using for:** LangGraph Studio's checkpoint inspection to capture the
walkthrough data — then draw your own clean figure, because reviewers prefer a
table to a screenshot of somebody's IDE.

**The instinct to correct:** a named tool does not read as more rigorous than a
loop you wrote. Reviewers award credit for a control condition and a reproducible
measurement. A 40-line pandas sweep writing a seeded CSV is *more* reproducible
than a hosted service that may not exist in five years.

---

## 9. Notebook readability work

Two cells were rewritten in plainer style at request.

- **`upsert_findings`** — the dict comprehension became an explicit loop, `or []`
  became an explicit `is None` check, `merged` → `slots`, and the docstring now
  leads with the two rules (different agents accumulate; the same agent replaces)
  before explaining why. Behaviour verified unchanged by a three-step trace.
- A deliberate choice: **no `if/else` with identical branches** in the second
  loop. Spelling out "insert" and "replace" separately would be dead code that
  makes a reader stop and hunt for a difference. One assignment plus a comment.

`CricketMind_Results_Plan.md` was also rewritten end to end in plain English, with
formal academic text isolated into blocks marked **"Copy this into the paper"** so
the explanation and the submission text are never confused.

---

## 10. Current state and next actions

### Working

- 130-cell notebook, 30 nodes, both correction loops, two entry paths, runs clean
- 8-page architecture PDF, 0 errors, 0 overfull/underfull, 0 float warnings,
  diagram verified by rendering the page
- Four planning documents that agree with each other

### Immediate next step

**Build the Day-1 harness.** It is the critical path — without it there is no
results section and nothing else in the five-day plan can start.

- four knobs: `CORRUPTION_EPS`, `gate_accepts` with three modes,
  `FABRICATION_CLASS`, `CITE_PRECISION`
- `run_trial` returning all eleven metric fields
- `sweep` writing raw rows to `experiments/sim/`

Two of its sanity checks are **supposed to fail** — the plausibility gate must
miss a small fabrication, and exact matching must falsely reject a rounded
truthful value. Do not debug those; they are the result.

### Still open from soFar1

- **Cricsheet has no shot labels.** Route (c) — reframe plus a hand-labelled
  evaluation set — is still the recommendation. Undecided.
- **`git push` is still blocked** by the 511 MB `.keras` file. `.gitignore` covers
  `*.pt` but not `*.keras`.
- **`src/` and `configs/` remain deleted**, and the deletion is committed.

### New since soFar1

- Strip saved outputs from the notebook before committing (42 cells)
- Clean up the duplicated seed line, three empty cells, trailing whitespace
- `paper/CricketMind_Architecture_v2.log` is untracked build output — add to
  `.gitignore`
