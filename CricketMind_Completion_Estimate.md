# CricketMind — Completion Estimate

**Question:** the stub graph is built. How long to finish the whole system?

**Short answer:** 21–35 focused working days (≈ 6–10 calendar weeks) — *provided* the
shot-label data question below is answered early. If it is answered the expensive
way, add two to three months.

---

## 1. Where the project actually stands

More is done than "basic structure" implies.

**Complete:**

- The full LangGraph topology — all 30 nodes of Architecture v2, both correction
  loops, dynamic dispatch, bounded retry and regenerate budgets.
- The verification mechanism. Agents declare their citations; the Faith Check
  validates them against N8/N9 by dict comparison rather than by reading prose.
  This is the thesis's central claim, and it is demonstrably working.
- **Trained checkpoints for all five model variants**, including the temporal
  transformer, in `experiments/*/checkpoints/best.pt`.
- A test harness that already injects fabricated claims and proves they are caught,
  regenerated, and — when an agent will not correct itself — withheld.

**Not done:** every node body is a deterministic stub. No video is decoded, no
checkpoint is loaded, no LLM is called, no real ball-by-ball data is indexed.

The important consequence: **the architectural risk is retired.** Every stub keeps
the signature `(state) -> dict`, so making a node real is a change inside one
function body. The graph never changes. What remains is integration and evaluation,
which is more predictable work than architecture.

---

## 2. Remaining work

| Block | Work | Days | Risk |
|---|---|---|---|
| **A** Perception real | N1 video decode + 15-frame sampling; load the transformer checkpoint into N2/N4/N5; batching and latency | 3–5 | Low — checkpoints exist |
| **B** LLM layer | N3 query embedding + structured parse; N10 as an LLM router; twelve agent prompts through the existing factory; N12 synthesis | 4–6 | Medium |
| **B′** Structured citations | Force JSON / tool-use output so LLM prose still carries machine-checkable citations | 1–2 | Medium — subtler than it looks |
| **C** Data layer | Cricsheet ball-by-ball ingest, row schema, FAISS indexes | 2–3 | **See section 3** |
| **D** Evaluation | Routing accuracy against a labelled query set; hallucination-catch rate; end-to-end qualitative review | 4–7 | Medium |
| **E** Writing | Paper, figures, related work | 7–12 | Low |

**Total: 21–35 focused working days.** At a realistic student pace alongside other
commitments, **6–10 calendar weeks**.

### Notes on individual blocks

**Block A is the easy one.** The models are trained and the device helpers already
resolve XPU → CUDA → CPU, so the same code runs on both the Arc B580 and the RTX
3070 without modification.

**Block B′ deserves its own line** because it is the piece most likely to be
underestimated. The stub agents emit citations because a template makes them; an
LLM writing free prose will not, unless output is constrained by a JSON schema or a
tool definition. If citations degrade into "whatever number appeared in the text",
the Faith Check silently stops working and the thesis's main claim goes with it.
Budget the time and test it deliberately.

**Block D is cheaper than usual** because the Step 9 harness already injects known
falsehoods and measures whether they are caught. Extending that into a reportable
hallucination-catch rate is a small step. The routing-accuracy set (roughly a
hundred queries labelled with the agents that *should* wake) is the larger half.

---

## 3. The one thing that could double the timeline

### Cricsheet does not record shot type

Ball-by-ball data gives runs, wickets, bowler, over and phase. It does **not** say
"that delivery was a cover drive."

The premise of CricketMind is bridging *what a shot looks like* (kinematics) to
*what its outcome is* (statistics). That bridge needs a source where both facts
exist on the same delivery. As of now the planning documents do not name one.

Concretely: N7a currently retrieves `Kohli + Cover Drive + Powerplay + Pace`.
Against real Cricsheet data it can only retrieve `Kohli + Powerplay + Pace`. The
shot dimension — the part the video contributes — has nowhere to land.

This is not a bug in the graph. It is a data-availability question that determines
the shape of N7a's index, and it should be decided **before** Block C rather than
discovered during it.

### Three ways out

**(a) Reframe the join — about 1 day.**
The video's shot label becomes context the agents reason *with*, not a retrieval
key. N8 reports strike rate against pace in the powerplay; the Personal Performance
agent connects that figure to the technique actually observed in the clip. The
multi-agent contribution, the routing, and the faithfulness gate all survive
untouched. What weakens is the claim to shot-level outcome statistics.

**(b) Build the labelled join — 3 to 6 weeks.**
Run the trained classifier over broadcast footage time-aligned to ball-by-ball
records, producing shot-labelled outcomes. This is the strong version of the thesis
and a contribution in its own right. It requires aligned video, and the alignment
itself is fiddly, error-prone work that is easy to underestimate.

**(c) Hand-label an evaluation set — about 1 week.**
Build everything on the reframed join from (a), then hand-label a few hundred
deliveries purely as an evaluation set, enough to demonstrate the mechanism works
without claiming coverage.

| Route | Added time | Total |
|---|---|---|
| (a) Reframe | ~1 day | ≈ 6 weeks |
| (c) Reframe + labelled eval set | ~1 week | ≈ 7–8 weeks |
| (b) Full labelled join | 3–6 weeks | ≈ 3–4 months |

---

## 4. Recommendation

**Take route (c).**

Build the whole system on the reframed join so it runs end to end on real data
early, then hand-label a few hundred deliveries as an evaluation set. That gives
both a working system and a defensible measurement, without betting the schedule on
video alignment.

Route (b) is the better thesis if the time exists. It is not the better thesis if it
consumes the time that Blocks D and E need — an unevaluated, unwritten system is
worth less than an evaluated, written one with a narrower claim.

---

## 5. Suggested ordering

The blocks are not equally reorderable. This sequence keeps the system runnable at
every point and front-loads the decisions that constrain later work.

1. **Decide the data route (a/b/c).** Everything in Block C depends on it.
2. **Block A — perception real.** Fastest visible progress, lowest risk, and it
   makes the demo genuine rather than simulated.
3. **Block C — data layer.** Real rows behind N7a/N7b. N8 and N9 need no changes;
   they already compute from whatever rows they are given.
4. **Block B + B′ — the LLM layer.** Do the citation plumbing *first*, then the
   twelve agents, so the faithfulness gate is never bypassed even temporarily.
5. **Block D — evaluation.**
6. **Block E — writing**, started in parallel with D rather than after it.

---

## 6. Assumptions behind these numbers

- "Focused working day" means roughly six uninterrupted hours on this project.
- The trained transformer checkpoint loads and reproduces its reported accuracy.
  If it needs retraining, add 2–4 days per run depending on hardware.
- LLM access is available and not rate-limited to the point of slowing iteration.
- The Kaggle historical dataset for N7b is already identified and downloadable.
- No change to the twelve-agent roster or the graph topology. Adding or splitting
  agents is cheap now (the factory absorbs it); changing the *topology* is not.
