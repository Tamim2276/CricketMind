# Five-Day Plan — Methodology and Results

Target: a complete **Methodology** section and a complete **Results** section for a
6-page conference paper, evaluated in simulation.

Assumes roughly six focused hours a day. Each day ends with something finished, not
something started.

> **Revision note.** Days 1-3 have been rebuilt to produce the R1/R2/R3 results in
> `CricketMind_Results_Plan.md` rather than the earlier containment / routing /
> termination set. The termination sweep is deleted (it is now an argument, not an
> experiment) and routing is demoted, which frees roughly half a day — spend it on
> R2, which is the result that most raises the paper's ceiling.

**What already exists:** the 119-cell stub notebook with all 30 nodes running, the
compiled architecture document (`paper/CricketMind_Architecture_v2.tex`, 8 pages,
0 errors), and the results plan.

**What does not exist yet:** the experiment harness, any measured numbers, the
labelled query set, and every figure except the architecture diagram.

---

## The rule for the whole week

**Do not write a sentence of the results section until the numbers exist in a CSV.**

Writing around numbers you expect to get is how people end up quietly adjusting
experiments to match prose they already wrote. Numbers first, prose second. Days 1
to 3 produce numbers. Days 4 and 5 write.

---

## Day 1 — Build the experiment harness

Nothing is measured today. Today builds the thing that measures. There are four
knobs to add, and they are what the entire results section is made of.

### 1.1 Make the corruption size controllable

Right now a fabricating agent always claims `999.0`. That only tests the easy case.
Replace the fixed value with a relative error:

```python
CORRUPTION_EPS = 0.0      # relative size of the injected error, e.g. 0.01 = 1%

# inside the agent factory, where the fact is currently overwritten with 999.0:
key        = config["cites"][0]
true_value = facts[key]
facts = {**facts, key: round(true_value * (1 + CORRUPTION_EPS), 6)}
```

Keep six decimal places, not four — at eps = 0.001 on a value like 0.88 you need the
precision for the corruption to survive rounding at all.

- [ ] `CORRUPTION_EPS` knob added
- [ ] Sanity check at eps = 0.001: the agent cites 142.142 instead of 142.0, and the
      Faith Check still rejects it

### 1.2 Three gate modes, not two

This is the single most important change on Day 1, because it is what turns R1 from
a description into a comparison. You need the control **and** an alternative
verification strategy to compare against.

```python
GATE_MODE      = "citation"   # "none" | "plausibility" | "citation"
PLAUSIBILITY_K = 0.10         # reject if relative deviation exceeds this

def gate_accepts(cited, engine, mode=None, k=None):
    """True if this cited value is allowed through to the report."""
    mode = mode or GATE_MODE
    k    = PLAUSIBILITY_K if k is None else k
    if mode == "none":
        return True
    if mode == "citation":
        return cited == engine
    if mode == "plausibility":
        if engine == 0:
            return cited == 0
        return abs(cited - engine) / abs(engine) <= k
    raise ValueError(mode)
```

Route the Faith Check through `gate_accepts` instead of comparing dicts inline, and
thread `mode` / `k` down from the graph builder.

- [ ] `gate_accepts` written and used by `n_faith_check`
- [ ] Sanity check with `mode="none"` and a fabricating agent: the false number
      appears in the final report
- [ ] Sanity check with `mode="plausibility", k=0.10` at eps = 0.01: the fabrication
      **passes** — this is the result, not a bug

That last checkbox is the one to celebrate. If the plausibility gate never misses,
your k is wrong or the corruption is not being applied.

### 1.3 Three fabrication classes

R3 needs an agent that fabricates in ways the gate cannot catch. Add a class knob to
the agent factory:

```python
FABRICATION_CLASS = "value"   # "value" | "misattribution" | "uncited"
```

| Class | What the agent does | Expected |
|---|---|---|
| `value` | cites a corrupted number under the right key | caught |
| `misattribution` | cites `best_sr = 142.0` correctly, but prose attributes it to the Pull | **passes** |
| `uncited` | writes a number into the prose that appears in no `cites` entry | **passes** |

`misattribution` and `uncited` need one hand-written template each. They are
demonstrations, not sweeps — you need one trace of each, not a hundred runs.

- [ ] Three classes implemented
- [ ] One trace saved per class

### 1.4 The precision channel, for the false-rejection axis

R1 claims no threshold has a viable operating point. To show the other half of that
you need truthful claims that a tight threshold would reject. Model display rounding
in the citation channel:

```python
CITE_PRECISION = None   # None = full engine precision (the contract);
                        # an int = agent rounds its citation to that many dp
```

With `CITE_PRECISION = 1` and an engine value of `142.037`, a *truthful* agent cites
`142.0`. A plausibility gate at low k rejects it; so does exact match. That is the
brittleness the paper should state openly, and it is why the citation contract
requires full precision.

- [ ] `CITE_PRECISION` knob added
- [ ] Sanity check: with `CITE_PRECISION = 1` and no fabrication, the citation gate
      **falsely rejects** a truthful agent

### 1.5 Write the trial runner

One function, one run, one row of results:

```python
def run_trial(*, eps, fab_rate, fab_class, max_regens, gate_mode, k,
              cite_precision, query, seed) -> dict:
    """Run once and return the metrics for that run."""
```

Return these fields. The two in bold are the headline metrics for R1 and R2 — do not
omit them and reconstruct later.

| Field | Meaning |
|---|---|
| `n_injected` | how many agents were told to fabricate |
| `n_detected` | how many the gate caught |
| `n_corrected` | how many were fixed by regeneration |
| `n_withheld` | how many were dropped after the budget ran out |
| **`n_false_in_report`** | false claims that reached the user — R1 |
| `n_false_rejected` | truthful findings the gate rejected — R1's second axis |
| `n_findings_intended` | agents woken, i.e. findings the user should have got |
| **`n_findings_surviving`** | findings actually in the report — R2 |
| `agent_invocations` | initial calls + regenerations — R2's cost axis |
| `regen_count`, `retry_count` | loop counters |
| `agents_woken` | of twelve |
| `terminated`, `final_status` | sanity, not a result |

Echo every input parameter into the row as well, including the seed. A CSV you cannot
re-run from is a CSV you will re-generate at 2am on Day 5.

- [ ] `run_trial` written and returns all fields
- [ ] Runs in well under a second

### 1.6 Write the sweep driver

```python
def sweep(conditions, n_trials, out_csv) -> pd.DataFrame:
    """Run every condition n_trials times, save raw rows to CSV, return a DataFrame."""
```

**Save the raw rows, not just the summary.** A reviewer asking "what was the
variance?" is answerable from a CSV and unanswerable from a mean.

- [ ] Sweep driver written
- [ ] Global seed fixed and recorded in every row
- [ ] Output goes to `experiments/sim/`

### End of Day 1

- [ ] `experiments/sim/runner.py` (or a notebook section) runs a 10-trial smoke sweep
      and writes a CSV
- [ ] The CSV opens in pandas and has the columns above
- [ ] All four sanity checks above pass, including the two that are supposed to fail

---

## Day 2 — Run R1 and R2, and make both figures

Both of the paper's measured figures come out of today.

### 2.1 R1 — the three-gate comparison (morning)

| Factor | Levels |
|---|---|
| eps | 0.001, 0.01, 0.05, 0.20, 6.0 |
| gate mode | `none`, `plausibility`, `citation` |
| k (plausibility only) | 0.20, 0.05, 0.01, 0.0005 |
| liars | 1, 2, 3, 4 |

Deterministic cells need few trials; only the random choice of *which* agents lie
varies. Twenty trials per cell is ample and the whole sweep is minutes.

- [ ] Sweep complete, raw CSV saved
- [ ] Detection-rate table built: gate mode x eps
- [ ] Threshold table built: k vs (catches 0.1% error, rejects truthful claims)
- [ ] **Confirm the plausibility gate misses at small eps.** If it does not, the
      comparison collapses — investigate before writing anything.

### 2.2 R1's false-rejection half (late morning)

Re-run with no fabrication at all and `CITE_PRECISION = 1`, sweeping k.

- [ ] False-rejection rate per k recorded, for both plausibility and citation gates
- [ ] The full-precision contract row (`CITE_PRECISION = None`) recorded as the
      zero-false-rejection baseline

### 2.3 R2 — the cost of containment (early afternoon)

The result your competitors will not have.

| Factor | Levels |
|---|---|
| fabrication rate | 0, 0.25, 0.5, 0.75, 1.0 |
| `MAX_REGENS` | 0, 1, 2, 3 |
| agent behaviour | corrects on retry / never corrects |
| gate mode | `citation` |

- [ ] Sweep complete, CSV saved
- [ ] Utility computed: `n_findings_surviving / n_findings_intended`
- [ ] Invocation overhead computed against the fabrication-rate-0 baseline
- [ ] Confirm false claims stay at 0 across the whole sweep — this is now a
      **baseline row**, not the headline

### 2.4 The two figures (late afternoon)

**Figure 3** — detection rate against eps, one curve per gate. The plausibility curve
should show a visible cliff at k. Inset or second panel: false-rejection rate vs k.

**Figure 4** — fabrication rate on x; two lines: false claims reaching the user (flat
at zero) and findings surviving (decaying). The contrast between the flat line and
the falling one *is* the argument, so make both clearly labelled.

- [ ] `figures/detection.pdf` and `figures/cost.pdf` saved as **vector**, not PNG
- [ ] Both readable at column width (roughly 8.5 cm) and in greyscale

### End of Day 2

- [ ] Three CSVs of raw results
- [ ] Both measured figures finished
- [ ] The numbers for R1 and R2 written down somewhere you can copy from

---

## Day 3 — R3, the routing note, and the walkthrough

Lighter than the old Day 3, because the termination sweep is gone and routing no
longer needs sixty labelled queries.

### 3.1 R3 — the boundary (morning)

No sweep. Three traces and a table.

- [ ] `value` fabrication: caught, regenerated, verified — save the trace
- [ ] `misattribution`: *"Kohli's pull shot is his strongest, at SR 142.0"* citing
      `best_sr = 142.0` — **passes the gate**, save the trace
- [ ] `uncited`: a number in the prose absent from `cites` — **passes**, save the trace
- [ ] Fabrication-class table filled in

Put the caught trace and the misattribution trace side by side. That contrast is R3.

### 3.2 Routing efficiency note (early afternoon)

Thirty queries is enough now that you are reporting invocation reduction rather than
precision and recall.

```
query,expected_agents
"How should we bowl to Kohli in the powerplay?",n11b_tactical_analysis
"Any injury risk from his workload this week?",n11k_injury_management
"What is his IPL auction valuation?","n11l_player_representation,n11a_club_operations"
```

- [ ] `experiments/sim/queries.csv` with ~30 rows
- [ ] Labels written **before** looking at what the router does
- [ ] Mean agents woken and reduction vs twelve computed
- [ ] Per-query results saved so mismatches can be inspected

### 3.3 The walkthrough (late afternoon)

One instrumented run of the running example, captured stage by stage.

- [ ] Stage-by-stage state table (Section 8 of the results plan)
- [ ] The two agent responses side by side, same evidence
- [ ] All saved as text you can paste, not screenshots

### End of Day 3

- [ ] Every number the results section needs now exists
- [ ] Nothing left to run

---

## Day 4 — Write the Methodology section

Target **1.5 pages** plus Figure 1.

### 4.1 Deal with the architecture figure first (morning)

The existing diagram is full-page portrait. In a two-column 6-page paper it will not
fit at column width and will be unreadable if shrunk.

| Option | Cost | Result |
|---|---|---|
| Full-width figure (`figure*`) across both columns | 30 min | Readable, costs about half a page |
| Simplified diagram — twelve agents collapsed into one stacked box | 1-2 hours | Fits one column, loses detail |
| Rotate 90 degrees | 15 min | Fits, but reviewers dislike rotated figures |

**Recommended: the simplified version as `figure*`.** Twelve individually labelled
boxes is detail the roster table already carries; the diagram needs to show the
*tiers* and the two loops.

- [ ] Figure 1 fits the template and is legible at print size

### 4.2 Write the methodology (afternoon)

Transplant and compress from `paper/CricketMind_Architecture_v2.tex`:

- [ ] **Perception and grounding** — N1-N6 in a paragraph. Do not dwell; the
      classification results belong to your other paper.
- [ ] **Retrieval and the analytical services** — N7a/N7b, then N8/N9 and *why they
      are not language-model nodes*. This is the core idea; give it the most space.
- [ ] **Routing and the stakeholder agents** — N10, the roster, and the citation
      contract, **including the full-precision requirement** from Day 1.4
- [ ] **Verification** — the Faith Check, the two bounded loops, and the termination
      guarantee stated as a guarantee (see the results plan, Section 7)

Then the walkthrough subsection from Day 3.

### End of Day 4

- [ ] Methodology drafted in the paper template
- [ ] Figure 1 placed and legible
- [ ] Walkthrough subsection written

---

## Day 5 — Write the Results section

Target **1.5 pages**, two tables, two measured figures.

### 5.1 Setup paragraph (morning)

- [ ] The graph: 30 nodes, all bodies deterministic
- [ ] The synthetic corpus: how generated, why these aggregates
- [ ] The error model paragraph — **including all three fabrication classes**
- [ ] The implementation-status paragraph
- [ ] The determinism note (results plan, Section 9) — do not fake variance
- [ ] Seed, trial counts, hardware, runtime

### 5.2 The three results (late morning)

- [ ] **R1** — three-gate table, threshold table, Figure 3, and the sentence about
      no threshold having a viable operating point
- [ ] **R2** — cost table, Figure 4, and the exchange-rate sentence
- [ ] **R3** — fabrication-class table and the misattribution trace
- [ ] **Note** — routing invocation reduction, then termination as a guarantee

Every number you type must be traceable to a CSV row. Check them off as you copy.

### 5.3 Threats to validity (afternoon)

- [ ] Synthetic corpus, not real match data
- [ ] Simplified error model, stated
- [ ] No language model invoked; template agents isolate architecture from
      generation quality
- [ ] **The plausibility gate is our own implementation, not a published system**
- [ ] Routing labels are the author's own judgement
- [ ] Single domain, one query style

### 5.4 Read it back (end of day)

- [ ] Every claim is supported by a table or figure in the paper
- [ ] No number appears that is not in a CSV
- [ ] No "mean +/- sd" where sd is zero by construction
- [ ] The word "shows" is not doing work that "suggests" should do
- [ ] Nothing implies the statistics describe real cricket

### End of Day 5

- [ ] Methodology and Results complete

---

## If you fall behind

Cut in this order — first item goes first:

1. **The `MAX_REGENS` dimension of R2.** Fix it at 2 and report the fabrication-rate
   curve only. You lose the utility/invocation tradeoff table but keep the result.
2. **R3's `uncited` class.** The misattribution example alone carries the argument.
3. **The routing note.** It is two sentences; losing it costs little.
4. **The simplified architecture figure.** Fall back to the full-width original.

**Never cut:**

- **The `none` and `plausibility` gate modes.** Without them R1 proves nothing and
  you are back to asserting `142.1 != 142.0`.
- **R2's surviving-findings column.** It is the paper's credibility.
- **The error-model and implementation-status paragraphs.** Without them the paper is
  misleading rather than limited.

If the week collapses entirely, **R1 and R2 alone are a complete, defensible paper.**
Build them in that order.

---

## What is not in this plan

Introduction, related work, abstract and conclusion. Budget one more day, or write
the introduction in the gaps while sweeps are running — they take minutes of
wall-clock but you will be sitting there anyway.
