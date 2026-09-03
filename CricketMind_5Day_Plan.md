# Five-Day Plan — Methodology and Results

Target: a complete **Methodology** section and a complete **Results** section for a
6-page conference paper, evaluated in simulation.

Assumes roughly six focused hours a day. Each day ends with something finished, not
something started.

**What already exists:** the 116-cell stub notebook with all 30 nodes running, the
compiled architecture document (`paper/CricketMind_Architecture_v2.tex`, 8 pages,
0 errors), and the results plan (`CricketMind_Results_Plan.md`).

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

Nothing is measured today. Today builds the thing that measures.

### 1.1 Make the corruption size controllable

Right now a fabricating agent always claims `999.0`. That only tests the easy case.
Replace the fixed value with a relative error:

```python
CORRUPTION_EPS = 0.0      # relative size of the injected error, e.g. 0.01 = 1%

# inside the agent factory, where the fact is currently overwritten with 999.0:
true_value = facts[config["cites"][0]]
facts = {**facts, config["cites"][0]: round(true_value * (1 + CORRUPTION_EPS), 4)}
```

- [ ] `CORRUPTION_EPS` knob added
- [ ] Sanity check: at ε = 0.001, the agent cites 142.1 instead of 142.0, and the
      Faith Check still rejects it

### 1.2 Add the control condition

You need to run the same graph **with verification switched off**. Add a flag to the
graph builder:

```python
def build_cricketmind_graph(gate_enabled: bool = True):
    ...
    if gate_enabled:
        g.add_conditional_edges("faith_check", faith_gate, {...})
    else:
        g.add_edge("faith_check", "n12_supervisor_synthesis")   # never rejects
```

- [ ] `gate_enabled` parameter added
- [ ] Sanity check: with `gate_enabled=False` and a fabricating agent, the false
      number appears in the final report

Without this, you have described a mechanism rather than shown it does anything.

### 1.3 Write the trial runner

One function, one run, one row of results:

```python
def run_trial(*, eps, n_liars, max_regens, gate_enabled, query, seed) -> dict:
    """Run once and return the metrics for that run."""
```

Return these fields:

| Field | Meaning |
|---|---|
| `n_injected` | how many agents were told to lie |
| `n_detected` | how many the gate caught |
| `n_corrected` | how many were fixed by regeneration |
| `n_withheld` | how many were dropped after the budget ran out |
| `n_false_in_report` | **false claims that reached the user — the headline metric** |
| `regen_count`, `retry_count` | loop counters |
| `agents_woken` | of twelve |
| `terminated` | did the run finish |

- [ ] `run_trial` written and returns all fields
- [ ] Runs in well under a second

### 1.4 Write the sweep driver

```python
def sweep(conditions, n_trials, out_csv) -> pd.DataFrame:
    """Run every condition n_trials times, save raw rows to CSV, return a DataFrame."""
```

**Save the raw rows, not just the summary.** A reviewer asking "what was the
variance?" is answerable from a CSV and unanswerable from a mean.

- [ ] Sweep driver written
- [ ] Global seed fixed and recorded in the CSV
- [ ] Output goes to `experiments/sim/`

### End of Day 1

- [ ] `experiments/sim/runner.py` or a notebook section that runs a 10-trial smoke
      sweep and writes a CSV
- [ ] The CSV opens in pandas and has the columns above

---

## Day 2 — Run Result 1 and Result 3, and make the figure

### 2.1 Result 1 — containment sweep (morning)

| Factor | Levels |
|---|---|
| ε | 0.001, 0.01, 0.05, 0.20, 6.0 |
| liars | 1, 2, 3, 4 |
| `MAX_REGENS` | 0, 1, 2, 3 |
| gate | on, off |

Even at 50 trials per cell that is a few thousand runs of a sub-second graph —
minutes, not hours.

- [ ] Sweep complete, raw CSV saved
- [ ] Summary table built: false claims reaching the report, by ε, gate on vs off
- [ ] **Check the expected result actually holds**: zero escapes with the gate on at
      every ε. If it does not, that is a finding — investigate before writing.

### 2.2 Result 3 — termination stress (early afternoon)

Three adversarial conditions, 500 runs each:

- an agent that never corrects itself
- a classifier that is never confident
- both at once

- [ ] Sweep complete, CSV saved
- [ ] Termination rate and maximum superstep count recorded
- [ ] Confirm final statuses are `unverified_claims_dropped` and `low_confidence`

### 2.3 Figure 3 — the containment chart (late afternoon)

A simple grouped bar chart: error size on the x-axis, false claims reaching the
report on the y-axis, two bars per group (gate off, gate on).

Keep it plain. Two colours, axis labels with units, no chart junk. It will be
printed in greyscale by at least one reviewer, so make the two bars distinguishable
without colour.

- [ ] `figures/containment.pdf` saved as **vector**, not PNG
- [ ] Readable at column width (roughly 8.5 cm)

### End of Day 2

- [ ] Two CSVs of raw results
- [ ] One finished figure
- [ ] The numbers for Results 1 and 3 written down somewhere you can copy from

---

## Day 3 — Result 2 and the walkthrough

### 3.1 The labelled query set (morning)

Write about **60 queries**, each with the agents you think should wake. This is the
only hand-labelling in the paper. Spread them across stakeholders — five per agent
gives 60.

```
query,expected_agents
"How should we bowl to Kohli in the powerplay?",n11b_tactical_analysis
"Any injury risk from his workload this week?",n11k_injury_management
"What is his IPL auction valuation?","n11l_player_representation,n11a_club_operations"
```

- [ ] `experiments/sim/queries.csv` with ~60 rows
- [ ] Labels written **before** looking at what the router does — otherwise you are
      grading the answer against itself

That last point matters. Label first, run second.

### 3.2 Routing evaluation (early afternoon)

- [ ] Run every query, record which agents woke
- [ ] Compute precision, recall, mean agents woken, reduction vs twelve
- [ ] Save per-query results to CSV so failures can be inspected
- [ ] Look at the five worst mismatches — they are your discussion material

### 3.3 The walkthrough (late afternoon)

One instrumented run of the running example, captured stage by stage.

- [ ] Table of what state holds after each node (the Section 6 table in the results
      plan)
- [ ] The two agent responses side by side, same evidence
- [ ] The fabrication trace showing catch → regenerate → pass
- [ ] All three saved as text you can paste, not screenshots

### End of Day 3

- [ ] Every number the results section needs now exists
- [ ] Nothing left to run

---

## Day 4 — Write the Methodology section

Now you write. Target **1.5 pages** plus Figure 1.

### 4.1 Deal with the architecture figure first (morning)

The existing diagram is full-page portrait. In a two-column 6-page paper it will not
fit at column width and will be unreadable if shrunk.

Pick one:

| Option | Cost | Result |
|---|---|---|
| Full-width figure (`figure*`) across both columns | 30 min | Readable, costs about half a page |
| Simplified diagram — collapse the twelve agents into one stacked box labelled "12 stakeholder agents" | 1–2 hours | Fits one column, loses detail |
| Rotate 90° | 15 min | Fits, but reviewers dislike rotated figures |

**Recommended: the simplified version as `figure*`.** Twelve individually labelled
boxes is detail the roster table already carries; the diagram only needs to show the
*tiers* and the two loops.

- [ ] Figure 1 fits the template and is legible at print size

### 4.2 Write the methodology (afternoon)

Transplant and compress from `paper/CricketMind_Architecture_v2.tex`. Four
subsections:

- [ ] **Perception and grounding** — N1–N6 in a paragraph. Do not dwell; the
      classification results belong to your other paper.
- [ ] **Retrieval and the analytical services** — N7a/N7b, then N8/N9 and *why they
      are not language-model nodes*. This is the paper's core idea; give it the most
      space.
- [ ] **Routing and the stakeholder agents** — N10, the roster, and the citation
      contract every agent must satisfy.
- [ ] **Verification** — the Faith Check, the two bounded loops, and what happens
      when a budget is exhausted.

Then the walkthrough subsection from Day 3.

### End of Day 4

- [ ] Methodology drafted in the paper template
- [ ] Figure 1 placed and legible
- [ ] Walkthrough subsection written

---

## Day 5 — Write the Results section

Target **1.5 pages**, two tables, one measured figure.

### 5.1 Setup paragraph (morning)

Two short paragraphs, and they must contain:

- [ ] The graph: 30 nodes, all bodies deterministic
- [ ] The synthetic corpus: how it is generated, why aggregates were chosen
- [ ] The error model paragraph (wording is in the results plan)
- [ ] The implementation-status paragraph
- [ ] Seed, number of trials, hardware, runtime

### 5.2 The three results (late morning)

- [ ] **Result 1** — containment table, Figure 3, and the sentence about detection
      being independent of error magnitude
- [ ] **Result 2** — routing metrics table and the invocation-reduction sentence
- [ ] **Result 3** — termination, one paragraph

Every number you type must be traceable to a CSV row. Check them off as you copy
them across.

### 5.3 Threats to validity (afternoon)

- [ ] Synthetic corpus, not real match data
- [ ] Simplified error model, stated
- [ ] No language model invoked; template agents isolate architecture from
      generation quality
- [ ] Routing labels are the author's own judgement
- [ ] Single domain, one query style

### 5.4 Read it back (end of day)

- [ ] Every claim in the results is supported by a table or figure in the paper
- [ ] No number appears that is not in a CSV
- [ ] The word "shows" is not doing work that "suggests" should do
- [ ] Nothing implies the statistics describe real cricket

### End of Day 5

- [ ] Methodology and Results complete

---

## If you fall behind

Cut in this order — first item goes first:

1. **Result 3 (termination).** Reduce to one sentence in the discussion. It closes a
   reviewer question but proves little on its own.
2. **The `MAX_REGENS` sweep dimension.** Fix it at 2 and report that. The ε sweep is
   the interesting axis.
3. **The query set, 60 down to 30.** Weaker statistics but still a real measurement.
4. **The simplified architecture figure.** Fall back to the full-width original.

**Never cut:**

- The **gate-off control condition**. Without it Result 1 proves nothing.
- The **error-model and implementation-status paragraphs**. Without them the paper is
  misleading rather than limited.

---

## What is not in this plan

Introduction, related work, abstract and conclusion. Budget one more day for those,
or write the introduction in gaps while sweeps are running — they take minutes of
wall-clock but you will be sitting there anyway.
