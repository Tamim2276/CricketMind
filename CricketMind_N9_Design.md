# N9 — What-If Simulator: Design for the Full System

**Status:** design only. Not to be built for the one-week simulation paper.
**Target:** the complete system, roughly two months out.

**Why this document exists.** The current N9 answers one narrow question. The
architecture calls N9 *"the only source of forward-looking, non-observational
claims"*, and questions like *"what if Rohit had batted at #2 instead of Kohli?"*
are exactly what that phrase promises. This is the plan for making the promise true.

---

## 1. Where N9 is today, and where it must get to

### What it does now

```python
p_ball = outs / balls              # for the batter's weakest shot
for 20,000 trials:
    for 6 deliveries:
        if random() < p_ball: -> dismissed, stop
```

One batter, one shot, six balls, one probability out. The analytic cross-check in the
code (`1 - (1-p)**6`) confirms the model: it is a coin flipped six times.

This is **correct and useful**, but it is one member of a much larger family.

### The family of questions N9 should own

| Tier | Question type | Example |
|---|---|---|
| 1 | **Delivery** | "Six short balls at Kohli — chance of a wicket?" |
| 2 | **Matchup** | "What if Bumrah bowled this over instead of Shami?" |
| 2 | **Phase** | "What if we used this plan in the death overs instead?" |
| 3 | **Order** | "What if Rohit batted at #2 instead of Kohli?" |
| 3 | **Chase** | "What if we needed 60 off 30 instead of 40 off 30?" |

Tier 1 exists. Tier 3 is what you asked for. They are separated because **Tier 3 costs
roughly four times as much to build**, and Tier 2 is a genuine middle option if the
schedule tightens.

Every one of these has the same shape:

> Change one thing in the setup. Re-simulate. **Report the difference.**

That shape is the design.

---

## 2. The single most important design rule

**N9 must report the difference against a baseline, never a bare counterfactual
number.**

"With Rohit at #2, the mean score was 187" is meaningless on its own. The claim is:

```
baseline       (Kohli #2):  178.4 runs
counterfactual (Rohit #2):  187.1 runs
difference               :   +8.7 runs   95% CI [+1.2, +16.3]
```

And if the interval contains zero, the honest output is **"no detectable
difference"** — N9 must be capable of saying that, and must say it.

### Why this matters for your thesis specifically

Your Faith Check verifies that an agent **quoted the engine correctly**. It does not
verify that the quote is **meaningful**.

If N9 returns `+8.7` with a CI of `[-5.0, +22.4]`, an agent can cite `8.7` exactly,
pass verification, and tell the user something statistically empty — with your
system's verification stamp on it.

**So the citation contract must be extended:** any point estimate produced by N9
carries its interval as part of the same fact, and an agent citing the estimate is
required to cite the interval too. Make this a rule in `sanctioned_facts`, not a
convention.

```python
"delta_runs":     8.7,
"delta_runs_ci":  [1.2, 16.3],     # citing delta_runs REQUIRES citing this
"delta_runs_sig": True,            # does the CI exclude zero
```

This is a direct strengthening of the R3 boundary in the results plan: it closes one
specific way a correct number can carry a false claim.

---

## 3. The core: a ball-by-ball innings simulator

### The state machine

```
state = (over, ball, wickets, runs, striker, non_striker, next_batter_index)
```

For each delivery:

1. Look up the outcome distribution for `(striker, phase, bowler_type)`
2. Sample an outcome from `{0, 1, 2, 3, 4, 6, wicket, wide, no-ball}`
3. Update runs and wickets
4. Rotate strike on odd runs
5. On a wicket, bring in `next_batter_index` and increment it
6. At the end of the over, swap strike and change the bowler

Stop at 20 overs or 10 wickets (or when the target is passed, for a chase).

Run the whole thing `N_INNINGS` times for the baseline, `N_INNINGS` times for the
counterfactual, and compare the two distributions.

### Reproducibility

Use a seeded generator, and **put the seed in the output**:

```python
"seed": 42, "n_innings": 10_000
```

A claim that cannot be re-run is not auditable, and auditability is the entire reason
N9 exists as code rather than as an LLM call.

### Hardware note

This is pure Python or NumPy and is CPU-bound. It touches no GPU, so it is portable
across both machines by default — no device ladder needed here. If you later vectorise
it with torch for speed, route the device through the usual XPU -> CUDA -> CPU helper.

---

## 4. The hard part: estimating the probability tables

This is where the real work is, and where naive implementations fall over.

### The straightforward version

From Cricsheet ball-by-ball records, count outcomes grouped by
`(batter, phase, bowler_type)`:

```
P(runs=0 | Rohit, powerplay, pace) = 0.42
P(runs=4 | Rohit, powerplay, pace) = 0.11
P(out    | Rohit, powerplay, pace) = 0.031
```

### Why the straightforward version breaks

**Sparsity.** A specific batter, in a specific phase, against a specific bowler type
may have only thirty deliveries on record. Estimating a seven-outcome distribution
plus a dismissal rate from thirty balls gives you noise, and it is worst exactly where
you can least afford it — the lower order, where sample sizes are tiny.

Left uncorrected, this produces a simulator that confidently reports that your number
nine is a better opener than Rohit.

### The fix: shrink toward the population

Blend each player's own rate with the phase-level average, weighted by how much data
you actually have:

```python
p_adjusted = (n * p_player + k * p_population) / (n + k)
```

`k` is a smoothing constant in balls — start around 50 to 100 and tune it. With plenty
of data the player's own rate dominates; with almost none, it falls back to the
population average.

This is standard practice in sports analytics and is straightforward to defend in
writing. **Budget real time for it** — it is not a one-line addition.

---

## 5. Calibration: the step that is always underestimated

**Do not let N9 into the graph until it has been validated.** Here is why this matters
more in your system than in most.

In CricketMind, N9's output is *sanctioned truth*. The Faith Check treats it as
ground truth by definition. So if N9 is miscalibrated, the architecture will
faithfully verify wrong numbers and deliver them to the user **with a verification
stamp on them**. Your containment guarantee protects against agents inventing numbers.
It offers no protection at all against an engine that is simply wrong.

### The validation to run

| Check | What to compare | Pass condition |
|---|---|---|
| **Score distribution** | simulated final scores vs real T20 scores | means and spread in the same range |
| **Innings replay** | simulate real innings with the real order | actual score falls inside the simulated distribution most of the time |
| **Sanity ordering** | opener vs tail-ender in the same slot | the opener must come out ahead |
| **Degenerate input** | a player with no data | falls back to population rates, does not crash |

If simulated scores come out systematically twenty runs high, the model is wrong and
every counterfactual built on it is wrong too.

**This validation belongs in the paper**, as its own short subsection. It is also a
genuine contribution — most cricket what-if tools publish point estimates with no
calibration evidence at all.

---

## 6. Changes required elsewhere in the graph

N9 cannot grow in isolation. Five things around it must change.

### 6.1 Video must become optional

This is the one that will bite you.

Your graph currently runs, on every single query:

```
N1 video -> N2 -> N4 -> N5 classifier -> N6 shot DNA -> N7a/N7b -> N8/N9
```

*"What if Rohit batted at #2?"* **has no video.** There is no clip and no shot to
classify, but `n6_shot_dna` needs a shot label and `conf_gate` assumes the classifier
ran.

Add a branch at `START`:

```python
def has_video(state):
    return "video" if state.get("video_path") else "text_only"

g.add_conditional_edges(START, has_video, {
    "video":     "n1_video_input",
    "text_only": "n6_shot_dna",      # build shot_dna from query filters alone
})
```

`n6_shot_dna` then has to tolerate a missing `shot_label` and missing posture metrics.
Plan for this now; retrofitting it later means touching the confidence gate, the retry
loop and `sanctioned_facts` at the same time.

### 6.2 N3 must parse the scenario

The text encoder currently produces retrieval filters. It now also needs to produce a
scenario description — what is being changed, from what, to what:

```python
"scenario": {
    "type":           "batting_order_swap",
    "baseline":       {"position": 2, "player": "Kohli"},
    "counterfactual": {"position": 2, "player": "Rohit"},
}
```

In the real system this is an LLM call with a constrained JSON schema. It is the same
structured-output problem as Block B-prime in the completion estimate, so solve it once
and reuse the approach.

### 6.3 N7a must retrieve more

Today it retrieves deliveries for the queried player. An innings simulation needs
ball-by-ball data for **every player in the order**, plus the bowling attack. Different
query shape, larger result set.

### 6.4 `sim_results` grows

```python
{
  "scenario_type":  "batting_order_swap",
  "baseline":       {"mean_score": 178.4, "ci95": [174.1, 182.7]},
  "counterfactual": {"mean_score": 187.1, "ci95": [182.6, 191.6]},
  "delta_runs":     8.7,
  "delta_runs_ci":  [1.2, 16.3],
  "delta_runs_sig": True,
  "n_innings":      10_000,
  "seed":           42,
}
```

### 6.5 `sanctioned_facts` must expose the intervals

Flatten the new fields in, and enforce the pairing rule from Section 2 — citing a point
estimate without its interval should fail the Faith Check.

---

## 7. The multi-modal bridge

As specified above, this counterfactual **does not use your video branch at all**,
which weakens the multi-modal story for exactly the query type you most want to show
off.

The connection to make:

> The video observes *how* a player bats — shot profile, weight transfer, which
> deliveries they punish. Those observations **adjust the probability tables** used in
> the simulation. So "Rohit at #2 in the powerplay against pace" runs on rates informed
> by observed technique, not career averages alone.

Implementation: a multiplier on the base rates, derived from the shot-quality metrics
N6 already produces.

**Two warnings.**

First, this is the least defensible component in the whole system unless you validate
it. "We multiplied the rate by a number derived from weight transfer" invites the
question *"why that number?"*, and hand-waving there will cost you more credibility
than the feature gains.

Second, **make it optional and off by default, and report both.** If the adjusted and
unadjusted simulations disagree, that difference is itself an interesting result — and
if you cannot justify the adjustment, you still have a working simulator without it.

---

## 8. What it costs

| Piece | Days |
|---|---|
| Probability tables from Cricsheet, with shrinkage | 3-4 |
| Innings simulator core | 2-3 |
| Scenario parsing in N3 (structured output) | 2 |
| **Calibration and validation** | 3-5 |
| Graph changes: optional video, state, sanctioned facts | 1-2 |
| Video-to-rate bridge (optional) | 2-3 |
| **Total** | **13-19 days (3-4 weeks)** |

### This changes your timeline

The completion estimate budgets **2-3 days** for "Block C — data layer". This work does
not fit inside that. It is a **new block**, and adding it means either extending the
schedule or taking time from Blocks B, B-prime or D.

An unevaluated system with a beautiful simulator is worth less than an evaluated system
with a simple one. Decide deliberately, not by drift.

### The fallback: build Tier 2 instead

If the schedule tightens, **matchup counterfactuals** are the middle option:

> "What if Bumrah bowled to Kohli in the powerplay instead of Shami?"

This needs only the two players' head-to-head rates over a fixed number of deliveries.
No batting order, no strike rotation, no innings loop, and far less calibration —
roughly **4-5 days** rather than fifteen.

It is a real extension of today's N9, it keeps the "what-if" name honest, and it leaves
the door open to Tier 3 later.

---

## 9. The invariant that must not break

**N9 stays code. It never becomes an LLM call.**

The entire architecture rests on there being exactly two auditable sources of numbers.
The moment N9 asks a language model to estimate an outcome, it stops being a ground
truth the Faith Check can verify against, and the central claim of the thesis
collapses.

Whatever N9 grows into, it must remain: deterministic given its seed, readable as
source code, and re-runnable by anyone with the same data.

---

## 10. Build order, when the time comes

1. **Probability tables with shrinkage**, validated on their own before anything
   consumes them
2. **Innings simulator**, checked against real scores — Section 5
3. **Graph changes** so a video-free query can reach N9 at all — Section 6.1
4. **Scenario parsing** in N3
5. **Wire it in**: `sim_results`, `sanctioned_facts`, the interval-pairing rule
6. **The video bridge**, last and optional

Steps 1 and 2 are independent of CricketMind entirely. They can be built and tested as
a standalone module long before the graph is ready for them — which makes this the
easiest part of the two-month plan to start early or hand to someone else.
