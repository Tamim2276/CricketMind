# What Goes in the Results Section

A plan for the results of a **6-page conference paper** about the CricketMind
agentic framework, tested in simulation.

> **How to read this file.** The explanations are written in plain language so they
> are easy to follow. The grey quote blocks marked **"Copy this into the paper"** are
> different — those are written in formal academic English and should go into your
> submission as they are.

> `[measure]` means "a number you do not have yet". You must get it by running the
> experiment. Never invent it.

---

## 1. What the paper says, in one sentence

> Only two engines (N8 and N9) are allowed to produce numbers. Every agent must say
> which engine value it used. So a made-up number cannot reach the user.

Every result in the paper must support this one sentence. Nothing else.

---

## 2. The trap you must avoid

This section is the most important one in the file. Please read it before anything
else.

### The obvious result is a trap

The result you probably want to show is:

> "We told agents to lie. Zero lies reached the final report."

**Do not make this your main result.** Here is why.

Your Faith Check works like this:

```
agent says     best_sr = 142.1
engine says    best_sr = 142.0
142.1 is not equal to 142.0   ->  reject
```

That is just Python checking whether two numbers are equal. It **cannot fail**. You
do not need 100 test runs to prove that `142.1 != 142.0`. It is true because of how
you wrote the code.

People call this **"true by construction"** — it is guaranteed by the design, not
discovered by an experiment.

A reviewer will read your table and write:

> *"The main result is just an equality check, presented as if it were a discovery."*

And they would be right. This can reject your paper.

### The same problem hits termination

You also planned to prove "the system always stops" by running 500 hard cases.

But your loops stop because you wrote `MAX_REGENS = 2`. A loop with a limit of 2 stops
after 2 times. That is the same reason a `for` loop stops.

Running 500 tests to check this makes you look **unsure about your own code**.

### The fix

You cannot fix this by running more tests. **You fix it by measuring different
things.**

Here is the new plan in three lines:

| Old question | New question |
|---|---|
| Does the gate block fake numbers? | **Where do other checking methods fail, and mine does not?** |
| (not asked) | **What does the blocking cost me?** |
| (not asked) | **Where does my gate stop protecting the user?** |

Those three questions become R1, R2 and R3. All three have answers you do **not**
already know, which is what makes them real results.

> **Copy this into the paper.**
>
> Containment in this architecture is a structural property rather than an empirical
> one: it holds by construction. We therefore do not evaluate whether it holds. We
> evaluate the conditions under which alternative verification strategies fail, the
> cost the property imposes on report utility, and the boundary beyond which it no
> longer protects the user.

---

## 3. How much space you have

Six pages is small. Plan it before you write.

| Part of the paper | Pages |
|---|---|
| Introduction | 0.75 |
| Related work | 0.5 |
| Architecture + Figure 1 | 1.5 |
| Walkthrough (Section 8) | 0.75 |
| **Results (R1, R2, R3)** | 1.5 |
| Discussion + limitations + conclusion | 0.5 |
| References | 0.5 |

So the results section is only **one and a half pages**. That is about two figures,
two small tables and five or six paragraphs. Split it like this:

| | What goes there | Pages |
|---|---|---|
| Setup | the graph, the fake data, the error model | 0.2 |
| **R1** | comparing three gates, Figure 3 | 0.5 |
| **R2** | what containment costs, Figure 4 | 0.4 |
| **R3** | where the gate fails | 0.3 |
| Small note | routing + termination | 0.1 |

---

## 4. R1 — Other checking methods fail here, mine does not

**This is your main result. Put it first.**

### The idea in simple words

There are two ways to check if a number is fake.

**Way 1 — "Does this number look wrong?"**
This is what most systems do, including LLM-as-a-judge. It asks whether the number
seems too big, too small, or strange.

**Way 2 — "Is this the number the engine actually gave?"**
This is what you do. You do not care if it looks fine. You only care if it matches.

Now compare them on the same fake numbers.

### The example

The engine says Kohli's cover drive strike rate is **142.0**. You force an agent to
report a different number.

| Agent reports | How big is the lie | Does it *look* wrong? | Way 1 catches it? | Way 2 (yours) catches it? |
|---|---|---|---|---|
| 142.1 | 0.1% | no | **no** | yes |
| 143.4 | 1% | no | **no** | yes |
| 149.1 | 5% | probably not | **no** | yes |
| 170.4 | 20% | maybe | yes | yes |
| 999.0 | 600% | obviously | yes | yes |

**Look at the top three rows.** Those are the dangerous lies — small enough that
nobody notices, big enough to change a decision. Way 1 misses all of them. Your gate
catches them, because it never asks "does this look right?".

### You must also run it with NO gate

Add a third column: the system with checking turned **off**. Without this you have
only described a mechanism. You have not shown it does anything.

So the paper's table has **three** conditions:

| Injected error | No gate | Way 1 (plausibility, limit = k) | Way 2 (yours) |
|---|---|---|---|
| 0.1% | reaches user | **misses** | catches |
| 1% | reaches user | **misses** | catches |
| 5% | reaches user | **misses** | catches |
| 20% | reaches user | catches | catches |
| 600% | reaches user | catches | catches |

### Answering the obvious objection

A reviewer will immediately ask:

> *"Your Way 1 uses a limit of 10%. Why not just lower the limit to 0.05%?"*

You must answer this **before** they ask. So test many limits, and check **two**
things for each one:

| Limit k | Catches a 0.1% lie? | Also rejects TRUE numbers? |
|---|---|---|
| 20% | no | no |
| 5% | no | no |
| 1% | no | `[measure]` |
| 0.05% | yes | `[measure]` |

**The finding: there is no good limit.**

- Set the limit high → small lies get through.
- Set the limit low → the system starts rejecting numbers that were correct.

Your method is not on this scale at all. It has no limit to tune, so it has no
trade-off to lose.

That is a real discovery. It is not something your code guarantees.

### One honest problem you must admit

Exact matching only works if the agent reports the **exact machine number**.

If the engine holds `142.037` and the agent writes `142.0` (just rounding for
display), your gate says "these are different" and **rejects a true statement**.

This is a genuine weakness. Do not hide it — explain it and give the rule that fixes
it:

> **Copy this into the paper.**
>
> Citations carry full engine precision; rounding is a presentation concern applied
> to prose after verification, and is never applied to the cited value.

Then measure both cases: with the full-precision rule, false rejections are zero;
without it, exact matching is the most fragile of the three gates.

Admitting this costs you three sentences and removes the reviewer's best attack.

### Figure 3

A line chart. Error size on the x-axis, detection rate on the y-axis, one line per
gate. Way 1's line drops off a cliff at its limit. Your line stays flat at the top.
Add a small second panel showing false rejections against k.

> **Copy this into the paper.**
>
> Threshold-based verification failed on every fabrication below its tolerance and
> began rejecting truthful claims below k = `[measure]`; no threshold separated the
> two. Exact citation matching detected fabrications independently of magnitude,
> because cited values are compared to the engines by identity rather than by
> judgement.

---

## 5. R2 — What the safety costs you

**This is the result other papers will not have. Do not skip it.**

### The idea in simple words

Your system is safe. But **how does it stay safe?**

By **throwing things away**. When an agent keeps giving a wrong number, the system
drops that agent's finding completely. So the report stays truthful — but it gets
**shorter**.

Think of a newspaper that refuses to print anything it cannot verify. It will never
print a lie. But if the checking is too strict, it prints a blank page. A blank page
is safe and useless.

**So: how empty does your report get?** Nobody has measured this. It is the first
question a smart reviewer asks.

### The table

Change how many agents lie, from none to all of them:

| Agents lying | Fake numbers reaching user | **How much of the report survives** | Extra agent calls |
|---|---|---|---|
| 0% | 0 | 100% | 1.0x |
| 25% | 0 | `[measure]` | `[measure]` |
| 50% | 0 | `[measure]` | `[measure]` |
| 75% | 0 | `[measure]` | `[measure]` |
| 100% (never fixes itself) | 0 | **0%** | `[measure]` |

Look at the two middle columns.

- Column 2 stays at **0** the whole way down. Safety never breaks.
- Column 3 **falls**. The report empties out.

That is the trade. Show it clearly. This is the honest heart of the paper.

### MAX_REGENS is not a safety setting

Many people would assume `MAX_REGENS` controls safety. **It does not.** Safety holds
even at `MAX_REGENS = 0`.

What it actually controls is **how much of the report you get back**, and you pay for
that in extra agent calls:

| MAX_REGENS | Report surviving | Agent calls per query |
|---|---|---|
| 0 | `[measure]` | `[measure]` |
| 1 | `[measure]` | `[measure]` |
| 2 | `[measure]` | `[measure]` |
| 3 | `[measure]` | `[measure]` |

This is a nice, clear point and it costs nothing extra to measure — it is the same
sweep with one more column.

### Figure 4

One chart, two lines, sharing the same x-axis (how many agents lie):

- Line A — fake numbers reaching the user: **flat at zero**
- Line B — report surviving: **falling**

The picture of one flat line and one falling line *is* your argument. Label both
clearly.

> **Copy this into the paper.**
>
> Containment held at every fabrication rate, but report completeness did not: with
> `MAX_REGENS = 2`, surviving findings fell from 100% to `[measure]`% as the
> fabrication rate rose to 100%, at a cost of `[measure]`x agent invocations. The
> architecture trades completeness for correctness, and the exchange rate is governed
> by the regeneration budget rather than by the verification mechanism.

### Why this result is so valuable

A paper that only says good things about itself reads like advertising.
A paper that says *"here is exactly what my safety costs"* reads like science.

Reviewers trust the second one. This single result will help your acceptance more
than any extra containment data.

---

## 6. R3 — Where your gate does NOT protect the user

This part makes you look honest and careful. Write it before a reviewer finds it for
you.

### The main point

**Your gate checks numbers. It does not check the sentence around the number.**

### The example that proves it

An agent writes:

> *"Kohli's pull shot is his strongest stroke, at SR 142.0."*
> It reports: `best_sr = 142.0`

Your gate checks 142.0 against the engine. They match. **It passes.**

But the sentence is **false**. 142.0 belongs to the *cover drive*. The pull shot is
112.0. The agent took a real number and attached it to the wrong shot — and your
system cannot see this at all.

Show this trace right next to a caught lie. The contrast is the whole point.

### The table of what you catch and what you miss

| Type of lie | Caught? | Why |
|---|---|---|
| Wrong number for a reported key | **yes** | it does not match the engine |
| Right number, wrong subject | **no** | you check numbers, not sentences |
| A number with no citation at all | **no** | you only check numbers that are cited |
| A made-up player or match | **no** | it is not a number |

### What this means for the real system

> **Copy this into the paper.**
>
> In deployment, agent output must be constrained so that every numeric token is a
> citation and every citation carries its subject — through a JSON schema or a tool
> definition rather than free prose. Where that constraint is absent the guarantee
> degrades silently, since an uncited figure is indistinguishable from a verified one.

In plain words: when you replace the templates with a real LLM, you **must** force it
to output structured data. If you let it write free text, an uncited number looks
exactly like a checked one, and your whole guarantee quietly disappears.

**Good news:** R3 needs **no experiment**. You write two agent templates by hand and
save two traces. That is all.

---

## 7. Two things you should make smaller

### Routing — reduce it to two sentences

Your router picks which agents wake up. You planned to measure precision and recall
against your own labels.

**Problem:** you wrote the labels yourself, and the router is a stub. Measuring your
own opinion against your own code is close to circular. A reviewer will not trust it.

So report it as a **cost saving** instead, which is a plain fact:

> **Copy this into the paper.**
>
> The router activated a mean of `[measure]` of twelve agents, a `[measure]`%
> reduction in agent invocations against an all-agents baseline.

Count **agent calls**, not tokens or money — no LLM runs, so agent calls are the only
honest unit. Thirty labelled queries is enough for this. You only needed sixty for the
precision/recall claim you are no longer making.

### Termination — turn the experiment into a statement

Do not run 500 tests. Just say it:

> **Copy this into the paper.**
>
> Both correction loops are bounded by fixed iteration budgets, so worst-case run
> length is determined at graph construction and is at most *N* supersteps. On budget
> exhaustion the graph degrades to an explicit status (`low_confidence`,
> `unverified_claims_dropped`) rather than failing or looping.

This is **stronger** than an experiment. An experiment says "it stopped 500 times out
of 500". This says "it can never not stop". A proof beats an observation. Delete the
sweep and save the day.

---

## 8. Show one query travelling through the system

This is not a result. It is a **walkthrough**. It is also probably the page a reviewer
will actually read carefully, so make it good.

Because every node is deterministic, you can print exactly what goes in and out of
each stage. Nothing is hidden.

**The question:** *"Compare Kohli's cover drive vs his pull shot against fast bowlers
in the powerplay"*

| Stage | What the system knows now |
|---|---|
| **N3** Text encoder | `player: Kohli`, `bowler_type: pace`, `phase: powerplay`, `shots: [Cover Drive, Pull]` |
| **N5** Classifier | `Cover Drive`, confidence `0.92` |
| **Conf. Check** | 0.92 >= 0.70 -> accept |
| **N6** Shot DNA | shot + posture (`weight_transfer 0.88`) + filters, joined together |
| **N7a** Stats retrieval | 100 of 150 deliveries match |
| **N7b** Context retrieval | 4 historical rows |
| **N8** Stats engine | `Cover Drive SR 142.0`, `Pull SR 112.0` |
| **N9** What-if simulator | 6 short balls -> `39.4%` chance of dismissal, CI `[38.7, 40.1]` |
| **N10** Router | wakes 2 of 12: Players, Coaches |
| **N11c** Players agent | reports `best_sr, worst_sr, weight_transfer` |
| **N11b** Coaches agent | reports `best_sr, worst_sr, p_pct` |
| **Faith Check** | all reported numbers match the engines -> pass |
| **N13** Output | final report |

### Show two agents using the SAME numbers

This is the most convincing thing in your whole paper:

> **Players:** "Weight transfer on the Cover Drive measures 0.88, and it is the
> stronger stroke at SR 142.0. The Pull lags at SR 112.0 — shot selection, not
> technique, is the gap to close."
>
> **Coaches:** "Bowl short at Kohli. The Pull returns SR 112.0 against the Cover
> Drive's SR 142.0; 6 consecutive back-of-a-length deliveries carry a 39.4% chance of
> dismissal."

Same numbers. Two different readers. Two different pieces of advice. **Zero
disagreement about the facts.** That is your whole thesis in five lines.

You must also say how the text was made:

> **Copy this into the paper.**
>
> Agent responses are generated from stakeholder-specific templates over the
> sanctioned facts. In deployment each template is replaced by a language-model call
> constrained to the same citation contract; the simulation isolates the
> architecture's containment behaviour from any particular model's generation quality.

That sentence turns a possible criticism into a stated design choice.

### And show it failing

```
[N11  ] Coaches: cites ['best_sr', 'worst_sr', 'p_pct']   <-- FABRICATED
[FAITH] REJECT n11b_tactical_analysis: best_sr=999.0 but engines say 142.0
[REGEN] pass 1/2, re-running 1: n11b_tactical_analysis
[N11  ] Coaches: cites ['best_sr', 'worst_sr', 'p_pct']
[FAITH] all 2 findings verified
```

Caption:

> Illustrative execution trace. A fabricated citation is detected, the agent re-run,
> and the corrected finding verified. Values come from a synthetic corpus and do not
> describe real match data.

---

## 9. Do not fake your statistics

Your nodes are deterministic and your seed is fixed. That means many of your test
cells will give **exactly the same answer every single time**. The variation is zero.

So do **not** write things like "mean 0.0 ± 0.0 over 100 runs". It looks like real
statistics but there is nothing there, and a reviewer who notices will stop trusting
the rest of your numbers.

Say the truth instead:

> **Copy this into the paper.**
>
> Node bodies are deterministic. Stochasticity enters only through the Monte Carlo
> what-if simulator (N9) and the random selection of fabricating agents; conditions in
> which neither varies were run once and are reported without dispersion.

Run many trials **only where something actually changes**. Being open about this makes
you look careful, not weak.

---

## 10. Figures

Four figures maximum in six pages.

| Figure | What it shows | Where it comes from |
|---|---|---|
| 1 | Architecture diagram | `paper/CricketMind_Architecture_v2.pdf` — already done |
| 2 | The query walkthrough as a flow | Section 8 |
| 3 | **Detection rate vs error size, three gates** | R1 |
| 4 | **Safety stays flat while report shrinks** | R2 |

Figures 3 and 4 hold your real measurements. Save both as **vector PDF**, not PNG. Use
two colours only, no decoration, and check they are still readable in black and white
at about 8.5 cm wide — at least one reviewer will print your paper in greyscale.

---

## 11. Things you must NOT say

- That any number describes a real player or a real match. It does not.
- That your results predict what a real LLM will do. You never ran one.
- That analysis quality or user satisfaction improved. You did not measure either.
- That you beat a competing system. You did not run one.
- **Important:** the plausibility gate in R1 is **your own code**, not a published
  system. Say so plainly. If you imply you tested someone else's tool, that is a
  serious problem.

---

## 12. Two paragraphs your paper must contain

These protect you. Write them yourself, or a reviewer will write them for you in a
rejection.

> **Copy this into the paper.**
>
> **Error model.** We model agent fabrication as corruption of a single cited value by
> relative magnitude eps, and additionally as misattribution of a correct value and as
> emission of an uncited figure. This simplifies real language-model error, which is
> correlated across claims, context-dependent, and may fabricate entities as well as
> figures. Results characterise the architecture's containment behaviour under this
> model, not the behaviour of any particular language model.

> **Copy this into the paper.**
>
> **Implementation status.** All node bodies are deterministic simulations. No language
> model is invoked, no video is decoded, and retrieval operates over a synthetic corpus
> with known ground truth. Values in the worked example illustrate data flow and are
> not derived from match data.

Writing your own limitations is not weakness. It is what stops a reviewer using them
against you.

---

## 13. The order to write the section in

1. **Setup** — the graph, the fake data, the error model, the seed, how many runs, and
   the note about determinism. Two short paragraphs.
2. **R1** — the three-gate comparison. Your strongest result.
3. **R2** — what containment costs.
4. **R3** — where the gate fails, with the misattribution example.
5. **Small note** — routing saving and the termination guarantee.
6. **Threats to validity** — fake data, simple error model, no LLM used, the
   plausibility gate is your own code, only one sport and one question style.

---

## 14. Why this plan is better than the old one

The old plan told **one** story: *it works.*

The new plan tells **three**:

- **R1** — it works where the normal method fails
- **R2** — and here is exactly what that costs
- **R3** — and here is exactly where it stops working

Results 2 and 3 are what turn this from a product description into a research paper.
Both are also **cheaper** to produce than the experiments they replace.

### If you run out of time

Build in this order:

1. **R2 first.** Best value for the hours spent, and it mostly uses instrumentation
   you need anyway.
2. **R1 second.** The plausibility gate is about twenty lines of code.
3. **R3 last.** No experiment at all — two templates and two traces.

**R1 and R2 together are already a complete, defensible paper.** If everything else
falls apart, make sure those two exist.
