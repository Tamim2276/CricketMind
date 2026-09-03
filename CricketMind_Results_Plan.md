# What Goes in the Results Section

For a **6-page** conference paper on the CricketMind agentic framework, evaluated
in simulation.

> Numbers marked `[measure]` are placeholders. They must come from actually running
> the experiments. Values written in are ones the design guarantees, and those are
> labelled.

---

## 1. The one sentence the paper argues

> If two auditable engines are the only components allowed to produce numbers, and
> every agent must cite them, then fabricated figures cannot reach the user.

Everything in the results supports that single sentence.

---

## 2. Page budget

Six pages is tight. Roughly:

| Section | Pages |
|---|---|
| Introduction | 0.75 |
| Related work | 0.5 |
| Architecture + Figure 1 | 1.5 |
| **Walkthrough** (Section 6 below) | 0.75 |
| **Results** (three of them) | 1.5 |
| Discussion, limitations, conclusion | 0.5 |
| References | 0.5 |

So the results section is about **a page and a half**: two tables, two figures, and
roughly four paragraphs. Three results is the right number. Four would crowd it.

---

## 3. Result 1 — Fabricated numbers never reach the report

**This is now your headline result.** Put it first.

### The idea, simply

You order some agents to lie. You count how many lies survive into the final
report. You run it twice — once with the safety check on, once with it off.

It is a smoke-alarm test. You do not wait for a real fire. You hold a lighter under
the alarm and check that it goes off.

### A concrete example

The engines compute that Kohli's cover drive strike rate is **142.0**. You force the
Coaches agent to cite a different number instead:

| Agent says | Size of the lie | Would a person spot it? | Gate catches it? |
|---|---|---|---|
| 142.1 | 0.1% | No | Yes |
| 143.4 | 1% | No | Yes |
| 149.1 | 5% | Probably not | Yes |
| 170.4 | 20% | Maybe | Yes |
| 999.0 | 600% | Obviously | Yes |

**That is the interesting part.** Most hallucination checks judge whether a number
*looks* reasonable, so they catch 999 and miss 142.1. Yours compares the cited value
against what the engine actually produced. 142.1 ≠ 142.0, so it fails — the size of
the lie is irrelevant.

### The table for the paper

| Size of injected error | Runs | False claims in final report — **gate off** | **gate on** |
|---|---|---|---|
| 0.1% | 100 | `[measure]` | **0** *(by design)* |
| 1% | 100 | `[measure]` | **0** |
| 5% | 100 | `[measure]` | **0** |
| 20% | 100 | `[measure]` | **0** |
| 600% | 100 | `[measure]` | **0** |

The **gate-off column is essential.** Without a control condition you have described
a mechanism, not shown that it does anything.

### The sentence you write

> With verification disabled, every injected fabrication reached the final report.
> With verification enabled, none did, at any error size — including a 0.1%
> deviation that no plausibility check would flag. Detection is independent of error
> magnitude because cited values are compared to the engines by exact match rather
> than by judgement.

---

## 4. Result 2 — Routing keeps the report short

### The idea, simply

Twelve specialists are available. A coach asking about bowling tactics should not
receive a paragraph about floodlight scheduling. You check that the router picks
sensible agents, and measure how much work it saves.

### A concrete example

| The user asks | Agents that should wake | Agents that woke |
|---|---|---|
| "How should we bowl to Kohli in the powerplay?" | Coaches | Coaches, Players |
| "Any injury risk from his workload this week?" | Medical Teams | Medical Teams |
| "What is his IPL auction valuation?" | Player Agents, Franchises | `[measure]` |
| "Make a highlights clip for fans" | Fans, Broadcasters | `[measure]` |

You need a hand-labelled set of about **60 queries** — each written down with the
agents you think should wake. That is roughly half an hour of work and it is the
only hand-labelling in the entire paper.

### The table for the paper

| Metric | Value |
|---|---|
| Mean agents woken | `[measure]` of 12 |
| Reduction vs waking all twelve | `[measure]`% |
| Routing precision | `[measure]` |
| Routing recall | `[measure]` |

### The sentence you write

> The router activated a mean of `[measure]` of twelve agents — a `[measure]`%
> reduction in agent invocations against an all-agents baseline — at `[measure]`
> precision and `[measure]` recall on a hand-labelled query set.

Count **agent invocations**, not tokens or money. No language model is called, so
invocations are the honest unit.

---

## 5. Result 3 — The system always stops

### The idea, simply

Both correction loops could in theory spin forever. You prove they do not by
building the worst case on purpose: an agent that refuses to correct itself, and a
classifier that is never confident.

One short paragraph. It closes a question a reviewer would otherwise ask.

### The table for the paper

| Worst case | Runs | Terminated | Ends with status | Longest run |
|---|---|---|---|---|
| Agent never corrects its citation | 500 | `[measure]` | `unverified_claims_dropped` | `[measure]` steps |
| Classifier never becomes confident | 500 | `[measure]` | `low_confidence` | `[measure]` steps |
| Both at once | 500 | `[measure]` | `[measure]` | `[measure]` steps |

### The sentence you write

> Under adversarial conditions the system terminated in every run, degrading to an
> explicit status rather than failing or looping. Both loops are bounded by an
> iteration budget, so worst-case run length is fixed when the graph is built.

---

## 6. Showing how a query flows through the system

**Yes — you can and should do this, and it is one of the most valuable things in a
6-page paper.** It is not a "result"; it is a walkthrough. It shows a reviewer in
one figure what three pages of prose cannot.

Because every node is deterministic, you can print exactly what enters and leaves
each stage. Nothing is hidden or approximated.

### The walkthrough table

Take one query and follow it down the page:

**Input:** *"Compare Kohli's cover drive vs his pull shot against fast bowlers in
the powerplay"*

| Stage | What the system now holds |
|---|---|
| **N3** Text encoder | `player: Kohli`, `bowler_type: pace`, `phase: powerplay`, `shots: [Cover Drive, Pull]` |
| **N5** Classifier | `Cover Drive`, confidence `0.92` |
| **Conf. Check** | 0.92 ≥ 0.70 → accept |
| **N6** Shot DNA | shot + posture (`weight_transfer 0.88`) + query filters, fused |
| **N7a** Stats retrieval | 100 of 150 deliveries match the filters |
| **N7b** Context retrieval | 4 historical rows |
| **N8** Stats engine | `Cover Drive SR 142.0`, `Pull SR 112.0` |
| **N9** What-if simulator | 6 short balls → `39.4%` dismissal chance, CI `[38.7, 40.1]` |
| **N10** Router | wakes 2 of 12: Players, Coaches |
| **N11c** Players agent | *"Weight transfer on the Cover Drive measures 0.88…"* cites `best_sr, worst_sr, weight_transfer` |
| **N11b** Coaches agent | *"Bowl short at Kohli…"* cites `best_sr, worst_sr, p_pct` |
| **Faith Check** | all cited values match the engines → pass |
| **N13** Output | final report |

One query, one page, whole system. This is the figure a reviewer will actually read.

### Showing the agents' responses

You can absolutely show the agent text. Two rules.

**One — say how it was produced.** The agents are templates over the verified facts,
not language-model calls. Write it plainly:

> Agent responses are generated from stakeholder-specific templates over the
> sanctioned facts. In deployment each template is replaced by a language-model call
> constrained to the same citation contract; the simulation isolates the
> architecture's containment behaviour from any particular model's generation quality.

That sentence turns a possible criticism into a stated design choice.

**Two — show two agents on the same evidence.** This is the most persuasive thing in
the paper, because it demonstrates the whole point of the architecture in five lines:

> **Players:** "Weight transfer on the Cover Drive measures 0.88, and it is the
> stronger stroke at SR 142.0. The Pull lags at SR 112.0 — shot selection, not
> technique, is the gap to close."
>
> **Coaches:** "Bowl short at Kohli. The Pull returns SR 112.0 against the Cover
> Drive's SR 142.0; 6 consecutive back-of-a-length deliveries carry a 39.4% chance
> of dismissal."

Same numbers. Same evidence. Two different audiences, two different
recommendations, zero disagreement about the facts. That is the thesis in one
example.

### And show it failing

Include the fabrication trace next to it:

```
[N11  ] Coaches: cites ['best_sr', 'worst_sr', 'p_pct']   <-- FABRICATED
[FAITH] REJECT n11b_tactical_analysis: best_sr=999.0 but engines say 142.0
[REGEN] pass 1/2, re-running 1: n11b_tactical_analysis
[N11  ] Coaches: cites ['best_sr', 'worst_sr', 'p_pct']
[FAITH] all 2 findings verified
```

Caption it:

> Illustrative execution trace. A fabricated citation is detected, the agent re-run,
> and the corrected finding verified. Values come from a synthetic corpus and do not
> describe real match data.

---

## 7. Figures

Four, no more, in six pages.

| Figure | Content | Where it comes from |
|---|---|---|
| 1 | Architecture diagram | `paper/CricketMind_Architecture_v2.pdf` — already built |
| 2 | Query walkthrough (Section 6 table, drawn as a flow) | already have the data |
| 3 | Containment: false claims reaching the report, gate on vs off, by error size | Result 1 |
| 4 | Execution trace showing catch → regenerate → pass | already have it |

Figure 3 is the one with real measurement in it. Make it a simple bar chart: two
bars per error size, one tall and one at zero.

---

## 8. What you must not claim

- That any number describes a real player or a real match.
- That the results predict how a specific language model behaves.
- Any improvement in analysis quality or user satisfaction — neither was measured.
- Any comparison against a system you did not build and run.

The worked example is fine to show, clearly labelled, because it illustrates data
flow rather than reporting findings.

---

## 9. Two paragraphs that must appear

> **Error model.** We model agent fabrication as corruption of a single cited value
> by relative magnitude ε. This simplifies real language-model error, which is
> correlated across claims, context-dependent, and may fabricate entities as well as
> figures. Results characterise the architecture's containment behaviour under this
> model, not the behaviour of any particular language model.

> **Implementation status.** All node bodies are deterministic simulations. No
> language model is invoked, no video is decoded, and retrieval operates over a
> synthetic corpus with known ground truth. Values in the worked example illustrate
> data flow and are not derived from match data.

Writing your own limitations is not weakness. It is what stops a reviewer writing
them for you in a rejection.

---

## 10. Order of the results section

1. **Setup** — the graph, the synthetic corpus, the error model, the seed, how many
   runs. Two short paragraphs.
2. **Result 1** — containment, with the gate-off control. Your strongest result.
3. **Result 2** — routing efficiency.
4. **Result 3** — termination. One paragraph.
5. **Walkthrough** — the flow table and the two agent responses.
6. **Threats to validity** — synthetic corpus, simplified error model, no language
   model invoked, routing labels are your own.
