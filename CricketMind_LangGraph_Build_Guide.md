# CricketMind — LangGraph Build Guide

**Today's target:** a compiled LangGraph containing every node from Architecture v2,
with both feedback loops wired and working, where **every node is a dummy function**.

No torch forward passes. No FAISS. No LLM calls. No API keys. The whole graph
executes end-to-end in well under a second on either machine.

**Working file:** [notebooks/cricketmind_langgraph_demo.ipynb](notebooks/cricketmind_langgraph_demo.ipynb)
**Architecture reference:** [paper/CricketMind_Architecture_v2.pdf](paper/CricketMind_Architecture_v2.pdf)

---

## Why build it with dummies first

The graph *topology* is the thesis contribution — the routing, the two correction
loops, the separation of analytical services from stakeholder agents. None of that
needs a real model to be correct or incorrect.

If the wiring is proven with stubs, then swapping in a real EfficientNetV2-S or a
real supervisor LLM later is a change **inside one function body**. The graph never
changes. You debug the hard part (control flow) while it is cheap to run and
impossible to be confused by model noise.

The opposite order — real models first, wiring later — means every graph bug looks
like a model bug and every run costs a minute.

---

## Node inventory (Architecture v2)

| Node | Name | Tier | Step |
|---|---|---|---|
| N1 | Video Input (15 frames) | encoding | 2 |
| N2 | Vision Encoder (EfficientNetV2-S) | encoding | 2 |
| N3 | Text Encoder (query semantics) | encoding | 4 |
| N4 | Temporal Fusion (Temporal Transformer, 2L/4H) | encoding | 2 |
| N5 | Classifier Head (15 classes) | encoding | 2 |
| — | **Conf. Check** → retry loop | gate | 3 |
| N6 | Multimodal Shot DNA Gen. | grounding | 4 |
| N7a | Stats Retrieval (FAISS / Cricsheet) | retrieval | 5 |
| N7b | Context Retrieval (FAISS / Kaggle) | retrieval | 5 |
| N8 | Statistical Analysis Engine | **analytical service** | 6 |
| N9 | What-If Simulator (Monte Carlo) | **analytical service** | 6 |
| N10 | Supervisor Router (dynamic dispatch) | routing | 7 |
| N11a–N11l | The twelve stakeholder agents | agents | 8 |
| — | **Faith Check** → regenerate loop | gate | 9 |
| N12 | Supervisor LLM (report synthesis) | synthesis | 10 |
| N13 | Final Output (analytics report) | synthesis | 10 |

**14 agents total** = 12 stakeholder-facing (N11a–N11l) + 2 shared analytical
services (N8, N9).

---

## The steps

Each step is small on purpose. Build it, run the checkpoint, confirm it's right,
then move on. Nothing gets built ahead of its checkpoint.

### Step 0 — Install and verify

Install `langgraph`, `langchain-core`, `pydantic`, `ipykernel` into `.venv`, then
run the existing imports cell.

`torch` and `torchvision` are already present (2.13.0+xpu) and are **not** touched —
they're installed per-machine with different wheels.

> **Checkpoint:** the imports cell prints every version with no traceback, and the
> device cell reports `xpu` on the Arc box / `cuda` on the RTX box.

---

### Step 1 — The State object

A `TypedDict` that every node reads from and writes into. This is the single most
important design decision in the notebook — every later step only adds fields to it.

Fields it needs to carry:

- **inputs** — `video_path`, `query`
- **perception** — `frames`, `frame_features`, `temporal_features`, `shot_label`, `shot_confidence`
- **control** — `retry_count`, `regen_count`
- **grounding** — `shot_dna`, `query_embedding`
- **retrieval** — `stats_hits`, `context_hits`
- **analytics** — `stats_results` (from N8), `sim_results` (from N9)
- **dispatch** — `active_agents` (which of the twelve N10 woke)
- **findings** — `agent_findings`, appended in parallel → needs a reducer
- **output** — `report`, `status`

Parallel branches that write to the same field must use
`Annotated[list, operator.add]`, or the second writer silently overwrites the first.

> **Checkpoint:** you read the field list and agree it covers the architecture.
> No code runs yet.

---

### Step 2 — Vision path N1 → N2 → N4 → N5, linear

Four stub nodes wired `START → N1 → N2 → N4 → N5 → END`. Compile and invoke.

Each stub has the signature `(state) -> dict` and returns only the fields it
changed. N1 returns fake frame handles, N2 fake feature shapes, N4 a fake fused
vector, N5 a hardcoded label and confidence.

> **Checkpoint:** the graph actually runs. You pass in a video path and a query and
> get back a state containing `shot_label="cover_drive"`, `shot_confidence=0.92`.

---

### Step 3 — Conf. Check and the retry loop

A conditional edge after N5 with two outcomes: `confident → continue`,
`retry → back to N1`. Plus `retry_count` and a hard cap.

The cap is not optional. LangGraph will happily loop forever, and a stub that always
returns low confidence will hang the kernel.

On exhaustion the graph must degrade — report low confidence rather than proceed
with a label it does not trust.

> **Checkpoint:** force `shot_confidence` to 0.4 and watch the loop fire, increment,
> and bail cleanly at max retries.

---

### Step 4 — N3 Text Encoder and N6 Shot DNA

N3 embeds the query and runs **in parallel** with the vision branch — it does not
wait for classification. N6 is the join point: it fuses the accepted shot label,
the query embedding, and fake posture metrics into one `shot_dna` record.

After N6, nothing touches pixels again. That's the interface boundary.

> **Checkpoint:** both branches converge at N6 and `shot_dna` contains contributions
> from each. Confirm N3 ran even though the vision path was slower.

---

### Step 5 — N7a / N7b retrieval fan-out

Two stub retrievers running in parallel, returning fake rows — Cricsheet-shaped
ball-by-ball records from N7a, broader historical context from N7b.

This is where the Step 1 reducers earn their keep. Without them, one retriever's
results overwrite the other's.

> **Checkpoint:** both retrievers' results are present in the final state
> simultaneously. If one is missing, the reducer is wrong.

---

### Step 6 — N8 Stats Engine and N9 What-If Simulator

The two shared analytical services. These are **not** LLM nodes even in the real
system — N8 is deterministic aggregation, N9 is Monte Carlo sampling.

They hold exclusive authority to emit numbers. Every agent downstream consumes their
output rather than doing arithmetic itself. Stub them with the figures from the
thesis running example: SR 142 on cover drives vs 112 on pulls; 40% dismissal
probability on the pull counterfactual.

> **Checkpoint:** `stats_results` and `sim_results` both populate, and every number
> in them is traceable to one of these two nodes.

---

### Step 7 — N10 Supervisor Router

Inspects the query and the analytics, then decides **which subset** of the twelve
agents to wake. Not all twelve on every query — that's the whole point of routing.

Keyword matching is fine for the dummy version. The real version becomes an LLM call
behind the same signature.

> **Checkpoint:** a technique query wakes Personal Performance + Talent Development.
> A tactics query wakes Tactical Analysis. A workload query wakes Injury Management.
> `active_agents` reflects the choice and the others never run.

---

### Step 8 — The twelve stakeholder agents N11a–N11l

Write **one factory function**, not twelve near-identical node bodies. Each agent is
the same shape — read the evidence, produce a finding in its own vocabulary — and
differs only in configuration: its id, its stakeholder, its vocabulary, its output
format.

Twelve copy-pasted functions is twelve places to fix every future change.

> **Checkpoint:** all twelve are registered on the graph, but only the routed subset
> executes. `agent_findings` accumulates one entry per active agent, in any order,
> without loss.

---

### Step 9 — Faith Check and the regenerate loop

The hallucination gate. Every numeric claim in every agent finding is checked against
what N8 and N9 actually emitted. Anything unsupported fails the gate and sends that
agent back.

This is much easier to implement in v2 than it would have been in v1, because there
are exactly two sanctioned sources of numbers to check against.

Needs its own counter and cap, same as the retry loop.

> **Checkpoint:** inject a finding claiming an average of 999, watch the gate catch
> it, route back, and regenerate. Confirm a clean finding passes untouched.

---

### Step 10 — N12 synthesis, N13 output, and the picture

N12 assembles the validated findings into a report ordered by relevance to the
original query. N13 emits it.

Then render the compiled graph as a Mermaid diagram
(`graph.get_graph().draw_mermaid()`) and compare it against Figure 1 of the
architecture PDF. They should match node for node.

> **Checkpoint:** one invoke, from a video path plus a natural-language query, all
> the way to a formatted report — and a generated diagram that matches your thesis
> figure.

---

## Design decisions

These hold for every step unless you say otherwise.

1. **Stubs are pure Python** — no torch, no randomness. Deterministic, so a failing
   checkpoint is always a wiring bug and never noise.
2. **State is one flat `TypedDict`** with `Annotated` reducers on the parallel fields.
   Flat is far easier to inspect in a notebook than nested models.
3. **Every loop has a counter and a hard cap.** Non-negotiable — see Step 3.
4. **Stub signatures match the real thing**: `(state) -> dict`. Later you replace a
   function body, never a signature and never an edge.
5. **The twelve agents come from a factory**, never twelve pasted functions.

---

## Hardware

The dummy graph is pure Python and runs identically on the Intel Arc B580, the
NVIDIA RTX 3070, and CPU-only — there is nothing to accelerate yet.

The device helpers already in the notebook (`get_device`, `get_amp_settings`,
`empty_cache`) stay in place for when the real N2 and N4 land in a later session.
They resolve **XPU → CUDA → CPU** at runtime, so nothing in this project ever
hardcodes `cuda`.

---

## Two open issues

1. **`langgraph` is not installed yet** — nor `pydantic` or `ipykernel`. Step 0
   handles this. `torch` and `torchvision` are already present and stay untouched.

2. **The 15-class list in the notebook is a placeholder.** [src/](src/) is currently
   deleted from the working tree, so the real label ordering used by the trained
   checkpoint could not be read. If the ordering is wrong, the classifier head is
   silently mislabeled. `git restore src configs data` brings it back, and the list
   should be corrected before N5 is made real.
