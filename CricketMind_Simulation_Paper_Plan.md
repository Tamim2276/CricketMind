# CricketMind — Simulation-Based Paper Plan

**Situation:** a conference deadline one week out. The CricShot10k classification
results are reserved for a different submission, so this paper cannot use them.
The system's node bodies are deterministic stubs, not real models.

**Question:** can the agentic framework be submitted on simulation results alone,
how novel is that, and what should the simulation be built with?

**Short answer:** yes, and losing the classification results forces a cleaner
paper. The contribution now rests entirely on the agentic layer, which is where the
novelty was in the first place. Do not add a simulation framework — the stub graph
already is one. What is missing is an experiment harness around it.

---

## 1. Reframe the contribution

Not *"a cricket analytics system"* — that claim needs real cricket results.
Instead:

> A verification-first multi-agent architecture in which numeric authority is
> centralised in two auditable engines, evaluated **in simulation** for its ability
> to contain fabricated claims.

The object under test is the **control flow and the verification mechanism**, not
cricket analytics. For that question simulation is not a compromise, it is the
correct method — and this should be argued explicitly in the paper rather than
apologised for:

> With a live language model there is no ground truth about which claims were
> fabricated, so containment cannot be measured directly. In simulation the
> fabrication is injected and therefore known exactly, which permits precise
> measurement of detection and correction rates.

That single argument converts an apparent weakness into a methodological
justification. Reviewers will accept simulation for architectural properties; they
will not accept it for claims about cricket or about real model behaviour.

---

## 2. Honest novelty assessment

| Element | Novelty | Notes |
|---|---|---|
| LLM multi-agent system with a supervisor | **Low** | AutoGen, CrewAI, standard LangGraph supervisor patterns |
| Self-correction / verification loops | **Low** | Reflexion, Self-Refine, CRITIC |
| Centralising numeric authority and validating agent **citations** against it | **Moderate** | Related to tool-grounding and program-aided LMs; the citation-contract framing is a genuine design pattern |
| Twelve stakeholder-specific agents narrating shared evidence | **Moderate as application**, low as method | The stakeholder decomposition is the domain contribution |
| A simulation testbed for measuring hallucination containment in a MAS | **Moderate to good** | Underexplored, and the most defensible novel angle here |

**Realistic target:** workshop, short paper, or an applied / work-in-progress
track. Not a top-tier full research paper.

The genuine strength is that the system is complete, executable and measurable.
Most architecture papers are none of those three. Lead with that.

---

## 3. Tooling: do not add a simulation framework

The stub graph **is** the simulator. Every node is deterministic, every parameter
is a knob, and the whole pipeline runs end to end in well under a second.

Explicitly **do not** use:

- **SimPy** — discrete-event simulation. Wrong execution model; LangGraph already
  provides one, and a reviewer will ask why two are present.
- **Mesa** — agent-based modelling for spatial and social systems. Wrong domain.

What is actually needed:

| Component | Status |
|---|---|
| System under test — the LangGraph stub graph | already built |
| Parameterised synthetic corpus — `_expand()` | already built |
| Fabrication injection — `HALLUCINATE_ONCE` / `HALLUCINATE_ALWAYS` | already built, needs magnitude control |
| Sweep driver — conditions x trials | **to build** |
| Aggregation and statistics — pandas | **to build** |
| Figures — matplotlib | **to build** |

Roughly a day's work, most of it in the sweep driver.

---

## 4. The experiments

Three of these are enough for a short paper. **Experiment 2 is the headline** — it
directly evidences the central design claim and it is a *comparison*, which
reviewers weight far more heavily than a description.

### Experiment 1 — Fabrication containment

**Design.** Inject a corrupted citation with controlled relative magnitude.

| Factor | Levels |
|---|---|
| Corruption magnitude ε | 0.1%, 1%, 5%, 20%, 500% |
| Lying agents per run | 1, 2, 3, 4 |
| `MAX_REGENS` | 0, 1, 2, 3 |
| Condition | gate enabled / **gate disabled (control)** |

The control condition is essential. Without it the result is a description of a
mechanism rather than evidence that it does anything.

**Metrics.** Residual false-claim rate in the final report; detection rate;
correction rate; withhold rate.

**Expected result.** Zero false claims reach output with the gate enabled, at every
ε including 0.1%, because verification is exact-match against the sanctioned facts
rather than a plausibility judgement. Without the gate, the rate rises with the
number of lying agents.

The ε = 0.1% row is the interesting one: a near-miss of 142.0 → 142.4 would defeat
any plausibility-based check, and is caught here by construction. It also raises a
real discussion point — whether the gate should tolerate rounding, and what that
tolerance would cost.

### Experiment 2 — Version 1 versus Version 2 architecture

**Design.** Simulate both architectures on identical evidence.

* **Version 1** — six generic workers, each computing its own statistics, with
  small independent arithmetic errors standing in for per-prompt LLM error.
* **Version 2** — twelve stakeholder agents consuming centralised engine output.

**Metric.** *Internal inconsistency rate*: the fraction of reports in which two
agents quote different values for the same underlying statistic.

**Expected result.** Version 1 produces inconsistent reports at some rate that
grows with the number of active workers; Version 2 cannot produce them at all, by
construction, because there is only one source for each figure.

This is the strongest and most novel result obtainable in a week. It converts the
architectural argument from an assertion into a measurement.

### Experiment 3 — Routing efficiency

**Design.** A labelled query set — roughly 60–100 queries, each annotated with the
agents that *should* wake.

**Metrics.** Precision and recall of agent selection; agent invocations per query
against the all-twelve baseline.

**Expected result.** With `MAX_ACTIVE_AGENTS = 4`, a 67–83% reduction in agent
invocations. Report **agent invocations** as the cost unit rather than tokens or
currency — honest, since no language model is called.

### Experiment 4 — Termination guarantees

**Design.** Adversarial conditions: an agent that never corrects, a classifier that
never becomes confident.

**Metrics.** Termination rate; distribution of supersteps per run.

**Expected result.** Both bounded loops always terminate, and the graph degrades to
a declared status (`low_confidence`, `unverified_claims_dropped`) rather than
failing or looping. Cheap to run, and it closes an obvious reviewer question.

---

## 5. The disclosure the paper must contain

The simulation's validity rests entirely on its error model being stated. Without
this paragraph, a reviewer can correctly say the paper measures its own
assumptions.

> **Error model.** We model agent fabrication as corruption of a single cited value
> by relative magnitude ε. This is a simplification of real language-model error,
> which is correlated across claims, context-dependent, and may fabricate entities
> as well as figures. Our results characterise the architecture's containment
> properties under this model; they are not measurements of the fabrication
> behaviour of any specific language model.

Alongside it, an implementation-status note:

> **Implementation status.** All node bodies are deterministic simulations. No
> language model is invoked, no video is decoded, and retrieval operates over a
> synthetic corpus with known ground truth. Figures shown in the worked example are
> illustrative of data flow and are not derived from match data.

---

## 6. What must not be claimed

- That any simulated statistic says anything about a real player or real match.
- That the results predict how a specific LLM would behave.
- Any accuracy, quality or user-satisfaction improvement that has not been measured.
- Any comparison against a baseline system that was not actually run.

The worked example may be shown — clearly labelled as illustrative — because it
demonstrates data flow, not findings.

---

## 7. Suggested week

| Day | Work |
|---|---|
| 1 | Read the CFP, choose the track, obtain the template. Fix the exact claims. |
| 2 | Build the experiment harness: sweep driver, metrics, aggregation. |
| 3 | Run Experiments 1 and 2; produce figures. |
| 4 | Run Experiments 3 and 4; write the results section. |
| 5 | Methodology section — largely transplantable from `paper/CricketMind_Architecture_v2.tex`. |
| 6 | Introduction, related work, limitations, error-model and implementation-status disclosures. |
| 7 | Abstract, proofread, figures, formatting, submit with buffer. |

Day 5 is assembly rather than writing: the architecture document already contains
the node-by-node methodology, the diagram and the agent roster table.

---

## 8. Related work to cite

Worth positioning against, since these are what a reviewer will have in mind:

- **Multi-agent LLM frameworks** — AutoGen, CrewAI, LangGraph supervisor patterns.
- **Self-correction** — Reflexion, Self-Refine, CRITIC.
- **Grounding numeric reasoning outside the model** — program-aided language models,
  tool use, retrieval-augmented generation.
- **Hallucination detection and measurement** — factuality benchmarks and
  citation-verification work.

The gap to claim: these verify *a model's own output against retrieved text*,
whereas this architecture makes numeric authority **structural** — a class of
component that is the only permitted source of figures, with every downstream
claim carrying a machine-checkable citation to it.
