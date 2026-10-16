# Related work notes

For each paper: what they did, and how CricketMind differs. Add a BibTeX entry to
`references.bib` the same day. **Verify every paper yourself before citing it**; the first
entries below come from a quick search on 2026-10-16.

## Verifying numbers and claims from LLMs (the closest work; position against it)

**Proof-Carrying Numbers (Solatorio, 2025, arXiv:2509.06902).** Each number an LLM displays
must link to a verified structured claim, or be marked unverified.
*How we differ:* CricketMind also binds each claim to its *subject* and re-queries the engine
for that subject (C2, C6); checks wording against uncertainty (C4); and refuses unsupported
*kinds* of claims at build time (registry).

**Tool receipts / NabaOS (Basu, 2026, arXiv:2603.10060).** The runtime signs a receipt for every
tool call (HMAC); the agent's claims are checked against the receipts it cannot forge.
*How we differ:* we verify against sanctioned engines with subject-keyed re-queries, not
against a record of calls; and we check claim strength, not only existence.

**Collective hallucination in multi-agent LLMs (Jamshidi, 2026, arXiv:2606.07941).** Splits
multi-agent outputs into atomic claims to study how hallucinations spread between agents.
*How we differ:* we prevent rather than measure. Claims are atomic by construction, and a
gate blocks them before release.

## LLMs expressing uncertainty

**Lin, Hilton and Evans (2022, TMLR; arXiv:2205.14334).** Teaching models to state their own
confidence in words.
**Wang et al. (2024, arXiv:2410.06707).** Calibrating verbalised probabilities.
*How we differ:* both concern a model's *own* confidence. We check whether an agent's words
match an *external engine's* interval, one that includes estimation uncertainty.

**Kent (1964), "Words of estimative probability".** The source of the wording bands.

## Cricket and multi-agent systems

**FanCric (Bhatnagar, 2024, arXiv:2410.01307).** LLM agents selecting fantasy cricket teams.
Already cited in the ICCIT paper.
**CricBench (Devraj, Kumar and Challa, 2025, arXiv:2512.21877).** A benchmark of LLMs on cricket
analytics questions (text-to-SQL): most queries execute, few are semantically correct.
*Use:* direct evidence that cricket numbers produced by LLMs need verification (N-007).

## Methods the engines and the bridge rely on

- Efron and Morris (1975): shrinkage (Stein's estimator), the classic baseball example.
- Brier (1950): scoring probability forecasts.
- Cohen (1960); Landis and Koch (1977): agreement, κ, and the "substantial" threshold.
- Ratner et al. (2017), Snorkel: weak supervision.
- Hulin et al. (2014); Impellizzeri et al. (2021): the workload ratio, and its critique.

## To read (verify that each exists, then add notes)

- Self-correction and verification loops: Reflexion; Self-Refine; CRITIC.
- Tool-grounded and program-aided reasoning: PAL; Toolformer.
- Fine-grained factuality: FActScore; RARR.
- Multi-agent frameworks: AutoGen; CrewAI; LangGraph documentation.
