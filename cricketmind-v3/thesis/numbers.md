# Numbers the paper may print

Every number gets a row **before** it goes in a draft. A number with no source does not go in
the paper.

The **Kind** column says what sort of number it is:
- **ours** — computed by this project
- **cited** — from someone else's paper
- **prior** — from our own v2/ICCIT paper
- **synthetic** — computed on made-up data, illustrative only

| ID | Number | Kind | Source | How it was made | Re-made by |
|---|---|---|---|---|---|
| N-001 | Cover drive strike rate 142.0 (50 balls) vs pull 112.0 (50 balls), batter vs pace in the powerplay | synthetic | v2 corpus, `cricshot-improve/cricketmind/corpus.py` | Counted, 2026-10-16 | Day 39 test |
| N-002 | Bootstrap 95% interval for that difference: [−16.0, +76.0] | synthetic | same | 20,000 resamples, seed 42, 2026-10-16 | Day 39 test |
| N-003 | Pull: 4 dismissals in 50 balls; P(out within 6 balls) = 0.394 (analytic); v2 Monte Carlo interval [0.387, 0.400] | synthetic | same; v2 `nodes/analytics.py` | 20,000 trials; interval from the Monte Carlo standard error only | Day 49 test |
| N-004 | Same with estimation uncertainty: median 0.405, 95% [0.155, 0.697] | synthetic | same | Jeffreys prior, Beta(4.5, 46.5); 20,000 draws, seed 42, 2026-10-16 | Day 49 test |
| N-005 | v2 worst case: 29 supersteps and 12 agent invocations | prior | ICCIT paper | closed-form cost model, measured | Day 79, Day 176 |
| N-006 | v2 routing: on 12 paraphrased prompts, the LLM router activated the intended agent 9 times, the keyword router 0 times | prior | ICCIT paper | 3 runs, identical | Day 159 (new set) |
| N-007 | CricBench: over 98% of LLM-generated SQL executes; under 29% is semantically correct | cited | arXiv:2512.21877, v3 abstract (Agarwal et al., Apr 2026, under review; v1 Devraj et al., Dec 2025) | Read 2026-10-19; some tables in the paper do not add up, so re-check the published version before citing | — |
| N-008 | Workload spikes ≈ 2× injury risk the next week (RR 2.1–2.2), 28 fast bowlers | cited | Hulin et al. 2014 (BJSM) | — | — |
| N-009 | Basketball summaries by three neural systems: 418 accuracy errors in 21 stories (about 20 each): 184 numbers, 105 named entities, 80 words, 19 context; Fleiss' κ 0.79 on error type | cited | Thomson and Reiter, INLG 2020, Sec. 4.3 (arXiv:2011.03992, HTML v1) | Read 2026-10-21 | — |
