# Evidence map: every paper claim and its proof

CricketMind will not let an agent say something no engine computed. This file holds the paper
to the same rule. Each claim the paper will make gets a row. When the experiment runs, the
**Evidence** column points to the result file and the script that made it.

Status: **verified** (the evidence exists) · **planned** (a plan day will produce it) ·
**dropped** (the claim will not be made).

| ID | The paper will say… | Evidence | Status | Plan day |
|---|---|---|---|---|
| E-001 | v2's gate passes a correct figure attached to the wrong subject, and a figure left uncited | ICCIT paper, boundary experiment; v2 `nodes/verification.py` docstring | verified (prior work) | — |
| E-002 | Four v2 agents assert claims no engine computes (Stadium, Compliance, Injury, Sponsorship) | v2 templates in `cricshot-improve/cricketmind/agents.py` (N11f, N11i, N11k, N11g), read 2026-10-16 | verified | — |
| E-003 | v2's N9 interval reflects Monte Carlo error only | v2 `nodes/analytics.py`: `se = sqrt(p(1-p)/trials)` | verified | — |
| E-004 | On synthetic findings, the full gate (C1–C7, v4 design) catches all eight fabrication classes F1–F8; the v3 design (C1–C6) misses F8; v2 mode catches F1 only. Both correct findings pass all three gates | `results/fabrication_synthetic.csv` and its line in `results/ledger.jsonl`, made by `scripts/fabrication_synthetic.py`; checked by `backend/tests/unit/test_harness.py` | verified (synthetic, 2026-10-21) | 26 |
| E-005 | The full gate catches F1–F8 on real-data findings, and its false-rejection rate is X | `results/fabrication_real.csv`, `results/false_rejections.csv` | planned | 151, 155 |
| E-006 | The cost of containment: invocations and completeness as the fabrication rate rises | `results/containment_cost.csv` | planned | 156 |
| E-007 | N9 T1/T2 are calibrated on a held-out season (Brier score, reliability) | `results/calibration_*.csv`, `figures/reliability_*.pdf` | planned | 53, 175 |
| E-008 | The posture metrics separate front-foot from back-foot shots (AUC) | `results/posture_auc.csv` | planned | 124 |
| E-009 | The commentary bridge reaches κ = X against hand labels (or fails, and the fallback is used) | `results/bridge_agreement.csv` | planned | 135 |
| E-010 | The LLM router beats the keyword router on 100 labelled questions; stable over 3 runs | `results/routing.csv` | planned | 159 |
| E-011 | Measured cost matches the closed-form bound in every scenario | `results/cost_bounds.csv` | planned | 79, 176 |
| E-012 | Under v3, the v2 running example becomes "no detectable difference" | N-001, N-002; Day 39 test | planned (synthetic) | 39 |
| E-013 | On the locked 100-question final exam (written on Day 15; its SHA-256 goes here), CricketMind answers X% with checked facts, explains Y%, honestly declines Z%, gives W wrong answers, and gives the same checked facts for every wording of a question in V% of cases | `thesis/exam/final_exam.csv`; `results/coverage_final.csv` | planned | 15, 162 |
| E-014 | On the locked exam, a plain AI states wrong numbers in X% of answers and an AI given the engine's facts but no checks in Y%; CricketMind states no number its engines did not produce | `results/baselines.csv` | planned | 163–165 |
| E-015 | Human raters score the three systems' answers for trust and usefulness; the paper reports whether CricketMind is rated higher, and by how much (or that it is not) | `results/human_study.csv` | planned | 172–174 |
| E-016 | Each check C1–C7 is needed: turning any one off lets at least one fabrication class through | `results/ablation.csv` | planned | 151 |
| E-017 | On the 8-lie test, the Faith Check catches more lies than an AI judge and Chain-of-Verification, with fewer correct findings wrongly stopped (or the paper reports where it does not) | `results/ai_checkers.csv` | planned | 152–154 |
| E-018 | Whatever the language model (two local, one online), every number CricketMind shows was produced by its engines; models differ only in how often they are sent back or withheld | `results/three_models.csv` | planned | 166–167 |
| E-019 | Of the 20 tricky questions in the locked exam, X false premises are corrected with checked facts and Y trick attempts change nothing | `results/tricky.csv` | planned | 168 |
| E-020 | On CricBench's English questions, CricketMind's answers match the correct answers in X% of cases, beside the published text-to-SQL results (if the data is released) | `results/cricbench.csv` | planned | 169–171 |
| E-021 | The method works for football with zero changed lines in the checks: the 8-lie test and a locked 40-question football exam | `results/football.csv` | planned | 143–148 |
| E-022 | Every number in the final exam is re-checked from its receipt and matches | the receipt checker's log | planned | 47–48, 162 |
| E-023 | A claim type that is not on the menu, has no engine behind it, or has no words to show it is reported before the system starts, every problem at once | `backend/tests/unit/test_registry.py`, the `check_registry` tests (on hand-made agent and engine lists until Day 71 runs it on the real graph) | verified (unit test) | 11, 71 |
| E-024 | As of the search date, no system was found that gates probability wording against an external interval, or requires tone words to be backed by a checked comparison; both problems are documented (Cerda-Mardini et al. 2026; Thomson and Reiter 2020; Balani and Panda 2026) | `thesis/related_work.md`, Day 13 search log (S1–S16) and verdicts | verified at abstract level (full texts and a repeat search before submission) | 13 |
| E-025 | The load changes nothing: every ball of every file is stored, totals and extras add up, and every batter and bowler has a register id. The source itself disagrees with its own target or margin in 7 of 25,611 checks (by 1–3 runs, all associate matches), which the checks found and report | `results/data_checks.md` and its ledger line, made by `scripts/build_data.py`; the 7 re-added from the raw zip; every check tested on spoiled fixtures in `backend/tests/integration/test_quality.py` | verified (2026-10-21) | 32, 37 |
