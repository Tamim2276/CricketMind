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
| E-004 | The v3 gate catches all seven fabrication classes F1–F7 on synthetic findings; v2 mode catches F1 only | `results/fabrication_synthetic.csv` | planned | 25 |
| E-005 | The v3 gate catches F1–F7 on real-data findings, and its false-rejection rate is X | `results/fabrication_real.csv`, `results/false_rejections.csv` | planned | 144–145 |
| E-006 | The cost of containment: invocations and completeness as the fabrication rate rises | `results/containment_cost.csv` | planned | 146 |
| E-007 | N9 T1/T2 are calibrated on a held-out season (Brier score, reliability) | `results/calibration_*.csv`, `figures/reliability_*.pdf` | planned | 51, 150 |
| E-008 | The posture metrics separate front-foot from back-foot shots (AUC) | `results/posture_auc.csv` | planned | 124 |
| E-009 | The commentary bridge reaches κ = X against hand labels (or fails, and the fallback is used) | `results/bridge_agreement.csv` | planned | 135 |
| E-010 | The LLM router beats the keyword router on 100 labelled questions; stable over 3 runs | `results/routing.csv` | planned | 149 |
| E-011 | Measured cost matches the closed-form bound in every scenario | `results/cost_bounds.csv` | planned | 74, 151 |
| E-012 | Under v3, the v2 running example becomes "no detectable difference" | N-001, N-002; Day 40 test | planned (synthetic) | 40 |
