# The engine report (M2, Day 54)

Every engine number below was asked of the real engines on the D-032 snapshot, and
carries a receipt (its line in `engine_report_receipts.jsonl`); every receipt was
checked again on the same data before this report was written.

## 1. M1 against published records (E-026)

Retired players whose every match in a format is in the data (the matches column is a count of the data, not an engine claim).

| player | format | matches (data / published) | runs | dismissals | average | strike rate | receipts |
|---|---|---|---|---|---|---|---|
| AN Cook | Test | 161 / 161 | 12,472 / 12,472 | 275 / 275 | 45.35 / 45.35 | 46.95 | 1, 2, 3 |
| AN Cook | ODI | 92 / 92 | 3,204 / 3,204 | 87 / 88 | 36.83 / 36.41 | 77.26 / 77.13 | 4, 5, 6 |
| DA Warner | Test | 112 / 112 | 8,786 / 8,786 | 197 / 197 | 44.60 / 44.59 | 70.19 | 7, 8, 9 |
| SPD Smith | ODI | 170 / 170 | 5,668 / 5,800 | 132 / 134 | 42.94 / 43.28 | 86.91 / 86.96 | 10, 11, 12 |
| V Kohli | Test | 123 / 123 | 9,230 / 9,230 | 197 / 197 | 46.85 / 46.85 | 55.58 / 55.57 | 13, 14, 15 |

## 2. The engines' answers to the questions of Days 39 to 52

Asked again with today's engines; where an engine changed since (D-058 to D-066), the number may differ from its first report.

| first reported | question | engine | answer | n | range | receipt |
|---|---|---|---|---|---|---|
| N-016 | Kohli against Bumrah, against everyone else's | N8.M1 | 35.8 | 108 | 4.174 to 69.89 | 16 |
| N-020 | Kohli's T20I strike rate | N8.M1 | 136 | 2915 |  | 17 |
| N-020 | Kohli in T20Is, against every other man | N8.M1 | 18.8 | 2915 | 13.35 to 24.31 | 18 |
| N-020 | Kohli against Australia since 2020, all formats | N8.M1 | 69.62 | 2350 |  | 19 |
| N-020 | Kohli against Australia since 2020, Tests | N8.M1 | 49.22 | 1276 |  | 20 |
| N-021 | Bumrah's Test wickets | N8.M1 | 234 | 10031 |  | 21 |
| N-021 | Bumrah's Test economy | N8.M1 | 2.77 | 10031 |  | 22 |
| N-021 | Bumrah's Test strike rate | N8.M1 | 42.87 | 10031 |  | 23 |
| N-021 | Anderson's Test wickets of Kohli | N8.M1 | 7 | 710 |  | 24 |
| N-021 | Kohli's Test average against Anderson | N8.M1 | 43.57 | 710 |  | 25 |
| N-017 | Mirpur, men's T20, first-innings average | N8.M2 | 150.4 | 329 |  | 26 |
| N-017 | Mirpur, men's T20, chasing sides' win share | N8.M2 | 0.5529 | 331 |  | 27 |
| N-017 | Mirpur, men's T20, spin's share of the wickets | N8.M2 | 0.4054 | 3890 |  | 28 |
| N-018 | Bumrah's workload ratio, 31 December 2024 | N8.M3 | 1.912 | 684 |  | 29 |
| N-018 | Bumrah's workload ratio, 4 January 2025 | N8.M3 | 1.316 | 626 |  | 30 |
| N-019 | India's men: reviews overturned | N8.M4 | 0.2728 | 1063 |  | 31 |
| N-019 | Bangladesh's men: reviews overturned | N8.M4 | 0.2772 | 772 |  | 32 |
| N-019 | England's men: reviews overturned | N8.M4 | 0.285 | 1505 |  | 33 |
| N-019 | India's women: reviews overturned | N8.M4 | 0.2881 | 243 |  | 34 |
| N-022 | Pakistan beat India, 24 October 2021, by wickets | N8.M5 | 10 | 1 |  | 35 |
| N-022 | Bangladesh beat India, 3 November 2019, by wickets | N8.M5 | 7 | 1 |  | 36 |
| N-022 | Bangladesh's T20I win share against India | N8.M5 | 0.05556 | 18 |  | 37 |
| N-022 | Bangladesh's ODI win share at Mirpur | N8.M5 | 0.4898 | 98 |  | 38 |
| N-026 | Kohli out within 6 balls, T20 powerplay against pace | N9.T1 | 0.1794 | 2855 | 0.1503 to 0.2111 | 39 |
| N-026 | Tamim Iqbal out within 6 balls, T20 powerplay against spin | N9.T1 | 0.1863 | 1001 | 0.1415 to 0.2379 | 40 |
| N-026 | Bumrah batting, out within 6 balls, T20 | N9.T1 | 0.4596 | 107 | 0.3611 to 0.5592 | 41 |
| N-026 | Kohli out within 30 balls, Tests | N9.T1 | 0.3057 | 16608 | 0.2723 to 0.3407 | 42 |
| N-031 | Kohli: Bumrah instead of Shami, next 6 balls, T20 | N9.T2 | 0.0001428 | 2182 | -0.02527 to 0.02509 | 43 |
| N-031 | Smith: Anderson instead of Root, next 30 balls, Tests | N9.T2 | 0.05015 | 3216 | -0.01903 to 0.1169 | 44 |

## 3. Ranges that end within one replay step of zero (L-034)

Every man with at least 1,000 balls faced in T20Is whose name nobody else shares: his strike rate against every other man's, 150 ranges. 44 include zero (no detectable difference). **1 end within one step of zero**, where another seed could move the verdict:

- K Motlanka (-16.53 to +0.08, step 0.088; receipt 111)

## 4. Calibration and the ML baseline (Days 53 and 54)

Day 53's calibration, per ball, all four kinds (`calibration_day53.csv`, its ledger line):

| model | Brier score | calibration gap |
|---|---|---|
| D | 0.047443 | 0.00999 |
| A | 0.045677 | 0.00166 |
| B | 0.045655 | 0.00103 |
| C | 0.045647 | 0.00149 |
| B, on the balls E answers | 0.046232 | 0.00097 |
| E, on the balls E answers | 0.046235 | 0.00114 |

Day 54's ML baseline, per ball, all four kinds (`ml_baseline.csv`, its ledger line):

| model | Brier score | calibration gap |
|---|---|---|
| B | 0.045882 | 0.00106 |
| logistic | 0.045729 | 0.00202 |
| trees | 0.04559 | 0.00151 |

## The receipts

194 receipts written; checked again on the same data: **194 confirmed, 0 not**.

