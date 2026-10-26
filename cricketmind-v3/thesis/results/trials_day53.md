# Day 53: the two trials, decided by the pre-registered rules

Every test ball of the four kinds that every model answers: 483,481; in 2,110 matches, resampled 1,000 times (seed 42).

| model | Brier score | calibration gap (ECE) |
|---|---|---|
| D | 0.047443 | 0.00999 |
| A | 0.045677 | 0.00166 |
| B | 0.045655 | 0.00103 |
| C | 0.045647 | 0.00149 |

## 1. Batting place (D-059): B against A

- Brier(B) − Brier(A): 95% range -0.0000317 to -0.0000125
- ECE(B) − ECE(A): 95% range -0.001085 to +0.000182
- **Kept:** B replaces A.

## 2. Stages (D-056): C against B

- Brier(C) − Brier(B): 95% range -0.0000180 to +0.0000013
- ECE(C) − ECE(B): 95% range -0.000248 to +0.000967
- **Not kept:** B stays.

