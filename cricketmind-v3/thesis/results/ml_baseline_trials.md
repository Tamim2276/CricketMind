# Day 54: the ML baseline against the engine (D-062, pre-registration section 7)

Every test ball the engine answers: 487,381, in 2,110 matches; 1,000 paired resamples of whole matches (seed 42).

| model | Brier score | calibration gap (ECE) |
|---|---|---|
| B | 0.045882 | 0.00106 |
| logistic | 0.045729 | 0.00202 |
| trees | 0.045590 | 0.00151 |

The logistic regression ran 19 iterations; the trees, sized on the validation year, have 78 rounds.

## logistic against the engine (B)

- Brier(logistic) − Brier(B): 95% range -0.0001731 to -0.0001337
- ECE(logistic) − ECE(B): 95% range +0.000070 to +0.001358
- **A baseline:** it does not take the engine's place.

## trees against the engine (B)

- Brier(trees) − Brier(B): 95% range -0.0003222 to -0.0002633
- ECE(trees) − ECE(B): 95% range -0.000492 to +0.000970
- Better by more than luck, and calibration not clearly worse: it may take the engine's place only once it gives an honest range (20 refits on resampled training matches), the rule's third part.

