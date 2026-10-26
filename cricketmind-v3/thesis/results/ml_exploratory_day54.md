# Day 54, EXPLORATORY (not pre-registered): the trees with only the question

The same trees as section 7, given only what a question to the engine can carry: batter_outs, batter_balls, batter_rate, batter_log_balls, kind=, batting_order=, bowler_type=, phase=. On the 487,381 test balls the engine answers:

| model | Brier score | calibration gap (ECE) |
|---|---|---|
| B | 0.045882 | 0.00106 |
| question only | 0.045752 | 0.00166 |

- Brier(question only) − Brier(B): 95% range -0.0001472 to -0.0001125
- ECE(question only) − ECE(B): 95% range -0.000364 to +0.001114
- The trees have 48 rounds. This decides nothing; it says where the pre-registered trees' gain comes from.

