# Day 53 pre-registration: calibration, the two trials, and the ML baseline's rule

Written 2026-10-26, before any of it is run. Its SHA-256 goes in the journal before the run.
Nothing here is changed after the run; anything learned goes in a new decision.

## 1. The split

- Data: the D-032 snapshot, last match 2026-10-14.
- **Test year:** every match after 2025-10-14 (the newest 365 days). **Train:** every match
  on or before 2025-10-14. A **validation year** for later days (early stopping, temperature
  scaling): 2024-10-15 to 2025-10-14, inside the train side.
- **Kinds of cricket:** men's T20, women's T20, men's ODIs, men's Tests ("T20" is T20 and
  IT20, D-047). Each is fitted on its own train matches and reported on its own; the
  trials are decided on the four pooled.

## 2. What is predicted

- **Per ball (the main test):** every ball a batter faced in the test year, and every ball he
  was out on (a stumping off a wide). The chance he is out on it is made only from the train
  matches of its kind: his record there, and a prior fitted once per kind (and part of the
  order, and stage) on every batter's train record. The engine leaves the asked batter out
  of his own prior; here he stays in, a negligible difference, checked on a sample against
  the engine itself (section 6).
- **Out within k = 6 (the engine's question):** from each test ball, his next 6 balls faced,
  or as many as he faced before the innings ended or he retired; the chance for that many
  balls, and whether he was out within them. No window is left out, so the ones cut short
  do not bias the rate.
- A batter with no train balls gets no prediction, as the engine gives none; counted.

## 3. The models

| | Model | Its chance |
|---|---|---|
| D | his record alone (Day 49) | Beta(0.5 + outs, 0.5 + balls not out on) |
| A | one risk, prior from every batter (Day 50's first version, fitted by likelihood) | shrunk toward every batter |
| B | one risk, prior from batters in his part of the order (D-059; the engine now) | shrunk toward his part of the order |
| C | B by stage: balls 1-5, 6-15, 16-30, after 30 (D-056, on trial) | shrunk toward his part of the order in that stage |
| E | T2: the batter against this ball's bowler (D-061) | the odds rule and the meetings, on B's groups |

Each model's chance is its posterior mean: the chance it gives the event. The engine shows
the middle of its replays; that is checked too, for B, with the Beta median
(a − 1/3)/(a + b − 2/3) in place of 20,000 replays (Kerman, 2011).

## 4. The scores

- **Brier score:** the mean of (chance − outcome)², lower is better. Log loss is reported too.
- **Calibration:** 10 bins of equal size by predicted chance; in each, the mean chance and
  the share out. ECE is the size-weighted mean gap between them. The reliability plot draws
  the bins against the diagonal.
- **Uncertainty:** 1,000 bootstrap resamples of whole matches (balls of one match go
  together), seed 42, paired: the same resampled matches for every model.

## 5. The rules

1. **Batting place (D-059):** B is kept over A only if B's Brier score is lower and the 95%
   range of Brier(B) − Brier(A) lies wholly below zero, and B's calibration is not clearly
   worse: the 95% range of ECE(B) − ECE(A) does not lie wholly above zero. Otherwise the
   engine goes back to A.
2. **Stages (D-056):** C stays only if, against the winner of rule 1, the same holds. Otherwise
   the stages leave the engine (the menu's `faced_so_far` goes) and stay only as a baseline
   for Day 157.
3. **"Calibrated":** the paper calls a model calibrated on a kind of cricket only if its ECE
   is at most one tenth of its average chance. If not, it says so, and any recalibration is a
   later decision, never fitted on the test year.
4. **T2 (E)** answers another question (this bowler), so it has no rule here: its scores are
   reported beside B's on the same balls.

## 6. The checks on the run itself

- 30 test moments in men's T20, drawn with seed 42 from batters with a name nobody shares:
  the engine itself (with `before`, `faced_so_far` and k = 6) against the bulk computation;
  the gap should be under one percentage point (the batter in his own prior, and replays).
- Counts: balls, windows, batters with no prediction, windows cut short.

## 7. The ML baseline (D-062), fitted on Day 54, not today

- **Split:** the same; the validation year is used only to stop the boosted trees early.
- **Outcome:** out on the ball, per ball, as in section 2.
- **Inputs**, known before the ball, from train matches or the ball's own match so far: the
  kind of cricket; the batter's train record (outs, balls) and his B chance; his part of the
  order; his balls faced so far in the innings; the bowler's train record and his shrunk
  rate; the bowler's type (pace or spin, if known); the phase; the innings; the wickets down
  before the ball; in a chase, the runs needed and the balls left; otherwise, the runs so far
  and (limited overs) the balls left. Nothing about line, length, speed, field or pitch.
- **Models:** logistic regression (scikit-learn, inputs standardised, default settings) and
  gradient-boosted trees (HistGradientBoostingClassifier, default settings, early stopping
  on the validation year, seed 42).
- **Rule:** an ML model takes the place of N9's per-ball chance only if its Brier score is
  lower than the best N9 model's with the 95% paired range of the difference wholly below
  zero, its calibration is not clearly worse (as in rule 1), and it can give an honest range:
  refitted on 20 bootstrap resamples of the training matches (seeded), so its range carries
  the doubt of its fit. Otherwise it is reported as a baseline.
