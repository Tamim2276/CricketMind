# Day 54: the trees' honest range (the rule's third part)

20 refits on resampled training matches (seed 42); for every test ball the engine answers, the middle 95% of the refits' chances, beside the engine's own 95% range for one ball (its Beta curve). Mean widths, in percentage points, by how many train balls the batter had:

| his train balls | test balls | the trees' range | the engine's range |
|---|---|---|---|
| under 100 | 67,064 | 1.405 | 4.522 |
| 100 to 999 | 198,688 | 0.985 | 3.008 |
| 1,000 or more | 221,629 | 1.498 | 1.358 |

- The rule's third part is met: every test ball the engine answers has a range from the 20 refits.
- The two ranges measure different doubts. The refits redraw the training matches but keep each batter's own record as it is, so the trees' range is the doubt of the fit; the engine's is the doubt of his own rate, which is why it is wider for a batter with few balls.

