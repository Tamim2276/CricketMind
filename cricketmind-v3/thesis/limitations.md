# Limitations and threats to validity

Write a weakness down the day you notice it. A reviewer will find it anyway, and the paper is
stronger when it names it first.

| ID | Limitation | Why it matters | What the paper should say or do | Noticed |
|---|---|---|---|---|
| L-001 | All design-time examples use the v2 synthetic corpus | Says nothing about real players | Label as illustrative; real-data results come from Phase 10 | 2026-10-16 |
| L-002 | The probability bands come from Kent (1964), written for intelligence analysts | Cricket audiences may read words differently | Cite; call the bands a design choice; future work: a reader study | 2026-10-16 |
| L-003 | The workload ratio's validity is disputed (Impellizzeri et al. 2021), and match data omits training loads | Medical claims could mislead | The workload output is a monitoring flag, never an injury probability | 2026-10-16 |
| L-004 | A licensed commentary corpus with match and ball keys may not exist | The shot bridge may be impossible | Go/no-go gate; fallback evaluation set; report a negative result honestly | 2026-10-16 |
| L-005 | Weak shot labels carry noise, and joining them to deliveries is error-prone (wides, no-balls) | Shot statistics inherit the errors | Report κ, per-class F1 and join rate; carry label confidence | 2026-10-16 |
| L-006 | Pose estimation may fail on low-resolution clips; posture has no ground truth | Posture claims could be wrong | Validate against a proxy (front vs back foot); drop metrics below AUC 0.75 | 2026-10-16 |
| L-007 | Bowling type is missing for some players | Filters such as "vs pace" silently shrink n | Always report n; evidence floor C3; report coverage | 2026-10-16 |
| L-008 | A 7B local model may break the JSON schema | Findings fail to parse | Constrained decoding; count parse failures as C1 rejections and report them | 2026-10-16 |
| L-009 | The routing questions and their labels are written by the author | Optimistic bias | Include paraphrases; ideally a second labeller; say it plainly | 2026-10-16 |
| L-010 | Thresholds (n_min, κ, AUC, caps) are design parameters, not tuned values | Results depend on them | Name them as such; a sensitivity check if time allows | 2026-10-16 |
| L-011 | Measurements are made on one machine | Timings may not transfer | Report hardware; prefer counts (supersteps, calls) over seconds | 2026-10-16 |
| L-012 | The human study is small (about 10 raters) and not a field deployment | Usefulness to real stakeholders is only lightly tested | Report it as a small study with its numbers; claims stay mainly about verification; a field study is future work | 2026-10-16 (narrowed 2026-10-19 by D-019) |
| L-013 | Tone is checked only through C7's word list; words not on the list, sarcasm and subtle framing are not | A correct, tagged number can still be framed misleadingly | Say plainly that v3 guarantees numbers, subjects, certainty words and listed tone words, not every nuance; numbers always carry their own label; count the misses in the Phase 10 runs | 2026-10-17 (narrowed by D-017) |
| L-014 | Rules explanations (the Explainer) are not checked against any source | An explanation can be wrong | Label every explanation "not checked against data"; no digits or player names in it; a checked rules source (the MCC Laws) is future work | 2026-10-17 |
| L-015 | N3 can misread a question, especially an unusual wording | A correct answer to the wrong question | Every answer starts "I read this as: …"; measured on the coverage set, with 5 wordings per question | 2026-10-17 |
