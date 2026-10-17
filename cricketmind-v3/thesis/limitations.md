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
| L-012 | No user study | Usefulness to real stakeholders is untested | State as future work; claims stay about verification, not usefulness | 2026-10-16 |
| L-013 | The agent's own words around a slot are checked only for names (C6) and certainty words (C4), not for meaning | A correct, tagged number can still be framed misleadingly ("Kohli struggles with the cover drive: {c1}" when c1 is a strong 142.0) or placed where it reads oddly | Say plainly that v3 guarantees each number and its subject, not the tone of the prose; the renderer always prints the metric's own words and tag, so a number never appears unlabelled; count such cases in the Phase 10 real-LLM runs | 2026-10-17 |
