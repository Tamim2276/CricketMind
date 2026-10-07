# Implementation journal

One short entry per working day: **what I tried, what broke, why, and what fixed
it.** This file is the raw material for the Implementation chapter — a thesis
chapter is mostly a record of failures that were diagnosed, and that record is
impossible to reconstruct months later.

## How to use it

- **Append at the bottom.** Never reorder or rewrite earlier entries. Appending
  cannot lose what is already here, which matters when the power can cut at any
  moment.
- **Write it during the outage window.** It is the one task that needs no GPU.
- **Five minutes, not thirty.** A thin entry today beats a perfect entry never.
- **Commit it daily.** `git add notes/journal.md && git commit -m "journal: DD Mon"`
  — that is the second durable copy.

## The rule that makes it usable later

> Every claim carries a number or a file path.

"Accuracy was bad" is worthless in six months. "val 75.93% but test 23.67%,
`experiments/transformer/checkpoints/report_test.json`" can be written up, and
an examiner can follow it. Record what you **ruled out**, not only what you
concluded — that is the part of the reasoning nobody can reconstruct from code,
and the part a viva actually probes.

## Template

```markdown
## YYYY-MM-DD — <four words>

**Goal.**
**Broke.**
**Cause.** (and how I know — what I ruled out)
**Fixed.** (file, and the evidence it worked)
**Open.**
```

---

## 2026-08-27 — backfilled: the 75% ceiling

Backfilled on 2026-10-08 from `history.json` and `report_test.json`, before the
detail was lost. Entries from here on are written the same day.

**Goal.** Beat the CricShot10k authors' 89.09% top-1 by trying better
architectures than their EfficientNetV2-S + GRU.

**Broke.** Five different architectures all stalled in the same narrow band.
Train accuracy was 97.6–99.9% in every single run:

| run | epochs | best val | test top-1 | final train |
|---|---|---|---|---|
| `baseline` | 60 | 75.45% | 74.93% | 99.88% |
| `combined` | 29 | 75.58% | 73.94% | 97.58% |
| `mimic_author` | 22 | **76.75%** | 74.54% | 98.41% |
| `multiscale` | 27 | 74.62% | 71.29% | 98.84% |
| `transformer` | 44 | 75.93% | 23.67% ⚠ | 99.16% |

**Cause.** Not the architecture — that was the wrong thing to vary. Three pieces
of evidence, which only line up one way:

1. **Train accuracy is ~99% while val sits at ~75%.** The models have ample
   capacity and are fitting the training set almost perfectly. Changing the
   architecture cannot fix a gap of that shape.
2. **`data/processed/` and `data/splits/` do not exist.** Every one of these
   runs read the raw 896×540 clips directly — no striker crop, no segmentation.
3. **The authors' own weights file is named**
   `Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras`. They named
   the precondition in the filename.

So the 14-point gap is the preprocessing, not the model. It matches the authors'
own ablation row for no-crop / no-segmentation / no-augmentation, **75.50%**,
to within a point — and `baseline` scored 75.45%. Five experiments were spent
re-measuring the authors' worst ablation.

**Fixed.** Nothing yet; this is the premise of the 20-day plan. Phase 1 builds
the crop → sample → segment pipeline first and only then compares models.

**Open.** The `transformer` test figure, below.

---

## 2026-10-07 — training survives a power cut

**Goal.** Make every run in the 20-day plan resumable, before starting 30+
experiments rather than after losing one.

**Broke.** Nothing yet — this was the audit. Three real defects in `src/train.py`:
`best.pt` held only `model_state`, so a cut at epoch 40/60 meant restarting from
epoch 1 with no optimizer moments or scheduler position; `history.json` was
written once at the very end, so a cut at epoch 50 lost every training curve;
and `torch.save` writes in place, so a cut *during* the write truncates the best
checkpoint already earned.

**Cause.** The loop was written assuming it would run to completion.

**Fixed.** New [`src/utils/checkpoint.py`](../src/utils/checkpoint.py):
`atomic_save` writes to `<path>.tmp` then `os.replace` (atomic on NTFS);
`save_resume`/`load_resume` carry model + optimizer + scheduler + scaler +
history + RNG; `append_result` keeps an fsynced `experiments/results.jsonl`.
Wired into `train.py`, where **resuming is the default** — re-run the identical
command and it continues; `--fresh` forces a restart.

Evidence: [`tests/test_checkpoint.py`](../tests/test_checkpoint.py) trains a
model 6 epochs straight, then trains it 3 epochs → saves → discards the objects
→ reloads → trains 3 more, and the weights match **bit-for-bit**
(`rtol=0, atol=0`). A second test proves that test can fail: resuming from
weights alone diverges. 65 tests pass.

Writing the tests found a bug the audit had missed: a cut mid-`append` leaves a
line with no trailing newline, so the *next* result was glued onto the damaged
one — losing a good number as well as a bad one. `append_result` now terminates
the file before appending.

**Open.** The wiring is only verified to parse — `data/processed/` does not
exist, so no real training run has exercised the resume path. First real test is
the Day 5 smoke run.

---

## 2026-10-08 — the 23.67% is a stale artifact

**Goal.** Explain why `transformer` scored 75.93% on val but 23.67% on test,
when the other four runs lose only 0.5–3.3 points from val to test.

**Broke.** A 52.3-point val→test collapse in exactly one of five runs.

**Cause.** The predictions do not come from the checkpoint now on disk.

What I **ruled out** first, because it was the obvious suspect — a class-order
mismatch between the training and test datasets (e.g. `os.listdir` order against
`sorted` order). `preds_test.npz` stores logits, predictions and labels, so this
is decidable rather than arguable. I built the 15×15 prediction/label matrix and
solved for the best possible one-to-one relabelling of predicted classes:

```
raw accuracy           : 23.67%
best permutation acc   : 23.67%     <- no relabelling helps at all
preds == argmax logits : True
```

A permutation bug would be repaired by the optimal permutation. This one is not,
so the label mapping is not the fault. The test set itself is also fine — true
labels are spread 36–182 across all 15 classes, 1508 clips.

What the predictions actually look like:

```
true: [117  90 125 135  63  43 158 182 128  44  49 161  51 126  36]
pred: [ 20  62 265  74   6   0 395  88  39   0   0 301   8 250   0]
```

The model never once predicts 4 of the 15 classes, and dumps 61% of its
predictions onto four classes. That is a broken or half-initialised model, not a
mislabelled good one.

Then the decisive evidence, from file timestamps:

| file | written |
|---|---|
| `preds_test.npz`, `report_test.json`, `confusion_matrix_test.png` | Aug 26, **15:09** |
| `best.pt` | Aug 26, **21:52** |
| `history.json` | Aug 27, **01:18** |

The evaluation ran **6h43m before** the current `best.pt` was written. These
artifacts belong to an earlier, superseded training attempt; `best.pt` was then
overwritten by a later run. No epoch in `history.json` has val anywhere near
23.67% (epoch 1 = 10.1%, epoch 2 = 28.7%), which is consistent with the
predictions coming from a model that is simply not in this directory any more.

**So the current `transformer` checkpoint has never been evaluated at all.** The
23.67% is not a result; it is debris.

**Fixed.** Not yet — the re-evaluation needs `data/processed/`, which Phase 1
builds. Two actions recorded for Day 1.3:

1. Re-run `evaluate.py` against the current `best.pt` and overwrite the stale
   artifacts.
2. Make `evaluate.py` write the checkpoint's SHA-256 and mtime into
   `report_*.json`, and refuse to overwrite a report whose checkpoint hash does
   not match. A stale result that looks like a real one cost a wrong diagnosis
   here; it should not be able to recur silently.

**Open.** Both of the above. Note for the write-up: do not report 23.67%
anywhere — it measures a model that no longer exists.
