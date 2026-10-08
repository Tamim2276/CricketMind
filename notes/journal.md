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

---

## 2026-10-08 — the dataset is not what the paper says

**Goal.** Decide whether the CricShot10k copy on disk is usable or has to be
downloaded again, before building a pipeline on top of it.

**Broke.** Nothing was broken — but a 45-clip sample nearly caused a real bug.
Three clips per class all came back 896×540, 24 frames, 30 fps, exactly as the
paper states, and on that basis `sample.py` would have been written as a fixed
`24 → 15`.

**Cause.** The sample was too small and not drawn across source matches. Decoding
**every frame of all 10,091 clips** gives a different picture:

| frames | clips | matches |
|---|---|---|
| 22 | 57 | 2 |
| 23 | 6 | 1 |
| **24** | **7,657** | **442** |
| 28 | 1,390 | 45 |
| 29 | 25 | 1 |
| **49** | **956** | **43** |

Only **75.9%** of clips have the 24 frames the paper claims. The structure
behind it is clean: **all 534 source matches have one consistent frame count,
zero matches mix.** So frame count is a property of the broadcast each clip was
cut from, not of the clip, and it is spread across all 15 classes roughly in
proportion — which means clip length is *not* a shortcut to the label.

Also found: 27 clips are off-resolution (`vid121` × 25 at 598×360, `vid22` × 2
at 896×494), and **529 of 534 matches span more than one shot class**, which
independently confirms that grouped splitting is mandatory rather than merely
prudent.

**Fixed.** No re-download — 0 files fail to open, 0 zero-byte, 0 frames fail to
decode, 3.94 GB total, and the original `.rar` is still at
`data/unnecessary/raw/` as a fallback. Two plan steps rewritten before any code
was written against the wrong assumption: Day 2.3 (`video.py` must return the
real length) and Day 4.1 (`sample.py` is `T → 15` for any `T` in 22–49, step
size taken from the clip). Recorded as section 5 of the plan.

**Open.** Nothing. But note for the write-up: the paper's "24 frames" claim is
wrong for the released data, and that is worth a sentence — it is a small,
checkable correction to a published dataset description.

**What this cost to find:** about four minutes of decoding. A hardcoded `24 → 15`
would have silently mangled a quarter of the dataset, still run without error,
and shown up only as a couple of accuracy points I could never have explained.

---

## 2026-10-08 (later) — an XPU flake I could not reproduce

**Goal.** Build `src/utils/seed.py` (step 1.3) and test it.

**Broke.** On the first run of `tests/test_seed.py`, one test failed:

```
test_model_init_is_reproducible_on_the_gpu
>   assert torch.equal(a, b)
E   RuntimeError: bad allocation
```

Two 64-element tensors on `xpu:0`.

**Cause.** Unknown, and I want that written down rather than guessed at. What I
ruled out:

- Not logic: the same three lines pass outside pytest.
- Not test pollution: the test passes alone, and passes when paired with each
  of the five tests that precede it, one at a time.
- Not persistent: five consecutive full-file runs afterwards, all 10 tests pass.

"bad allocation" is a `std::bad_alloc` surfacing from the Level Zero / SYCL
runtime, so the likely source is the Intel driver, not torch and not this code.

**Fixed.** Nothing. There is nothing to fix yet, and wrapping the test in a
retry would only hide a recurrence.

**Open.** Watch for it. If an Arc allocation can fail at random, then a long
training run can die at random too — which makes the resume machinery from
2026-10-07 more valuable than I thought when I built it, and means a dead run
is not automatically a bug in my code. If it recurs, note the run and whether
anything else was using the GPU at the time.

---

## 2026-10-08 (later still) — the authors split at clip level

**Goal.** Answer a question I should have asked before writing the plan: did
Dihan et al. group their splits by source match?

**Broke.** The plan's Day 7 target. It said "reproduce 89.09%" and said to
group strictly by `vidNNN`. Those two instructions are not compatible, and I had
not noticed.

**Cause.** They split at **clip level**. Their whole protocol:

> "The test sets were manually constructed, comprising 20% of the total videos.
> Care was taken to ensure that similar types of shots from the same batter do
> not overlap between the training and test sets. From the remaining videos,
> 20% were taken randomly for the validation test."

"Videos" means clips — the paper says "10,086 videos" for the dataset and
"536 **match** videos" when it means matches. So the constraint is
batter-plus-shot-type, and validation has no constraint at all.

What that leaves open: same batter different shot type, same match different
batter, same ground / lighting / camera / broadcast graphics. Measured on our
copy: under a clip-level 20% test split, **489 of 534 matches (91.6%)** are
expected to have clips on both sides. A 19-clip match straddles with p=0.986,
and the mean match has 18.9 clips.

To be fair to them, they are careful about leakage elsewhere — they insist on
augmenting only after splitting, and they criticise earlier work for train/test
overlap possibly inflating a reported 93%. They controlled the sharpest case and
hand-built the test set rather than randomising it. They just did not reach
match-level grouping.

**Fixed.** The plan now builds **two** split sets and runs all 44 experiments on
both, 88 rows. `splits_author/` reproduces their protocol and is what the
89.09% comparison is made against; `splits_grouped/` is the honest one. The
reason for keeping both: if I grouped strictly and landed on 84%, "my pipeline
is broken" and "their number is inflated" would be indistinguishable, and
separating them would cost days.

Running all 44 twice is affordable because **a frozen backbone's features depend
only on the clip, not on the split** — one cache serves both. The 28 cached
experiments cost 1.7 extra hours for 28 extra rows; the 16 end-to-end ones cost
60. That also caught a real design bug: the cache was specified as
`features/<split>/<class>/<clip>.pt`, which would have forced a full
re-extraction on every split change. It is flat now.

Phase 3 becomes 123 GPU-hours and the plan is about 31 days; the file is renamed
`CricShot10k_Plan.md` since the "20 day" title was no longer true.

**Open.** Nothing blocking. The interesting question is now empirical: **does the
leak gap depend on the architecture?** The 28 cached runs answer that before any
of the 60 expensive hours are spent. If the gap is stable it is a property of
the data; if it varies by architecture, that is the more interesting paper.

**Note for the write-up:** the dataset has no batter labels, so `splits_author/`
uses `(vidNNN, class)` as the closest available proxy for "same batter, same
shot type". Say that plainly rather than implying batter labels existed.

---

## 2026-10-09 — video.py, and a second flake I cannot reproduce

**Goal.** Step 2.3: one function that turns an .avi into numbers, correctly,
whatever its length.

**Broke.** Two things, neither reproducible.

1. In a full-suite run, `read_clip` on `vid40_17.avi` returned **"opened but
   decoded 0 frames"** — a clip that reads fine every other time. The same run
   also produced a hard native crash dump.
2. `test_seed`'s XPU test failed again with **"bad allocation"**, the same flake
   as 2026-10-08.

**Cause.** Unknown, and I could not pin either one down. What I ruled out for
the video failure:

- Not the code: the clip reads correctly standalone, and in all four pairwise
  runs with other test files.
- Not GPU contention via hardware decode: OpenCV's default backend here is
  FFMPEG with `CAP_PROP_HW_ACCELERATION = 0.0`, i.e. software decoding, and
  60 reads under continuous Arc load gave 0 failures.
- Not persistent: four consecutive full-suite runs afterwards, all 136 pass.

Both flakes involve the same machine under load and both vanished. I suspect
they are the same underlying problem rather than two, but I have no evidence
for that beyond co-occurrence, so it stays a suspicion.

**Fixed.** Nothing. I added a one-retry guard to `read_clip` so that Day 4's
10,091-clip run could not condemn a good clip over a transient hiccup, but the
file was overwritten on disk shortly after — most likely a stale editor buffer
saving over it — and I left it off rather than re-applying it unilaterally. The
tests were matched to what is actually on disk.

**Open.** Decide whether `read_clip` gets the retry before Day 4. The argument
for: one bad read in 10,091 silently drops a clip from the dataset, and we have
seen exactly that failure once. The argument against: it is complexity bought
against a cause nobody has established, and a retry can hide a real decoder
problem. If it goes in, the retry count must be printed at the end of the
preprocessing run, not swallowed.
