# CricShot10k — 20-Day Plan

**Goal.** First reproduce what Dihan et al. achieved (89.09% top-1). Then run 30+
different approaches and record every result. Then combine and tune the best ones.

**How we work.** You are learning this while building it, so nothing is built
ahead of you and nothing is left in jargon. Every step follows the same
five-part loop, set out in [section 1](#1-how-every-step-works). Short version:
**one small piece → run it → I explain it in plain words → you say "next"**.
Every day ends with something finished, a number in the ledger, and a line in
the journal.

**Out of scope for these 20 days.** The LangGraph / CricketMind work is paused.
Nothing in this plan touches `cricketmind/`, `notebooks/` or `paper/`.

---

## 0. Read this first: why you were stuck at 75%

Five different architectures were trained and all five landed in the same
two-point band:

| run | best val | test |
|---|---|---|
| baseline | 75.45% | 74.93% |
| mimic_author | 76.75% | 74.54% |
| transformer | 75.93% | 23.67% ← broken evaluation, see below |
| combined | 75.58% | 73.94% |
| multiscale | 74.62% | 71.29% |

When a GRU, a transformer, a multi-scale CNN and a hybrid all plateau within two
points of each other, **the model is not the bottleneck — the input is.**

Now compare against the authors' own ablation table:

| Cropping | Segmentation | Augmentation | Accuracy |
|---|---|---|---|
| ✗ | ✗ | ✓ | **75.50%** ← you are here |
| ✓ | ✗ | ✓ | 85.76% |
| ✓ | ✓ | ✗ | 87.60% |
| ✓ | ✓ | ✓ | **89.09%** ← the target |

Your 74.9% is the "no cropping, no segmentation" row, almost to the decimal.

Three more pieces of evidence:

1. The raw clips are **896×540 full broadcast frames**, 24 frames each. The
   striker occupies a small part of that frame. The model is spending its
   capacity on crowd, sponsor boards and grass.
2. `data/processed/` and `data/splits/` **do not exist**. Both configs point at
   them. Whatever the old runs trained on, it was not the output of
   `src/preprocess/build_dataset.py`.
3. The authors named their own weights file:
   **`Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras`**.
   They are telling you, in the filename, that the model only works on cropped
   and segmented input.

**Conclusion: the preprocessing pipeline was never actually applied.** That is
the whole 14-point gap, and it is Phase 1 of this plan.

### The second bug — diagnosed 2026-10-08

`transformer` scored **75.93% val but 23.67% test**. A model does not lose 52
points between validation and test, so this is an evaluation problem, not a
training problem. It has now been diagnosed, and the first guess was wrong.

**It is not a class-order mismatch.** `preds_test.npz` saved the logits,
predictions and labels, which makes this decidable instead of arguable. Build
the 15x15 prediction-vs-label matrix and solve for the best possible one-to-one
relabelling of the predicted classes:

```
raw accuracy          : 23.67%
best permutation acc  : 23.67%     <- no relabelling helps at all
```

A label-mapping bug would be *repaired* by the right permutation. This one is
not, so the mapping is innocent. The model also never predicts 4 of the 15
classes and puts 61% of its predictions onto four of them — that is a broken
model, not a working model with scrambled labels.

**It is a stale artifact.** The file times settle it:

| file | written |
|---|---|
| `preds_test.npz`, `report_test.json` | Aug 26, **15:09** |
| `best.pt` | Aug 26, **21:52** |

The evaluation ran **6h43m before** the checkpoint it supposedly measured. Those
files belong to an earlier run that `best.pt` later overwrote, and no epoch in
`history.json` comes near 23.67% (epoch 1 = 10.1%, epoch 2 = 28.7%).

**So the current `transformer` checkpoint has never been evaluated at all.** The
23.67% is not a result, it is debris — do not report it anywhere. Day 1.4 is
therefore re-running the evaluation and making this failure impossible to
repeat, rather than hunting a bug that does not exist.

---

## 1. How every step works

Every numbered step in this plan — `1.1`, `6.2`, `C03` — has the same five
parts, in the same order, including the easy ones.

**1. Why.** What problem the step solves, in plain words, before any code
appears. If you cannot say it in one sentence, the step does not start.

**2. Build.** One small piece: a single function, a single file, or a single
command. Small enough that if the result is wrong, you know which line is
wrong.

**3. Run.** The exact command, copy-pasteable. Something actually executes — no
step ends on "this should work".

**4. Check.** What you should see on screen, and the one number or file that
proves it worked. A step with no check is not finished.

**5. Explain.** What just happened, in easy terms, and the one idea worth
keeping. This part is what becomes your Implementation chapter and your answers
in the viva.

Then I stop and wait for **"next"**. I will not run ahead through several steps
because they look easy.

### Why the steps are small

A step is the right size when only one thing in it can go wrong. If a step
builds three things and the output is wrong, you have to guess which of the
three broke — and guessing is slow and teaches you nothing. One piece at a time
means every failure points straight at its own cause.

### On jargon

Any term that is not plain English gets one sentence of plain English the first
time it appears — "stratified", "frozen backbone", "logits", "top-1", all of
them, where they are used rather than assumed. If something is still unclear,
say so and we stop there. An unexplained step is a step you cannot defend in a
viva, which makes it worthless even when the code is correct.

### The small steps are written on the day

Days 1–7 are broken into their small steps below, because we already know what
they are. Phase 3 and Phase 4 are listed as **one line per experiment**, and get
broken down when we reach them. That is deliberate: the small steps of
experiment 25 depend on what experiment 24 found, so writing them now would be
guessing.

---

## 2. Rules for the 20 days

1. **Every experiment appends one row to `experiments/results.jsonl`.** Written
   and fsynced by `append_result` the moment the run ends, so an outage cannot
   take it. If it is not in the ledger it did not happen. Your supervisor gets
   this file.
2. **One variable at a time.** Change the backbone *or* the temporal head, never
   both, or you cannot attribute the difference.
3. **Same splits for everything.** Generated once on Day 1, committed, never
   regenerated. Different splits make results incomparable.
4. **Fixed seed, and record it.** Also record the seed in the ledger row.
5. **Never tune on test.** Choose using validation. Test is touched once per
   experiment, at the end, and never looked at again.
6. **Preprocess once, reuse forever.** The expensive work happens on Days 2–3 and
   is cached. Everything after that is cheap.
7. **If a run looks too good, it is a leak.** Check the split first.
8. **One journal entry a day, in `notes/journal.md`.** What broke and how you
   fixed it, with a number or a file path behind every claim. This is the raw
   material for your Implementation chapter.
9. **Record it before you move on.** An outage between finishing something and
   writing it down costs you the work twice: the number, and the reason.

### Your hardware

- **Intel Arc B580 (XPU)**, torch 2.13.0+xpu, detected and working.
- bf16 autocast, **no** GradScaler (that is CUDA-only). `src/utils/device.py`
  already handles this — always import `get_device()` and `get_amp_settings()`.
- No CUDA on this machine. The RTX 3070 machine and Kaggle are both options for
  the heavier Phase 3 runs; configs for Kaggle already exist.

---

## 3. Working through load shedding

The power can cut at any moment and the desktop has no battery, so the process
dies instantly, mid-epoch. The plan is arranged so that costs minutes, not days.

**What is already safe.**

- **Preprocessing.** `build_dataset.py` skips any clip whose output already
  exists, so the Days 2–3 pipeline over 10,091 videos can be killed and
  restarted as often as needed. It continues; it does not start over.
- **Training.** `src/train.py` writes `resume.pt` every epoch — model,
  optimizer, scheduler, history and the random-number state. Resuming is the
  **default**: when the power comes back, re-run the identical command.

  ```bash
  python src/train.py --config configs/mimic_author.yaml
  #  -> RESUMED : epoch 37 finished, continuing at 38/100 | best val top-1 81.44%
  ```

  `--fresh` forces a restart from epoch 1. `tests/test_checkpoint.py` holds this
  to a strict standard: a resumed run must match an uninterrupted one
  bit-for-bit.
- **Results.** `append_result` writes one fsynced line to
  `experiments/results.jsonl` the moment a run finishes, so a measured number is
  never only in the terminal.
- **Checkpoints.** Every save is atomic — written to `.tmp`, then `os.replace`
  — so a cut during a save cannot corrupt the best model you had already earned.

**What is still expensive.** Families B, C, E and F fine-tune end to end, hours
per run. They resume, but they are the only jobs where an outage costs real
time. Everything in Families A, D and V trains in minutes from cached features,
so an outage rarely lands inside one at all.

**How to shape the day.**

| When | What to do |
|---|---|
| Power on, long block | Launch the hours-long runs (B, C, E, F) |
| Power on, short block | The cheap cached-feature experiments (A, D, V) |
| Outage | Error analysis, reading, the journal, the write-up — no GPU needed |

One cheap UPS, even 650 VA, buys 10–15 minutes: enough to finish an epoch and
shut down cleanly. It is the only hardware purchase that pays for itself here.

---

# PHASE 1 — Reproduce the authors (Days 1–5)

Target: **89.09% ± 1.5%**. Nothing else starts until this lands.

## Day 1 — Splits, and clearing the evaluation debris

Nothing trains today. Today is about making every later number trustworthy.

> The old step 1.2, "start the results ledger", is **already built**:
> `append_result` in `src/utils/checkpoint.py`, done 2026-10-07. Four steps
> remain.

### 1.1 Look at the filenames before writing any code (≈10 min)

**Why.** Every split decision depends on what a filename actually tells you.
Guessing the naming scheme and finding out on Day 15 that you guessed wrong
would invalidate everything in between.

**Build.** Nothing. Look at the data first.

**Run.**

```bash
ls data/CricShoot10kShootDataset | head
ls data/CricShoot10kShootDataset/*/ | head -20
```

**Check.** Clip names look like `vid100_41.avi`. Confirm 15 class folders exist
and the total clip count is near 10,091.

**Explain.** `vid100` is the **source match video**; `_41` means the 41st shot
cut out of it. So many clips share one `vidNNN`, and clips from the same match
share the same batter, kit, ground, lighting and camera angle. That is exactly
the problem step 1.2 exists to prevent.

### 1.2 Write the grouping function, and test it on its own (≈20 min)

**Why.** If `vid100_41.avi` lands in train and `vid100_42.avi` lands in test,
the model can score well by recognising *that match* rather than the shot. The
test number comes out high and means nothing. That is a **leak**: information
from the test set reaching the model during training.

**Build.** One small function — `match_id("vid100_41.avi") -> "vid100"` — and a
test covering the names that do not fit the pattern.

**Run.**

```bash
python -m pytest tests/test_splits.py -q
```

**Check.** The test passes, including the odd filenames.

**Explain.** Two ideas, pulling in opposite directions:

- **Stratified** splitting keeps the same proportion of each shot class in
  train, val and test. You want this.
- **Grouped** splitting forces every clip sharing a `vidNNN` into the *same*
  split. You need this more.

When clips are cut from shared source videos, grouping matters more than
stratification: a leak inflates your score and invalidates the result, while
slightly uneven class proportions only add a little noise.

### 1.3 Generate the splits and prove there is no leak (≈20 min)

**Why.** These three files are frozen for all 20 days. Every experiment must use
them, or the results cannot be compared with each other.

**Build.** `src/utils/make_splits.py` — grouped by match, stratified by class.

**Run.**

```bash
python -m src.utils.make_splits --data_root data/CricShoot10kShootDataset \
    --out data/splits --test_frac 0.2 --val_frac 0.2 --seed 42
```

**Check.** Four things, and all four must hold:

1. Three CSVs exist in `data/splits/`.
2. No filename appears in two splits.
3. **No `vidNNN` appears in two splits** — this is the one that matters.
4. Class proportions agree across the splits to within about 1%.

**Explain.** Then commit them immediately. Regenerating splits later, even with
the same seed, silently makes earlier results incomparable — and that is the
kind of mistake a results table cannot show you.

### 1.4 Make a stale evaluation impossible (≈30 min)

**Why.** `experiments/transformer/` holds a 23.67% test result measured on a
checkpoint that no longer exists (see *The second bug* above). It has already
cost one wrong diagnosis.

**Build.** Two changes to `src/evaluate.py`: record the checkpoint's SHA-256 and
modification time inside `report_*.json`, and refuse to trust a report whose
recorded hash does not match the checkpoint in front of it.

**Run.** The re-evaluation itself needs `data/processed/`, so it waits for Day 3.
The hash guard can be built and tested today.

**Check.** The regenerated `report_test.json` carries a hash matching the
current `best.pt`, and its test number sits within a few points of the val
figure, as it does for the other four runs.

**Explain.** The lesson is not really about this bug. A stale result that *looks*
real cost a wrong diagnosis that survived for weeks. Anything that records a
measurement should also record **what it measured** — the ledger row, the
journal entry and the report all name their checkpoint. Cheap to add now,
impossible to reconstruct later.

> **You will learn:** how to tell a training failure from an evaluation failure.
> They look identical in a results table and have completely different fixes.

---

## Day 2 — The preprocessing pipeline, on 30 videos

Do **not** launch the full run today. Get it right on a tiny subset first.

### 2.1 Fix the model paths (≈15 min)

`build_dataset.py` defaults to `data/CricShoot10kModels/...` but the models live
at the repo root in `CricShoot10kModels/`. Fix the defaults or pass them
explicitly.

### 2.2 Smoke test: 2 videos per class (≈1 h)

```bash
python -m src.preprocess.build_dataset --limit 2 --device xpu \
    --data_root data/CricShoot10kShootDataset \
    --striker_model CricShoot10kModels/Player_Type_Detection_Model.pt \
    --bat_model     CricShoot10kModels/Bat_Detection_Model.pt \
    --seg_model     CricShoot10kModels/Striker_Bat_Segmentation_Model.pt \
    --output_dir data/processed_smoke
```

**Check — and this is the important one: _look at the pictures_.** Save 10
processed frames as PNGs and open them. You must see:

- the frame tightly cropped around the striker and the bat, not the whole ground;
- the striker tinted **blue** and the bat tinted **green**, both at about 50%
  opacity;
- 15 frames per clip, 224×224.

If the crops are wrong, everything downstream is wasted. Three hours of looking
at images here saves a week.

> **You will learn:** what the two YOLO layers actually do, and why the authors
> gained 10 points from cropping alone — you remove the background the model
> was overfitting to.

### 2.3 Handle the failures (≈45 min)

`crop_video` drops frames where no striker is detected, so some clips come out
with fewer than 15 frames, and a few with zero. Decide and write down:

- fewer than 15 frames → loop/pad the last frame, or re-sample with replacement;
- zero frames → log the filename to `data/processed/failed.txt` and exclude it.

**Check:** the failure list is under ~2% of clips. If it is higher, the detector
confidence threshold is too strict.

---

## Day 3 — Run the full preprocessing (mostly waiting)

### 3.1 Launch it (≈20 min to start, then 4–8 h)

```bash
python -m src.preprocess.build_dataset --device xpu \
    --data_root data/CricShoot10kShootDataset --splits_dir data/splits \
    --output_dir data/processed  [--model paths as above]
```

10,091 clips × 24 frames × 3 detector passes ≈ 700k inferences. **Start it before
you go to sleep.** Output is `.pt` uint8 tensors of shape (15, 224, 224, 3),
roughly 2–4 GB total.

**Check next morning:** `train/ val/ test/` each contain 15 class folders; the
`.pt` count plus the failure count equals 10,091; load three at random and
confirm shape and dtype.

### 3.2 While it runs — read the authors' notebook (≈2 h)

Open `CricShot10kAuthorsCode/Codes/Model_Layer_CNN_RNN_Training_Testing.ipynb`
and write down, in your own words, every number: batch size, epochs, learning
rate, scheduler, what is frozen. `configs/mimic_author.yaml` already records the
discrepancies between the paper and the notebook (batch 4 not 8, 100 epochs not
50) — confirm each one yourself.

> **You will learn:** that papers and their code disagree, and that the code wins.

---

## Day 4 — Train the author replication

### 4.1 Launch (≈30 min to start, then 3–6 h)

```bash
python src/train.py --config configs/mimic_author.yaml
```

Architecture must be exactly: EfficientNetV2-S (ImageNet, TimeDistributed) →
Flatten → GRU(128) → BatchNorm → Dense(1024, ReLU) → Dense(15, softmax). **No
dropout.** Adam lr=1e-4, batch 4, ReduceLROnPlateau(0.1, patience 4), early stop
patience 10.

**Check at epoch 5:** val accuracy should already be past 60%. If it is stuck
near 7% (= 1/15) the labels are shuffled. If it is near 75% again, the model is
reading the old uncropped data — check `processed_root`.

### 4.2 Horizontal-flip augmentation (≈1 h)

Offline, **training split only**, applied *after* the split. Flipped clips are
not counted in the 10,091.

**Check:** the training set roughly doubles; val and test are untouched.

---

## Day 5 — Hit the number, and lock it in

### 5.1 Evaluate properly (≈1 h)

Top-1, top-2, top-3, precision, recall, macro-F1, AUC-ROC, and the full 15×15
confusion matrix.

**Target:** 89.09% ± 1.5%. The authors' per-class sanity checks:

- **Late Cut** is the worst class (F1 ≈ 0.745) — it looks like Upper Cut and
  Square Cut;
- **Scoop** is the best (recall ≈ 1.0) — the most visually distinct.

If your confusion matrix shows the same pattern, you have genuinely reproduced
their model. If it does not, something differs even if the headline number
matches.

### 5.2 If you are short of 89% (≈2–4 h)

In order of likelihood:

1. Crops are wrong → go back to Day 2.2 and look at the images again.
2. Flip augmentation missing or applied before the split.
3. Trained too few epochs — they converged around epoch 30 of 100.
4. Frozen backbone — the authors fine-tune the whole thing.
5. Wrong normalisation (ImageNet mean/std).

### 5.3 Write the baseline row and tag it (≈30 min)

```bash
git add -A && git commit -m "Reproduce CricShotNet baseline: <your> % top-1"
git tag baseline-reproduced
```

**This is the single most valuable artifact of the 20 days.** Everything after it
is measured against it.

---

# PHASE 2 — Make 30 experiments affordable (Days 6–7)

Thirty experiments at 4 hours each is 120 hours. You do not have that. The trick
is that most of them do not need to touch the video at all.

## Day 6 — Cache the CNN features

### 6.1 Extract once (≈2 h + 1 h compute)

Run the frozen EfficientNetV2-S over every processed clip and save a
**(15, 1280)** feature tensor per clip.

```
data/features/effnetv2s/{train,val,test}/<class>/<clip>.pt
```

**Check:** 10,091 files, each (15, 1280) float16; about 600 MB.

### 6.2 Why this changes everything

Any experiment that only replaces the **temporal head** — GRU, LSTM, BiLSTM,
transformer, TCN, attention pooling, SVM, XGBoost — now trains on these features
in **1–3 minutes instead of 4 hours**, because the CNN never runs again.

That is Family A and Family D below: roughly 17 of the 30 experiments, finishable
in a single day.

**Check:** retrain GRU-128 on the cached features alone. It should land within
~1 point of Day 5's number. If it is far off, the cache is wrong.

> **You will learn:** the difference between end-to-end training and
> feature-then-head, and why the second is the right tool for an ablation sweep.

### 6.3 Cache more extractors (≈2 h)

The cache is a *format*, not one model: anything that turns a clip into a
feature tensor plugs into the same harness. Add, in this order:

1. **ConvNeXt-Tiny**, per frame — another ImageNet view, (15, 768)
2. **r3d_18** (Kinetics) — your first *video-pretrained* features, one vector
   per clip, (512,)
3. **VideoMAE-Base** (Kinetics) — (768,)
4. **CLIP ViT-B/16**, per frame — (15, 512)

Each is one pass over 10,091 clips, roughly 20–40 minutes on the Arc. After
this, every experiment in Families A, D and V is a few minutes of training on
tensors that already exist.

**Check:** train the same linear head on each cache and compare. If the
Kinetics features beat the ImageNet ones on a linear probe alone, that tells you
motion pretraining matters here — before you have spent a single hour
fine-tuning.

**Also cache the un-segmented version.** Family V needs cropped-but-not-
segmented frames too (see Appendix A), so run `build_dataset.py` a second time
with segmentation off into `data/processed_nocrop_seg/`. Disk is cheap; a
re-run on day 10 is not.

## Day 7 — The experiment harness

### 7.1 One command per experiment (≈3 h)

```bash
python -m src.experiments.run --family A --exp A03_bilstm
```

It must: load the config, set the seed, train, evaluate, append one row to
`experiments/results.jsonl` via `append_result`, and save the confusion matrix
PNG. No manual steps — manual steps are how you lose six results in week three,
and an outage mid-experiment is how you lose the seventh.

**Check:** run the same experiment twice. Two identical rows (same seed) means
the harness is deterministic.

### 7.2 A results notebook (≈1 h)

Reads `results.jsonl` with `read_results`, prints the leaderboard sorted by val
accuracy, and plots accuracy per family. Re-run it at the end of every day.

---

# PHASE 3 — The 30+ experiments (Days 8–15)

Full list with IDs in **Appendix A**. Order is deliberate: cheap and informative
first.

| Day | Family | Experiments | Cost each |
|---|---|---|---|
| 8 | **A** — temporal heads | A01–A12 (12) | 2–5 min |
| 9 | **D** — features + classical ML | D01–D06 (6) | 1–10 min |
| 10 | **V** — video-pretrained features (frozen) | V01–V10 (10) | 20 min extract, 3 min train |
| 11–12 | **B** — CNN backbones | B01–B08 (8) | 2–4 h |
| 13–14 | **C** — 3D / video models (fine-tuned) | C01–C08 (8) | 3–6 h |
| 15 | **E** — pose and graph | E01–E04 (4) | 2–4 h |
| stretch | **F** — multi-scale and hybrid | F01–F05 (5) | 2–4 h |
| stretch | **W** — cricket-specific motion | W01–W05 (5) | 2–5 h |

**58 listed, 36 committed.** Days 8–10 alone give you **28 rows** because they
all reuse the cached features. Families F and W are stretch: do them if Phase 1
finished early, otherwise they are future work and you say so.

### Daily rhythm

Each experiment below is still one step of the loop in section 1 — why, build,
run, check, explain — broken down on the day it is run.

1. **Power on, long block:** launch the hours-long runs (B, C, E, F — queue them).
2. **While they run:** the cheap cached-feature ones, and read about the next family.
3. **Outage:** error analysis, reading and the journal. None of it needs a GPU.
4. **End of day:** check the ledger has one row per run, regenerate the
   leaderboard, and write the journal entry — what broke, and how you fixed it.

### Rules that keep this honest

- Same splits, same seed, same evaluation, every time.
- A failed experiment is **still a row** — with `notes="diverged at epoch 3"`.
  Negative results are results, and your supervisor should see them.
- Do not fix an experiment to make it win. Record what happened.

---

# PHASE 4 — Combine and improve (Days 16–19)

Only now do you look at the leaderboard and pick.

## Day 16 — Choose, honestly

Take the top 5 by **validation** accuracy. For each, ask: is it genuinely better,
or just lucky? Re-run the top 3 with **three different seeds** and report
mean ± std.

> A 0.4% gap with ±0.8% std is not an improvement. This is the single most common
> way student papers get rejected.

## Day 17 — Tune the best one

Only the winner. One axis at a time:

- learning rate: 3e-5, 1e-4, 3e-4
- frames: 15, 20, 25
- backbone LR multiplier: 0.1, 0.5, 1.0
- label smoothing: 0.0, 0.05, 0.1
- class-balanced loss (see Appendix B — your minority classes are 4× smaller)

**Check:** every tuning run is also a ledger row.

## Day 18 — Ensembles and test-time augmentation

- Soft-vote the top 3 (different backbones ensemble better than similar ones)
- TTA: horizontal flip + 2 temporal crops
- Weighted vote, weights chosen on **val**

**Check:** the ensemble beats its best member on val before you touch test.

## Day 19 — Error analysis

- Full 15×15 confusion matrix for the best model
- The 20 worst-misclassified clips — **watch them**
- Is Late Cut still the worst class? Did your model fix it or move the problem?
- Grad-CAM on 5 correct and 5 incorrect clips: is it looking at the bat or the
  crowd?

This section is what turns a results table into a paper.

---

# Day 20 — The package for your supervisor

1. `experiments/results.jsonl` — all 43+ rows (export to CSV for the write-up)
2. A leaderboard table, sorted, grouped by family
3. Confusion matrix of the baseline and of the best model, side by side
4. Accuracy-vs-parameters scatter (which models earn their size)
5. Top-3 with mean ± std over 3 seeds
6. A one-page summary: what you reproduced, what you tried, what won, by how
   much, and what you now believe is the bottleneck
7. Every checkpoint and config committed

---

# Appendix A — The experiment list

## Family A — Temporal heads (backbone frozen, cached features)
*Fast. 2–5 minutes each. Answers: how much does the temporal model matter?*

| ID | Model | Why |
|---|---|---|
| A01 | Mean pooling (no temporal model) | **Control.** If this ties GRU-128, order does not matter and that is a finding. |
| A02 | Max pooling | Does one decisive frame carry the class? |
| A03 | GRU-128 | The authors' head — your reference point |
| A04 | GRU-64 | Authors got 85.52%; confirm the trend |
| A05 | BiGRU-128 | Does seeing the follow-through first help? |
| A06 | LSTM-64 | Authors: 85.42% |
| A07 | BiLSTM-64+64 | Authors: 87.65% — their second best |
| A08 | Temporal Transformer 2L/4H | Frame 1 attends directly to frame 15 |
| A09 | Temporal Transformer 4L/8H | Does more capacity help on 15 tokens? |
| A10 | TCN (dilated 1-D conv) | Cheaper than attention, often as good |
| A11 | Attention pooling | Which frames matter? Gives a free figure |
| A12 | NetVLAD | Learned aggregation, strong in video retrieval |

## Family B — CNN backbones (GRU-128 head fixed)
*2–4 hours each. Answers: how much does the spatial encoder matter?*

| ID | Backbone | Params | Why |
|---|---|---|---|
| B01 | EfficientNetV2-S | 20.3M | The authors' choice, 89.09% |
| B02 | EfficientNetV2-B3 | 12.9M | Authors: 88.39% at 2/3 the size |
| B03 | Xception | 20.9M | Authors: 88.64% |
| B04 | ConvNeXt-Tiny | 27.8M | Authors: 87.20% — modern conv design |
| B05 | MobileNetV3-Large | 5.4M | Edge deployment story |
| B06 | ResNet-50 | 25.6M | The baseline everyone knows |
| B07 | RegNetY-16GF | 83M | Does raw capacity help? |
| B08 | Swin-Tiny (2D) | 28M | **Your "vision transformer"** as a per-frame encoder |

## Family C — 3D and video models (end-to-end)
*3–6 hours each. Answers: does joint space-time beat 2D-then-temporal?*

| ID | Model | Why |
|---|---|---|
| C01 | R(2+1)D-18 | Factorised 3D conv; strong, cheap |
| C02 | 3D ResNet-18 | **Your "3D CNN"** — the classic |
| C03 | I3D (inflated Inception) | The standard video baseline |
| C04 | SlowFast R50 | Two pathways, two frame rates |
| C05 | X3D-M | Efficient 3D, very good per FLOP |
| C06 | MViT-v2-S | Multiscale vision transformer |
| C07 | Video Swin-T | **Authors explicitly suggested this as future work** |
| C08 | VideoMAE / TimeSformer | Self-supervised pretraining; strongest candidate to beat 89% |

## Family D — Cached features + classical models
*1–10 minutes each. Answers: how separable are the features already?*

| ID | Head | Why |
|---|---|---|
| D01 | Linear probe | Lower bound on feature quality |
| D02 | SVM (RBF) | Strong on small, high-dim data |
| D03 | Random Forest | Gives feature importances |
| D04 | XGBoost | Usually the best classical option |
| D05 | k-NN | Are same-class clips neighbours in feature space? |
| D06 | MLP (2 hidden layers) | Non-linear head, no recurrence |

## Family E — Pose and graph
*2–4 hours each. Answers: does explicit body structure help?*

| ID | Model | Why |
|---|---|---|
| E01 | MediaPipe pose → GRU | Rao et al. got 80% pose-only |
| E02 | **ST-GCN** on pose keypoints | **Your "GNN"** — joints as graph nodes |
| E03 | Two-stream: RGB + pose, late fusion | Each stream's strength |
| E04 | RGB + pose cross-attention | Deeper fusion |

## Family F — Multi-scale and hybrid
*2–4 hours each.*

| ID | Model | Why |
|---|---|---|
| F01 | **Multi-scale CNN** (10/15/20 frames) | Your existing `multiscale_model.py` |
| F02 | **Hybrid CNN + Transformer** | Your existing `combined_model.py` |
| F03 | Two-stream RGB + optical flow | The classic video recipe |
| F04 | Multi-resolution (224 + 112) | Cheap context + detail |
| F05 | CNN + GRU + attention (all three) | Kitchen sink, as an upper bound |

## Family V — Video-pretrained feature extractors (frozen)

*The thing your current pipeline is missing entirely. See "Why these are
different" below. 20 min to extract, 3 min to train a head — all reuse the
Day 6 cache.*

| ID | Model | Pretrained on | Where from |
|---|---|---|---|
| V01 | **r3d_18** | Kinetics-400 | `torchvision.models.video` ✅ installed |
| V02 | **mc3_18** | Kinetics-400 | torchvision ✅ |
| V03 | **r2plus1d_18** | Kinetics-400 | torchvision ✅ |
| V04 | **s3d** | Kinetics-400 | torchvision ✅ |
| V05 | **swin3d_t** | Kinetics-400 | torchvision ✅ |
| V06 | **mvit_v2_s** | Kinetics-400 | torchvision ✅ |
| V07 | **VideoMAE-Base** | Kinetics-400, self-supervised | `transformers` ✅ |
| V08 | **TimeSformer-Base** | Kinetics-400 | `transformers` ✅ |
| V09 | **CLIP ViT-B/16**, per frame | 400M image–text pairs | `timm` / `transformers` ✅ |
| V10 | **DINOv2-Base**, per frame | self-supervised images | `timm` ✅ |

### Why these are different from Family B and C

**EfficientNetV2-S was trained on ImageNet — still photographs.** It has never
seen motion. It looks at each of your 15 frames separately and reports "a person
holding a bat." Your GRU then has to reconstruct the stroke from 15 such
descriptions. That is like identifying a dance from 15 snapshots.

**Kinetics-400 is 300,000 video clips of humans performing 400 actions — and one
of those 400 classes is "playing cricket."** A model pretrained on it already
encodes "a swinging motion that begins low and finishes high." You are starting
from motion understanding instead of building it from scratch.

**V differs from C** in how the model is used, and the comparison is itself a
result worth reporting:

- **Family V**: backbone **frozen**, used once as a feature extractor, cheap head
  on top. Minutes per experiment.
- **Family C**: the same architectures **fine-tuned end to end**. Hours per
  experiment, usually better.

Running `r3d_18` in both V01 and C02 tells you how much fine-tuning is worth on
a 10k-clip dataset — a genuinely useful number for a paper.

### Two things that will trip you up

1. **Segmentation may hurt these models.** The blue/green overlay removes jersey
   noise, which helped the authors' ImageNet backbone. But a Kinetics model has
   never seen a blue person holding a green bat. **Run every Family V experiment
   on cropped-but-not-segmented frames as well** and report both. If the
   unsegmented version wins, that is a real finding about transfer from natural
   video.
2. **They want different input.** 3D models expect 16 or 32 frames (not 15) and
   their own normalisation statistics. Check each model's `weights.transforms()`
   — torchvision ships the correct preprocessing with the weights, so use it
   rather than reusing your ImageNet transform.

## Family W — Cricket-specific motion features (stretch)

*Nobody has published these on CricShot10k. This is where an original
contribution would come from.*

| ID | Features | Why |
|---|---|---|
| W01 | Optical flow (RAFT) → 3D CNN | The classic motion stream |
| W02 | **Bat trajectory** — centroid path, angle, speed per frame | You already have `Bat_Detection_Model.pt` |
| W03 | **Ball trajectory** — position, approach angle | You already have `Ball_Detection_Model.pt` |
| W04 | Bat + ball + pose, fused | A compact, interpretable physical description of the stroke |
| W05 | Frame-difference stack | Cheapest possible motion signal; a good control |

W02–W04 are the interesting ones. You hold three fine-tuned YOLO detectors that
nobody has used for anything except cropping. A pull and a hook differ mainly in
**bat path and contact height** — exactly what a bat trajectory encodes, and
exactly what a 2D CNN throws away. A 40-dimensional trajectory feature that gets
within a few points of an 89% deep model would be a genuinely publishable result.

## Family G — Ensembles (Day 18)

| ID | Method |
|---|---|
| G01 | Soft-vote top 3 from different families |
| G02 | Weighted vote, weights fitted on val |
| G03 | Test-time augmentation (flip + temporal crops) |
| G04 | Stacking: a logistic head over the 3 models' probabilities |

**Total: 43 experiments.**

---

# Appendix B — What to apply to the dataset

You asked specifically what to do to the *data*. This matters more than the
model — Phase 1 is proof of that.

## Tier 1 — Do these (they are the 14 points)

| # | What | Why | Gain |
|---|---|---|---|
| 1 | **Crop to striker + bat** (YOLOv11) | Removes crowd, boards, grass | **+10%** |
| 2 | **Segmentation overlay** (striker blue, bat green, α=0.5) | Removes jersey-colour noise across formats | **+3%** |
| 3 | **Horizontal flip**, training split only, after the split | Simulates left/right-handed batters | **+1.5%** |
| 4 | **Grouped splits by `vidNNN`** | Same match in train and test is a leak | prevents fake gains |

## Tier 2 — Likely to help, worth a row each

| # | What | Why |
|---|---|---|
| 5 | **More frames**: 20 or 25 instead of 15 | The authors chose 15 by testing; re-test now that you crop |
| 6 | **Temporal jittering** (random rather than uniform sampling) | The authors never tried it; cheap |
| 7 | **Class-balanced loss or oversampling** | Reverse Sweep 252 vs Lofted Legside 1123 — a **4.5×** imbalance |
| 8 | **Focal loss** | Pushes capacity toward Late Cut and the other hard classes |
| 9 | **Colour jitter** | Day/night matches, different broadcasters |
| 10 | **Speed perturbation** (0.8×–1.2×) | Fast vs slow bowlers change stroke timing |
| 11 | **Random resized crop** (mild, 0.9–1.0) | Robustness to imperfect YOLO crops |
| 12 | **Test-time augmentation** | Nearly free 0.5–1% |

## Tier 3 — Try, but expect nothing

| # | What | Caution |
|---|---|---|
| 13 | MixUp / CutMix | **You already found this hurts.** Mixing Pull with Hook makes an ambiguous label. Keep the negative result — it is publishable. |
| 14 | RandAugment | Aggressive policies destroy fine-grained stroke differences |
| 15 | Heavy rotation / vertical flip | A batter never plays upside down; this is nonsense augmentation |

## Tier 4 — Data work that would make a real contribution

| # | What | Why it matters |
|---|---|---|
| 16 | Hand-label **front-foot vs back-foot** defensive shots | The authors list this as future work |
| 17 | Add **leave, leg glance, backfoot punch** | Three new classes they explicitly asked for |
| 18 | Label **batter handedness** | Enables multi-task learning |
| 19 | Label **match format** (Test/ODI/T20) | Enables the cross-format generalisation study |
| 20 | Re-check the 868 clips that lacked unanimous annotator agreement | Label noise caps your ceiling |

---

# Appendix C — When things go wrong

| Symptom | Most likely cause |
|---|---|
| Accuracy stuck near 6.7% | Labels shuffled, or `class_names` sorted differently in train and eval |
| Val 75%, test 24% | A report measured on a checkpoint that was later overwritten. Compare the file times before suspecting the labels (Day 1.4) |
| Val much higher than test | Leak: clips from one `vidNNN` in both splits |
| Train 99%, val 76% | Overfitting — add dropout, more augmentation, or stop earlier |
| Stuck at exactly ~75% | You are reading uncropped data. Check `processed_root`. |
| XPU out of memory | batch 4 → 2, `grad_accum_steps` 2 → 4 |
| `GradScaler` error on Arc | Do not use it on XPU — bf16 needs no scaler |
| Preprocessing crawls | Check the YOLO models are on XPU, not CPU |
| 3D models will not fit | Move to the RTX 3070 or Kaggle |
| A run restarted from epoch 1 after an outage | No `resume.pt` — either `--fresh` was passed, or the previous run had already finished |
| `best.pt` will not load | A cut during a non-atomic save. Use `atomic_save`; `best.pt.tmp` next to it is the giveaway |

---

## The schedule in one table

| Days | Phase | Output |
|---|---|---|
| 1 | Grouped splits; clear the stale evaluation artifacts | trustworthy foundation |
| 2–3 | Preprocessing: crop + segment | `data/processed/` |
| 4–5 | Author replication | **89% baseline, tagged** |
| 6–7 | Feature cache + harness | 30 experiments made affordable |
| 8–15 | 36 core experiments (58 listed, A–E + V) | `results.jsonl` |
| 16–19 | Combine, tune, ensemble, error analysis | best model + mean ± std |
| 20 | Package for supervisor | leaderboard, figures, summary |

---

*Plan written for the CricShot10k accuracy work. The CricketMind / LangGraph
work is paused for these 20 days.*
