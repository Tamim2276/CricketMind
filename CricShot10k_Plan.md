# CricShot10k — The Plan

**Goal.** First reproduce what Dihan et al. achieved (89.09% top-1), from code
you wrote. Then run 46 different approaches **on two different data splits** and
record all 92 results. Then combine and tune the best ones.

> **Status 2026-10-13.** Phase 1 is complete and tagged at **78.14%** test
> top-1, not 89.09%. Six candidate causes were eliminated by measurement
> (section 7.4), and the authors' own released weights score **43%** on our
> best crops against our model's 78%, which places the residual in their
> preprocessing rather than in anything here. Phase 2 begins from 78.14%.

**26 to 36 days.** Count GPU-hours, not days: the whole project is **180
GPU-hours**, of which **159 are sit-and-wait** (Families B and C, and the Phase 4
re-runs). The rest fit inside days that are paced by learning anyway. So:

> **16 learning-paced days + 159 GPU-hours / hours-the-power-is-on**

That denominator is *power* availability, not desk time — training auto-resumes,
so a run continues overnight. 8 h/day gives 36 days; 16 h/day gives 26. Section 3
has the detail.

**How we work.** We are starting from an empty `src/` and building every file
one at a time. You are learning this while building it, so nothing is built
ahead of you, nothing is left in jargon, and **every piece of code is walked
through line by line before it is run**. Every step follows the same six-part
loop, set out in [section 1](#1-how-every-step-works). Short version: **one
small piece → I explain the code → run it → check it → you say "next"**. Every
day ends with something finished, a number in the ledger, and a line in the
journal.

**Out of scope for this project.** The LangGraph / CricketMind work is paused.
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

1. The raw clips are **896×540 full broadcast frames**, 22–49 frames each
   (see [section 5](#5-what-the-dataset-actually-contains)). The
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

### That conclusion was half right -- corrected 2026-10-13

Phase 1 is done and the preprocessing was applied in full. It was worth less
than this section predicted, and **two framework bugs were worth far more**:

| | test top-1 |
|---|---|
| old runs, uncropped | 74.93% |
| cropped, with PyTorch's GRU init | 32.39% |
| cropped, Keras-style GRU init | ~55% |
| cropped, + the authors' BatchNorm eps and momentum | **78.14%** |

Neither bug was in this plan, because neither is in the paper. Both are
**Keras/PyTorch default mismatches**, and both are invisible -- no crash, no
warning, just a worse number:

1. **GRU init.** PyTorch uses `uniform(+-1/sqrt(hidden))` and ignores fan-in.
   At 62720 flattened inputs that is 9x too large and saturates 65% of the
   gates at step 0. Keras uses `glorot_uniform`. See `_init_gru_like_keras`.
2. **BatchNorm.** Keras defaults to `epsilon=1e-3, momentum=0.99`; torch to
   `1e-5` and `0.1` (and defines momentum inversely, so theirs is 0.01 here).
   Their own `.keras` file records both. The epsilon alone, on *already
   trained* weights, moved val loss 3.1031 -> 1.4819.

**The lesson for the rest of this plan:** when porting a published model across
frameworks, read the *numeric defaults* of every layer out of the authors'
saved config, not just the architecture. Both bugs were sitting in
`config.json` the whole time.

**And preprocessing did not close the rest.** Cropping took the model to
78.14%. Segmentation, implemented to the paper's written specification, landed
at 76.97% -- statistically indistinguishable from cropping alone, not the
+3.33 the authors report. The remaining gap to 89.09% is documented at the end
of Day 7.

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

Every numbered step in this plan — `1.1`, `6.2`, `C03` — has the same six
parts, in the same order, including the easy ones.

**1. Why.** What problem the step solves, in plain words, before any code
appears. If you cannot say it in one sentence, the step does not start.

**2. Build.** One small piece: a single function, a single file, or a single
command. Small enough that if the result is wrong, you know which line is
wrong.

**3. The code, explained.** Every block of what we just wrote, in plain English:
what it does, why it is written that way, and what would break if it were
written the obvious way instead. Jargon gets a sentence of plain English on
first use. **This part happens before the code is run**, so you are never
watching output from something you do not understand.

**4. Run.** The exact command, copy-pasteable. Something actually executes — no
step ends on "this should work".

**5. Check.** What you should see on screen, and the one number or file that
proves it worked. A step with no check is not finished.

**6. Takeaway.** The one idea worth keeping, separate from the code. This is
what becomes your Implementation chapter and your answers in the viva.

Then I stop and wait for **"next"**. I will not run ahead through several steps
because they look easy.

Four kinds of step are exempt, because there is no new code in them: launching a
long job, waiting for one, a troubleshooting checklist, and a commit. They are
marked as such and keep only the parts that apply.

### Why the steps are small

A step is the right size when only one thing in it can go wrong. If a step
builds three things and the output is wrong, you have to guess which of the
three broke — and guessing is slow and teaches you nothing. One piece at a time
means every failure points straight at its own cause.

### How the code gets explained

Part 3 is the one you asked for, so it has rules of its own:

- **Block by block, not line by line where lines are obvious.** `import os` does
  not need a paragraph. A reducer, a reshape, or an index trick does.
- **Every shape is named.** When a tensor changes shape, the comment says what
  it was and what it became: `(B, T, C, H, W) -> (B*T, C, H, W)`. Shape
  confusion causes more deep-learning bugs than algorithms do.
- **The alternative is named.** Why `os.replace` and not a plain write; why
  grouped splitting and not stratified; why bf16 and not fp16 on this GPU. A
  decision you cannot argue against is a decision you have not understood.
- **You get asked to predict.** Before some runs I will ask what you think the
  output will be. Being wrong there is the cheapest possible place to be wrong.

If anything is still unclear, say so and we stop there. An unexplained step is a
step you cannot defend in a viva, which makes it worthless even when the code
runs.

### The small steps are written on the day

Days 1–9 are broken into their small steps below, because we already know what
they are. Phase 3 and Phase 4 are listed as **one line per experiment**, and get
broken down when we reach them. That is deliberate: the small steps of
experiment 25 depend on what experiment 24 found, so writing them now would be
guessing.

---

## 2. Rules

1. **Every experiment appends one row to `experiments/results.jsonl`.** Written
   and fsynced by `append_result` the moment the run ends, so an outage cannot
   take it. If it is not in the ledger it did not happen. Your supervisor gets
   this file.
   **Every row carries a `split` field** (`"author"` or `"grouped"`), because
   each experiment is run twice and the two numbers are not comparable to each
   other — see [section 6](#6-two-splits-because-the-authors-protocol-has-a-gap).
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
- **16 GB RAM, and that is the binding constraint.** Three intermittent
  failures during Day 2 -- a clip decoding 0 frames, an XPU "bad allocation",
  and a `MemoryError` in `np.stack` -- were all allocation failures under
  memory pressure. The Python process peaked at 0.96 GB; VS Code was holding
  **3.73 GB**, leaving 3.78 GB free. **Close the editor before preprocessing or
  a long training run.** One 49-frame clip needs a 68 MB contiguous block, and
  `np.stack` briefly needs it twice.

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

## 4. Starting fresh

The old `src/` produced the five runs that all landed at 75%, and it is not
trusted. It gets **archived, not deleted** — moved where it cannot be imported
by accident, but kept, because it still records the authors' hyperparameters and
a working XPU/CUDA device check that are worth consulting.

```bash
mkdir -p archive
git mv src archive/src_old
git mv configs archive/configs_old
git mv experiments archive/experiments_old
```

**One file comes back out.** That move takes all of `src/` with it, including
`src/utils/checkpoint.py`, which we are keeping (see below) — so the tests that
import it break until it is restored:

```bash
git mv archive/src_old/utils/checkpoint.py src/utils/checkpoint.py
```

The 1.3 GB of `.pt` weights under `archive/experiments_old/` stay out of git via
the existing `*.pt` rule. The ~1.7 MB of `history.json`, `report_test.json` and
`preds_test.npz` next to them **should** be committed: they are the evidence the
journal cites, and `preds_test.npz` is what proved the 23.67% was not a label
permutation.

Nothing in `archive/` is imported by anything we write. When a step needs
something from it, we read it, understand it, and retype it — that is the point.

### What gets built, in order

```
src/
  utils/
    device.py        Day 1  pick XPU / CUDA / CPU, and the right autocast
    seed.py          Day 1  one seed for python, numpy and torch
    splits.py        Day 1  match_id + grouped stratified splitting
    checkpoint.py    Day 2  atomic saves, resume, the results ledger
  data/
    video.py         Day 2  read an .avi into frames
    dataset.py       Day 5  the Dataset the DataLoader feeds from
    augment.py       Day 7  horizontal flip, once per clip
    features.py      Day 8  the Dataset that reads cached features
  preprocess/
    detect.py        Day 3  YOLO: find the striker and the bat
    striker.py       Day 3  choose the striker among the candidates
    crop.py          Day 3  one clip -> cropped frames, overlay included
    cuts.py          Day 4  find the camera cut near the end
    sample.py        Day 4  any length -> 15 frames
    build_dataset.py Day 4  run all of it over 10,091 clips, resumably
  models/
    encoder.py       Day 5  EfficientNetV2-S over each frame
    head.py          Day 5  GRU-128 -> Dense -> 15 classes
    model.py         Day 5  encoder + head, wired together
  engine.py          Day 6  one epoch of training; one pass of evaluation
  train.py           Day 6  the epoch loop, resumable
  evaluate.py        Day 7  test-set metrics, with a checkpoint hash
  plot.py            Day 7  one figure per run, drawn automatically
                     Day 9  + a report card per run (confusion, per-class F1)
  extract.py         Day 8  clips -> cached feature tensors, resumable
  experiments/
    registry.py      Day 9  46 experiment IDs as data
    run.py           Day 9  one ID -> two splits -> two ledger rows
configs/
  mimic_author.yaml  Day 6  the authors' exact settings
  cached_gru.yaml    Day 8  the same thing, reading the cache
tools/leaderboard.py Day 9  the ledger as a table, both splits side by side
tools/figures.py     Day 9  every comparison figure, rebuilt from the ledger
tests/               every step that can be tested gets a test here
notes/journal.md     one entry a day
```

**There is no `segment.py`.** Day 4.2 originally planned one, and the overlay
ended up inside `crop.py` instead, because segmentation is a *choice about how
to draw the crop* rather than a separate stage — `crop_clip(segmented=True)`
paints the striker blue and the bat green before cropping. One function, one
pass over the frames. The plan said otherwise until 2026-10-15.

Eighteen modules, none of them long. The order matters: each one is only built
once the thing before it works, so nothing is written against an interface that
might still change.

### The diagnostic tools, built during Day 7 (2026-10-10 to 13)

None of these were in the original plan. They exist because three days of
guessing at the authors' preprocessing produced six negative results, and each
one turns a 6-hour GPU question into a minutes-long CPU one. **Reach for them
before launching a run.**

```
tools/
  authors_model.py   Score the AUTHORS' trained Keras model on our crops.
                     The single most useful instrument here: it partitions the
                     gap into "our preprocessing" vs "our training" in 13 min.
                     Needs the keras-3.8.0 venv; refuses to report a number
                     from a positional weight load. See 7.4.
  crop_sweep.py      Vary one cropping parameter and re-score their model.
                     Detection is cached once, so 20 settings cost barely more
                     than one. Two stages, two interpreters.
  mask_quality.py    How much of the batter the YOLO masks actually cover,
                     over thousands of frames. Turned "it misses the pads"
                     into a measurement that refuted it.
  overlay_sweep.py   Render the segmentation overlay at several alphas, side
                     by side, before rebuilding 9,880 clips with a guess.
```

**The habit worth keeping:** before any multi-hour run that changes
preprocessing, ask whether the authors' frozen model can answer the question
first. It moved 34.46% → 43.22% when the segmentation was corrected, so it
does respond to real changes — and a null from it is weak evidence, not strong.

### Two files that already exist

`src/utils/checkpoint.py` and `tests/test_checkpoint.py` were built on
2026-10-07 and are tested — a resumed run matches an uninterrupted one
bit-for-bit. They were built in one jump, though, without the walkthrough. So
they stay, and **Day 2.1 is their explanation**: we read them together, line by
line, and you can change anything you disagree with. We do not delete tested
code for ceremony, and we do not keep code you cannot explain.

---

## 5. What the dataset actually contains

Audited 2026-10-08 by decoding **every frame of all 10,091 clips**, not by
reading the paper. Three things differ from what the paper says, and one of them
changes the code you are about to write.

### The dataset is fine. Do not re-download it.

| check | result |
|---|---|
| clips found | **10,091** `.avi` — matches the paper exactly |
| files that fail to open | **0** |
| zero-byte files | **0** |
| frames that fail to decode | **0** |
| frame rate | **30 fps**, all 10,091 |
| total size | 3.94 GB |

Every file opens and every frame decodes. There is also a pristine copy of the
original archive at `data/unnecessary/raw/CricShoot10kShootDataset.rar` (3.6 GB),
so even a damaged extraction would not need a download.

### But the clips are not all 24 frames

The paper says 24 frames. Only **7,657 of 10,091 (75.9%)** actually are:

| frames | clips | source matches |
|---|---|---|
| 22 | 57 | 2 |
| 23 | 6 | 1 |
| **24** | **7,657** | **442** |
| 28 | 1,390 | 45 |
| 29 | 25 | 1 |
| **49** | **956** | **43** |

**Frame count is a property of the source match, not of the clip.** All 534
matches have one single consistent frame count — zero matches mix. So the
variation comes from the broadcasts the authors cut from, and it is spread
across all 15 shot classes roughly in proportion, which means **clip length is
not a shortcut to the label**.

Two consequences for the code:

1. `video.py` must return the real length, never assume 24 (Day 2.3).
2. `sample.py` must be `T → 15` for any `T` from 22 to 49, with the step size
   taken from the clip (Day 4.1). A hardcoded `24 → 15` would silently mangle a
   quarter of the dataset — and because it would still *run*, you would only see
   it as a couple of lost accuracy points you could never explain.

### Two matches are off-resolution

| match | clips | resolution |
|---|---|---|
| `vid121` | 25 | 598×360 |
| `vid22` | 2 | 896×494 |
| everything else | 10,064 | 896×540 |

27 clips out of 10,091. The resize to 224×224 absorbs this, but `vid121` is
upscaled from a smaller frame, so its crops will be softer. Not worth special
handling; worth knowing if those clips turn up in your error analysis.

### The split structure, measured

| | |
|---|---|
| distinct source matches | **534** |
| clips per match | 1 min, 18.9 mean, 53 max |
| matches with only one clip | 4 |
| **matches spanning more than one shot class** | **529 of 534** |

That last row is the one that matters. A `vidNNN` is a whole broadcast that
several different shots were cut from, so splitting by filename would scatter
one match across train, val and test almost every time. 534 groups is plenty to
split 70/20/10 cleanly — this is a constraint you can satisfy, not a problem.

### Most clips are really 24 unique frames

Measured 2026-10-09 over 400 random clips, in step 2.4. The 28- and 29-frame
clips are **24-frame clips with duplicates inserted** -- frame-rate conversion,
not extra content:

| frames | clips | duplicate frames | unique frames |
|---|---|---|---|
| 24 | 302 | 0% | 24.0 |
| 28 | 54 | **14%** | **24.2** |
| 29 | 1 | **17%** | **24.0** |
| 49 | 41 | 6% | 46.2 |

You can see it in the raw frame-to-frame differences -- every sixth frame is
near-identical to the one before:

```
 3-> 4  0.35     9->10  0.16    15->16  0.13    21->22  0.75
```

So the real story is simpler than "lengths vary from 22 to 49": **nearly all of
the dataset is 24 unique frames**, and only the 49-frame clips are genuinely
longer. `sample.py` must drop duplicates before sampling (Day 4.1), or it
spends its 15 slots on repeats.

### 29% of clips contain a camera cut, always near the end

Same 400-clip sample:

| | |
|---|---|
| clips with a camera cut | **29%** |
| clips with none | **71%** |
| cuts falling in the first half of a clip | **0** |
| median position of the first cut | **88% through** |
| cut rate among the 49-frame clips | **41%** |

**The shot itself is never interrupted.** Not one cut in 400 clips landed
before the halfway point. What the cut marks is the broadcast moving on after
the shot: following the ball to the boundary, panning to the crowd, cutting to
a fielder.

**What is after the cut, measured.** Run the authors' Striker detector over 30
clips with a cut and 20 without, 2026-10-09:

| | striker found in |
|---|---|
| clips with no cut (control) | **94.4%** of frames |
| cut clips, frames **before** the cut | **90.4%** |
| cut clips, frames **after** the cut | **71.0%** |

The control at 94.4% is what makes the rest trustworthy -- the detector is
reliable when the batter is there.

**This corrects an earlier claim.** Two clips were examined by eye
(`Cover Drive/vid1_0.avi` cuts to a wide stadium shot, `Sweep/vid306_8.avi`
follows the ball into the crowd) and both had no batter after the cut, from
which it was wrongly generalised that post-cut frames contain no batter. They
usually do: only **4 of 30** cut clips (13%) have no striker at all after the
cut, and 9 of 30 (30%) are more than half striker-free. Most cuts are to
another angle that still shows the batter, not to the crowd.

So the reason to trim at the cut is **not** "there is no batter there". It is
that the footage after a cut is a different camera angle on a shot that has
already finished -- a discontinuity handed to a model whose entire job is
reading motion through time. That is still a good reason, but a weaker and
different one, and it makes trimming a judgement call rather than an
obligation.

This matters in two places: `crop.py` (Day 3.3) must decide deliberately what
to do on a frame with no striker, and `sample.py` (Day 4.1) should trim at the
cut rather than sampling across it.

### The class imbalance

| | clips |
|---|---|
| Lofted Legside | 1,123 |
| Square Cut / Lofted Offside | 1,023 each |
| … | … |
| Scoop | 279 |
| Late Cut | 306 |
| **Reverse Sweep** | **252** |

**4.5:1** between the largest and smallest class. Keep this in mind twice: when
reading a confusion matrix (the small classes will look worse partly because
they are small), and on Day 17, where class-balanced loss is one of the tuning
axes.

---

## 6. Two splits, because the authors' protocol has a gap

Checked 2026-10-08 against the paper itself, not assumed. This is the one
finding that changes what gets measured, so it is worth reading carefully.

### What the authors actually did

Their whole protocol, quoted:

> "The test sets were manually constructed, comprising 20% of the total videos.
> Care was taken to ensure that **similar types of shots from the same batter**
> do not overlap between the training and test sets. From the remaining videos,
> **20% were taken randomly** for the validation test. Stratified splitting was
> applied to ensure that the class distribution is consistent across all splits."

"Videos" means clips: the paper says "10,086 videos" for the dataset, and says
"536 **match** videos" when it means matches. So **the split is at clip level**,
constrained by batter and shot type, and validation is unconstrained.

### What that rule does and does not prevent

| case | blocked? |
|---|---|
| Same batter, same shot type | yes |
| Same batter, *different* shot type | **no** |
| Same match, different batter | **no** |
| Same ground, lighting, camera, broadcast graphics | **no** |

Measured on your copy of the data: under a clip-level 20% test split,
**489 of 534 matches (91.6%)** are expected to have clips in both train and
test. A 19-clip match straddles with probability 0.986, and the mean match has
18.9 clips.

**Confirmed on the generated splits, 2026-10-12** — and this table is exactly
right, which is worth saying because the obvious suspicion turns out to be
wrong:

```
val clips whose source video also appears in train   1612 of 1614  (99.9%)
val clips with the same source video AND same class     0 of 1614  ( 0.0%)
```

So `splits_author` groups by **(video, class)**, blocking the leak that would
inflate a score — a Cover Drive in val never has its own video's Cover Drives
in train — while leaving stadium, lighting and broadcast style shared. That is
the "different shot type" row above. **Do not claim their 89.09% is inflated by
near-duplicate leakage; it is not, and the measurement says so.** `splits_grouped`
(0 shared video ids anywhere) remains the stricter protocol and the honest
number, but the gap between the two will be smaller than this section's framing
might suggest.

### Being fair to them

They are not careless about leakage. On augmentation they are stricter than most
published work:

> "It is crucial to perform augmentation **after** splitting the dataset...
> if augmentations are applied before the split, similar synthetic samples may
> appear across different sets. This scenario is known as data leakage."

They also criticise earlier work for precisely this, suggesting it "may have
contributed to the reported 93% accuracy of the VGG16-GRU model".

So they understood the problem, controlled its sharpest form -- a model
memorising one batter's particular stroke -- and hand-built the test set instead
of randomising it. They simply did not go as far as match-level grouping, and
applied nothing at all to validation. Match-level leakage is softer than
batter-level: it hands the model context (this ground, this broadcast style)
rather than the answer. **So 89.09% is likely a little optimistic relative to
strict grouping, by an amount nobody has measured.** That is not an accusation;
it is an open question, and you are in a position to answer it.

### So: build both, and run everything twice

```
data/splits_author/     clip-level, stratified, batter-aware -- their protocol
data/splits_grouped/    strict match grouping -- no vidNNN in two splits
```

| split | what it is for |
|---|---|
| **author** | the Day 7 number that is **comparable to 89.09%** |
| **grouped** | the honest number, and the one to believe |

Phase 1 reproduces their result **under their protocol**. That matters more than
it looks: if you grouped strictly and landed on 84%, you could not tell whether
your pipeline was broken or their number was inflated. Those two explanations
look identical, and separating them would cost days. Reproducing their protocol
first removes that ambiguity permanently.

Then every experiment runs on both, and the gap between the two numbers is a
result in its own right.

### Why running all 46 twice is affordable

**A frozen backbone's features depend only on the clip, not on the split.** One
feature cache serves both; changing the split is a different list of filenames
over the same files.

| family | n | h each | 1 split | both splits |
|---|---|---|---|---|
| A temporal heads | 14 | 0.15 | 2.1 h | 4.2 h |
| D classical ML | 6 | 0.08 | 0.5 h | 1.0 h |
| V video-pretrained | 10 | 0.05 | 0.5 h | 1.0 h |
| B CNN backbones | 8 | 3.0 | 24 h | 48 h |
| C 3D / video | 8 | 4.5 | 36 h | 72 h |
| **total** | **46** | | **63 h** | **126 h** |

The 30 cached experiments cost **3.1 extra hours** for 30 extra rows. The 16
end-to-end ones cost 60 extra hours, which is the real price of this decision.

**Family A's `h each` is 0.15, not 0.06, and that is now a measurement.** The
`flat` view took **70 minutes** a run on 2026-10-15 -- 2 min an epoch over 34
epochs, because a `62720 x 384` GRU kernel is 24.1 M parameters and an epoch
reads 11.0 GB off disk. The 0.15 h assumes the `avg` view for the heads that do
not need the grid; on `flat` alone Family A would cost 16 hours, not 2.1.
**A14 exists partly to find out which heads can use `avg`.**

### Run the cached families first, and here is why

It is not just about getting easy rows early. The 30 cached experiments
**measure whether the leak gap depends on the architecture**:

- If the gap is stable across all 30 -- say 6.2% ± 0.8% whatever the head -- it
  is a property of the **data**, not the model. That is a clean, defensible
  finding about a new benchmark.
- If it swings by architecture -- if 3D models exploit match context much harder
  than frozen-feature models do -- that is a **more interesting** result, and it
  tells you the expensive families genuinely need both splits rather than a
  correction factor.

Either way you learn it from the cheap runs before committing 60 hours to the
expensive ones.

---

# PHASE 1 — Build it, and reproduce the authors (Days 1–7)

Target: **89.09% ± 1.5%**, from code you wrote. Nothing else starts until this
lands.

> **Closed 2026-10-13 at 78.14%, not 89.09%.** The target was not reached and
> the reason is documented in 7.4 and 7.5: every cause the plan lists was
> eliminated by measurement, and the authors' own released model scores 43% on
> our best crops. **Phase 2 starts anyway** — the baseline needed to be fixed
> and honest, not equal to theirs, and nothing in Phases 2–4 depends on the
> absolute number. Read 7.4 before attempting any further preprocessing work.

Two days longer than the original plan, because building from nothing with the
code explained is slower than editing what was there. The two days come out of
Phase 3, where Family E joins F and W as stretch work — that still leaves **46
committed experiments**, well past the 30 you need.

## Day 1 — Foundations: device, seed, splits

Nothing trains today. Today is the ground every later number stands on.

### 1.1 Archive the old tree and lay out the new one (≈15 min)

**Why.** Code you do not trust must not be importable, or one stray
`from src.train import ...` quietly puts you back where you started.

**Build.** The `git mv` commands in [section 4](#4-starting-fresh), then the
empty package folders with their `__init__.py` files.

**Code to understand.** What `__init__.py` is actually for, and why
`python -m src.utils.splits` behaves differently from
`python src/utils/splits.py` — the second one cannot see `src` at all.

**Run.**

```bash
mkdir -p archive
git mv src archive/src_old && git mv configs archive/configs_old
git mv experiments archive/experiments_old
mkdir -p src/utils src/data src/preprocess src/models tests notes
git mv archive/src_old/utils/checkpoint.py src/utils/checkpoint.py
python -c "import pathlib; [pathlib.Path(p,'__init__.py').touch() for p in ['src','src/utils','src/data','src/preprocess','src/models']]"
```

> **If you see `fatal: bad source, source=src`** it means this already ran and
> there is nothing left to move — `git mv` resolved the destination as
> `archive/src_old/src` because `archive/src_old` now exists. Check with
> `git status`: staged `R src/... -> archive/src_old/...` lines mean it worked.
> Running it twice is harmless.

**Check.** `python -c "import src"` works. `python -c "import src.train"` fails
— there is nothing there yet, which is correct.

**Takeaway.** A clean import path is not housekeeping. Half of "it worked
yesterday" is really "it imported something else yesterday".

### 1.2 `device.py` — run on this GPU and on the other one (≈25 min)

**Why.** You have an Intel Arc B580 here and an RTX 3070 elsewhere. Code with
`"cuda"` written into it runs on neither reliably. One function, used
everywhere, and no script ever names a backend.

**Build.** `src/utils/device.py` with `get_device()` and `get_amp_settings()`.

**Code to understand.**

- The order **XPU → CUDA → CPU**, and why it is a try-list rather than a flag.
- **Autocast** in one sentence: the GPU does the heavy multiplications in a
  smaller number format, so they run faster and use less memory, while the
  weights stay full-size.
- Why Arc uses **bf16 with no GradScaler** and the 3070 uses **fp16 with one**.
  bf16 has the same range as fp32 so gradients do not vanish to zero; fp16 has a
  much smaller range, so it needs a scaler to multiply the loss up before the
  backward pass and divide it back after. Using a scaler on bf16 is not a small
  inefficiency — on XPU it errors out.

**Run.**

```bash
python -c "from src.utils.device import get_device, get_amp_settings; d=get_device(); print(d, get_amp_settings(d))"
```

**Check.** Prints `xpu:0` and a bf16 setting with the scaler off. If it prints
`cpu`, stop here — every later timing in this plan would be wrong.

**Takeaway.** Write the capability check once, call it everywhere. This is the
whole reason the same repo will run on Kaggle in Phase 3 without edits.

### 1.3 `seed.py` — make a run repeatable (≈15 min)

**Why.** Two runs of the same experiment must give the same number, or you
cannot tell a real improvement from luck. With 46 experiments coming, that
distinction is the entire point.

**Build.** `src/utils/seed.py`: one `set_seed(n)` that seeds Python, NumPy and
torch together.

**Code to understand.** Why three separate seeds are needed (three independent
random number generators are in play), and why a DataLoader with
`num_workers>0` needs a `worker_init_fn` as well — each worker is a fresh
process with its own generator.

**Run.**

```bash
python -c "
from src.utils.seed import set_seed; import random, torch
set_seed(42); a=(random.random(), torch.randn(1).item())
set_seed(42); b=(random.random(), torch.randn(1).item())
print(a); print(b); print('identical:', a==b)"
```

**Check.** `identical: True`.

**Takeaway.** Determinism is a property you build in on day one, not something
you add once results look suspicious.

### 1.4 Look at the filenames before writing any code (≈10 min)

**Why.** Every split decision depends on what a filename actually tells you.
Guessing the scheme and finding out on Day 15 that you guessed wrong would
invalidate everything in between.

**Build.** Nothing. Look at the data first.

**Run.**

```bash
ls data/CricShoot10kShootDataset | head
ls "data/CricShoot10kShootDataset/Cover Drive" | head
```

**Check.** Names look like `vid100_41.avi`, and there are 15 class folders
holding 10,091 clips between them.

**Takeaway.** `vid100` is the **source match video**; `_41` is the 41st shot cut
out of it. Many clips share one `vidNNN`, and clips from one match share the
batter, the kit, the ground, the lighting and the camera angle. That is exactly
the problem the next step exists to prevent.

### 1.5 `match_id`, and a test for it (≈20 min)

**Why.** If `vid100_41.avi` lands in train and `vid100_42.avi` lands in test,
the model can score well by recognising *that match* rather than the shot. The
test number comes out high and means nothing. That is a **leak**: information
from the test set reaching the model during training.

**Build.** One small function — `match_id("vid100_41.avi") -> "vid100"` — and a
test covering names that do not fit the pattern.

**Code to understand.** Why this is a named, tested function instead of an
inline `name.split("_")[0]`: the inline version appears in four places by
Phase 3, and the day a filename breaks the pattern, three of them are wrong.
Also what the test is really for — it documents what you decided to do about
the odd names.

**Run.**

```bash
python -m pytest tests/test_splits.py -q
```

**Check.** The test passes, including the odd filenames.

**Takeaway.** Two ideas that pull in opposite directions:

- **Stratified** splitting keeps the same proportion of each shot class in
  train, val and test. You want this.
- **Grouped** splitting forces every clip sharing a `vidNNN` into the *same*
  split. You need this more.

A leak inflates your score and invalidates the result; slightly uneven class
proportions only add a little noise. So grouping wins the argument.

### 1.6 Generate both split sets and prove the grouped one has no leak (≈45 min)

**Why.** These files are frozen for the whole project. Every experiment uses
them, or results cannot be compared with each other. There are **two** sets, for
the reason in [section 6](#6-two-splits-because-the-authors-protocol-has-a-gap).

**Build.** `src/utils/splits.py`, producing both:

- `data/splits_author/` — clip-level, stratified by class, and refusing to put
  the same `(batter, shot class)` pair on both sides. This reproduces the
  authors' described protocol. Batter identity is not labelled in the dataset,
  so the closest available proxy is `(vidNNN, class)`; say so in the write-up
  rather than implying you had batter labels.
- `data/splits_grouped/` — strict match grouping, no `vidNNN` in two splits.

**Code to understand.** Why perfect stratification is impossible once you group
(a whole match moves at once, so class counts move in lumps), and the greedy
rule we use: take matches in order of size, and put each one in whichever split
is furthest below its target. Also why the seed goes in the filename, and why
one function emits both sets rather than two scripts that could drift apart.

The numbers you are working with, measured not assumed: **534 source matches**,
18.9 clips each on average, up to 53 for the largest. 529 of those 534 span more
than one shot class, which is what makes grouping unavoidable — a match is a
broadcast, not a shot type. Four matches have only one clip.

**Run.**

```bash
python -m src.utils.splits --data_root data/CricShoot10kShootDataset \
    --out data/splits --test_frac 0.2 --val_frac 0.2 --seed 42
```

**Check.** Both sets: three CSVs each, no filename in two splits, class
proportions within a couple of percent. Then the one that separates them:

- `splits_grouped/` — **no `vidNNN` appears in two splits.** This must hold
  exactly.
- `splits_author/` — this will be **violated for most matches**, and that is
  the point. Print the number; expect roughly 490 of 534 matches straddling
  train/test, which matches the 91.6% predicted in section 6. If it comes out
  near zero you have accidentally built two grouped splits and the comparison
  measures nothing.

**Takeaway.** Commit them immediately. Regenerating splits later, even with the
same seed, silently makes earlier results incomparable — the kind of mistake a
results table cannot show you.

---

## Day 2 — Surviving a power cut, and reading video

### 2.1 Read `checkpoint.py` together (≈30 min)

**Why.** This file already exists and is tested, but it was written in one jump.
You cannot defend code you have not read.

**Build.** Nothing new. We read `src/utils/checkpoint.py` and
`tests/test_checkpoint.py` together, and you change anything you disagree with.

**Code to understand.**

- **Why `os.replace` and not a plain write.** `torch.save` streams hundreds of
  MB. Lose power halfway and the file is truncated garbage — so the cut costs
  you the *good* model you had already earned. Writing to `best.pt.tmp` and then
  renaming means the old file stays whole until the new one is complete, and the
  rename is a single atomic operation.
- **What has to be in a resume checkpoint, and why weights are not enough.**
  Adam keeps a running average of recent gradients for every parameter; the
  scheduler keeps its position; early stopping keeps a counter. Restart from
  weights alone and you get a *different* run that happens to start warm.
- **Why the ledger is JSON Lines.** Appending one line needs no read of the
  file, so a cut can damage at most the last line. A JSON array would have to be
  rewritten whole every single time — exactly the operation a power cut
  destroys.
- **The bug the test found.** A cut mid-append leaves a line with no newline, so
  the next append glues a good result onto a damaged one.

**Run.**

```bash
python -m pytest tests/test_checkpoint.py -q -v
```

**Check.** Seven tests pass. Read the names — they say what is guaranteed.

**Takeaway.** The strong test here trains a model 6 epochs, then trains it 3
epochs, throws the objects away, reloads, trains 3 more, and demands the weights
match **bit-for-bit**. A second test proves the first can fail. A test that
cannot fail proves nothing.

### 2.2 Prove the ledger works before you depend on it (≈15 min)

**Why.** 46 experiments are coming. Finding out in week three that rows were
being lost is not recoverable.

**Build.** Nothing new — `append_result` and `read_results` already exist.

**Run.**

```bash
python -c "
from src.utils.checkpoint import append_result, read_results
append_result({'run':'smoke','val_acc':1.23}, 'experiments/results.jsonl')
print(read_results('experiments/results.jsonl'))"
```

**Check.** One row comes back, with a `recorded_at` timestamp added. Then delete
the smoke row by hand before any real run.

**Takeaway.** Test the recording mechanism with throwaway data *before* it holds
something you cannot re-measure.

### 2.3 `video.py` — an `.avi` into an array (≈30 min)

**Why.** Everything downstream is "frames in, frames out". This is the one place
that touches a video file.

**Build.** `src/data/video.py` with `read_clip(path) -> array (T, H, W, 3)`.

**Code to understand.**

- The OpenCV read loop, and why you must check the returned `ok` flag rather
  than trusting `CAP_PROP_FRAME_COUNT`.
- **Why this function must not assume 24 frames.** The paper says 24; the
  released data disagrees. Only 7,657 of 10,091 clips have 24 frames — 1,390
  have 28, 956 have 49, and a few have 22, 23 or 29 (see
  [section 5](#5-what-the-dataset-actually-contains)). A hardcoded 24 would
  silently truncate a quarter of the dataset.
- **BGR vs RGB.** OpenCV returns Blue-Green-Red; every ImageNet-pretrained model
  expects Red-Green-Blue. Get this wrong and the model still trains, just a few
  points worse — which is why it is such a common silent bug.
- Why the function returns a NumPy array and not a tensor: it runs inside
  DataLoader worker processes, where a tensor would have to be pickled.

**Run.**

```bash
python -c "
from src.data.video import read_clip
c = read_clip('data/CricShoot10kShootDataset/Cover Drive/vid1_1.avi')
print(c.shape, c.dtype, c.min(), c.max())"
```

**Check.** `(24, 540, 896, 3) uint8 0 255` for that clip — then try one from
`vid121` and one 49-frame clip, and confirm the function returns their real
length instead of 24.

**Takeaway.** Shapes get stated out loud every time from here on. Shape
confusion causes more deep-learning bugs than algorithms do — and a shape you
assumed from a paper rather than measured is the worst kind.

### 2.4 Actually look at the frames (≈20 min)

**Why.** You are about to build a cropper. You cannot judge a crop without
knowing what the uncropped frame looks like.

**Build.** A few lines that save a grid of frames to a PNG.

**Run.** Save a grid of every frame of one clip, for three different classes,
and open them.

**Check.** You can see the batter, and you can see how small they are in a
896x540 broadcast frame.

**Takeaway.** This is the 75% problem, visible. The striker is maybe 10% of the
pixels; the rest is crowd, grass and sponsor boards, and a model with no crop
spends most of its capacity there.

---

## Day 3 — Finding the striker

### 3.1 YOLO on a single frame (≈30 min)

**Why.** Before cropping 10,091 clips, see what the detector actually returns
on one frame.

**Build.** `src/preprocess/detect.py`, loading YOLOv11 from
`CricShoot10kModels/` and running it on one image.

Two notes, checked 2026-10-09. `Player_Type_Detection_Model.pt` and
`Striker_Bat_Segmentation_Model.pt` are **byte-identical** (same MD5) -- one
model shipped twice, classes `{0: Striker, 1: Bat}`, task `segment`. Load one.
And it runs on the Arc: `model.predict(frame, device=str(get_device()))`.

**Code to understand.** What a detection *is* — a box as four numbers, a class
id and a confidence — and what the confidence threshold does. Also how to put
the model on the Arc GPU, and how to tell it actually went there rather than
silently staying on CPU.

**Run.** Detect on one frame, print every box with its class and confidence,
then draw them and look at the image.

**Check.** More than one person is detected — the striker, the keeper, the
umpire, sometimes fielders. **That is the real problem**, and it is why 3.2
exists.

**Takeaway.** "Detect the batter" is not a detection problem, it is a *selection*
problem. The detector gives you candidates; choosing among them is your logic,
and it is where the accuracy actually comes from.

### 3.2 Choose the striker among the candidates (≈45 min)

**Why.** Crop the wrong person and the clip is worse than useless — it is a
confidently mislabelled training example.

**Build.** `src/preprocess/striker.py` — `pick_striker(dets, shape)`, 54 lines,
13 tests. Returns `None` rather than a guess: 9% of frames contain no batter
at all, and on 6% of multi-striker frames the detector misses the batter even
though he is there.

> **The rule this plan originally proposed is wrong.** Measured 2026-10-09 on
> `Sweep/vid306_8.avi` before writing any of 3.2. Frame 5 has **6 detections**
> for one batter and one bat — 3 Strikers, 3 Bats:
>
> ```
> Bat      0.55  centre ( 867, 165)   1.7% of frame
> Bat      0.54  centre ( 211, 259)   1.6%
> Striker  0.45  centre ( 390, 221)  10.0%   <- the real batter
> Bat      0.39  centre ( 218, 261)   2.7%
> Striker  0.30  centre ( 865, 128)   3.2%   <- a fielder, detected twice
> Striker  0.30  centre ( 872, 112)   2.2%
> ```
>
> | rule | picks | right? |
> |---|---|---|
> | largest striker box | batter | yes |
> | highest-confidence striker | batter | yes |
> | nearest the centre of frame | batter | yes |
> | **nearest the best bat** | **fielder** | **no** |
>
> The highest-confidence detection in the frame is a Bat at 0.55 sitting beside
> that fielder, 480 px from the batter. Frame 10 fails the same way: its bat is
> 115 px from the wrong person and 147 px from the right one.
>
> The reason is structural, not bad luck. A bat is 1.7% of the frame, so bat
> detection is *less* reliable than person detection — "nearest the bat" chains
> the weaker signal in front of the stronger one and inherits its errors.

**Settled 2026-10-09**, by labelling 36 random multi-striker frames from the
train split by eye and scoring every candidate rule (a pick correct at IoU >=
0.5, so a duplicate box of the same person is not punished):

| rule | correct |
|---|---|
| **nearest the frame centre** | **33/34 = 97%** |
| nearest the best bat | 30/34 = 88% |
| largest box | 26/34 = 76% |
| highest confidence | 23/34 = 68% |

So the simplest rule wins outright, and adding area and confidence on top of
centrality makes it *worse* (32/34). Over 720 frames from 120 clips, **51% have
two or more Strikers**, so this choice is made on half of all frames, and
largest-box and highest-confidence disagree on 45.8% of them.

Centrality wins for a reason that is worth stating in the thesis: the true
batter's box centre sits at **x = 0.44 to 0.54** of frame width in 80% of
frames. CricShot10k is already built around the batter. That is a property of
the dataset, not of cricket, and the rule would not transfer to uncropped
broadcast footage.

**Code to understand.** Why the one-line rule beat the clever ones, and the
difference between a rule that is right and a rule that is right *for a reason
you can state*. Also the plausibility filter — the detector calls the station
logo and the scorebar people, and both sit against the frame border.

**Known failure.** F19 of the labelled set: the batter is 131 px from centre, a
fielder 113 px. 18 px decided it. Only 3 of 34 frames are decided by under 20
px, median margin 286 px, so most picks are not close.

**Not done.** 34 frames is small. The 97%-to-76% gap is far too large to be
noise; the 97%-to-94% gap is not, so no choice was made on it. A second
labelled batch before Day 4 would firm up the number.

**Run.** Run the selection over 20 frames from different classes, draw the
chosen box, and look at all 20.

**Check.** You judge these by eye, not by a number. Count how many are right.
Below about 18 of 20, fix the rule before going further.

**Takeaway.** Three hours of looking at crops now is cheaper than a wasted
8-hour preprocessing run and a day of confusion afterwards.

### 3.3 `crop.py` — one clip, end to end (≈30 min)

**Why.** A clip is not 24 independent frames. A box that jumps around between
frames makes the shot look jittery to the temporal model.

**Build.** `src/preprocess/crop.py`: detect per frame, select, then **smooth the
boxes across the clip** before cropping.

**Code to understand.** Why one box for the whole clip is too coarse (the batter
moves) and a per-frame independent box is too noisy; the middle ground is a
running average, plus padding around the box so the bat does not get cut off.

Then the case that is now known to be common: **frames with no striker at
all.** 29% of clips end with a camera cut after which there is no batter in
shot (section 5). Carrying the last good box over those frames crops empty
grass or crowd and calls it a cricket shot. The options are to carry the box,
to drop the frame, or to stop the clip at the cut — decide deliberately,
record which, and prefer stopping at the cut, because the frames after it are
not part of the shot.

**Done 2026-10-09.** `src/preprocess/crop.py`, 148 lines, 19 tests. Both
thresholds measured rather than guessed:

- **outlier cutoff 1.0 body-heights** from the clip's median centre. Strays are
  median 0.21, 90th pct 0.76, 99th 2.30, so 1.0 rejects 8.0% — which matches
  the independently measured 9.3% of frame-to-frame jumps over 150 px.
- **padding 25%**, measured against the square crop actually taken (squaring a
  tall person box already adds width where the bat is):

| padding | bat stays whole | batter fills crop |
|---|---|---|
| 0% | 37% | 100% |
| 20% | 77% | 71% |
| **25%** | **84%** | **67%** |
| 40% | 93% | 56% |

**Validation that was not circular.** Two clips whose cuts were found on Day 2
by frame differencing come out right without being told: `vid1_0.avi` 29 -> 20
frames (cut measured at 20) and `vid306_8.avi` 24 -> 16 (cut measured at 16).

**Run.** Crop one clip, save the cropped frames as a grid.

**Check.** The batter stays roughly centred and roughly the same size across all
the clip. Jitter means the smoothing is not working.

**Gap policy, fixed the same day.** The first version called any gap over two
frames a camera cut and dropped 7% of clips. Measured over 60 clips: of the
gaps 3+ frames long, only **21% sit next to a real cut**; the other 79% are the
detector failing mid-shot, worth **230 frames against 48**.

So `src/preprocess/cuts.py` (`find_cuts`, `shots`, 11 tests) splits the clip at
its cuts, `crop_clip` keeps the shot with the most detections, and gaps inside
that shot are interpolated whatever their length.

| | before | after |
|---|---|---|
| median frames kept | 20 | **22** |
| 10th percentile | 10 | **15** |
| clips dropped entirely | 7% | **2%** |
| clips with 15+ frames for Day 4 | 75% | **90%** |

The cut threshold changed slightly (drop the top decile before taking the
median, so cuts cannot set the bar they are measured against). On 120 real
clips old and new agree on 117, and Day 2's findings are reproduced exactly:
first cut at median 88% through, none in the first half.

**Crop size comes from box height**, smoothed over 9 frames, not from a
clip-wide constant. Height varies x1.3 within a clip against width's x2.3 --
width is the bat leaving his outline, height is the camera. A constant size
clips the bat off when the broadcast zooms in (`Flick/vid116_8.avi`).

**Temporal link, done the same day.** The crop walking off the batter was
reported as 0.4% of frame *pairs*, which undersold it: a switch ruins a run of
frames, and at frame level it was **2.2% of frames across 22% of clips**.

`track_boxes` now runs two passes — centrality, then a reference track built
from the frames that agree with each other, then a re-pick against it. Two
passes rather than chaining off the previous frame, because a chain lets one
bad frame poison everything after it.

| | start | two passes | + no end-hold |
|---|---|---|---|
| frames cropping the wrong person | 2.2% | 1.5% | **0.6%** |
| clips affected | 22% | 17% | **8%** |
| clips with 15+ frames for Day 4 | 90% | 92% | **92%** |

**The stopping rule is worth copying.** After two passes the remainder looked
tunable, so instead of tuning I asked whether it was fixable at all: of the 21
frames still wrong, **0 had a correct box available and 21 had none**. Nothing
was left for selection to do. The last change was therefore to stop guessing
— no reference outside the agreed stretch, so a frame showing only the bowler
is trimmed instead of cropped onto him.

**Takeaway.** Temporal consistency is a requirement of the preprocessing, not
just of the model.

### 3.4 Eyeball 30 clips before committing to 10,091 (≈45 min)

**Why.** The full run takes hours. Finding a systematic crop error afterwards
means doing all of it again.

**Build.** Nothing — run 3.3 over two clips per class and look at every one.

**Done 2026-10-09.** 15 classes x 2 clips x 3 moments = 90 frames, both crop
variants. **29 of 30 clips are right**: batter centred, bat in frame, shot
readable. The exception is `Defensive/vid239_7.avi`, whose final frame zooms in
far enough to show a fielder's legs. End frames run tighter than start frames
across the board, because crop size follows box height and broadcasts zoom in
during a shot.

**The segmented variant exists now too.** The authors' detector is a
*segmentation* model (`task: segment`, mAP50(M) 0.897) and their classifier is
named `..._NEEDS_CROPPED_SEGMENTED_SHOTS.keras`, so background removal is
probably part of the recipe we were missing. `detect(masks=True)` +
`crop_clip(segmented=True)` blacks out everything but the batter and his bat.
Free -- same model, same inference, masks we already pay for.

**Settled.** An interpolated frame has no mask (9.6% of frames, 57% of clips).
Tested on 463 frames that *do* have one, by pretending they do not: carrying
the nearest real silhouette scores **IoU 0.80** one frame away and 0.65 at
three, against **0.51** for filling the bounding box in as a rectangle. So
`carried_mask()` carries, from whichever side is nearer.

**Run.** 30 clips, 30 grids, opened and examined one by one.

**Check.** Write down the count of good crops out of 30, and **put that number
in the journal**. It is the honest baseline for everything that follows.

**Takeaway.** The authors' ablation says crop-plus-segmentation is worth 14
points. That upside only exists if the crops are actually right, and the only
way to know is to look.

---

## Day 4 — The rest of the pipeline, then run it

### 4.1 `sample.py` — any length down to 15 (≈30 min)

**Why.** The authors use 15 frames, and 15 is what their 89.09% was measured
with. But clips are **not** all 24 frames long — they run from 22 to 49,
depending on the source match — so this has to be `T → 15` for any `T`, not
`24 → 15`.

**Build.** `src/preprocess/sample.py`, taking the real length as input.

**Code to understand.** Uniform sampling against taking the first 15, and why
uniform wins — a cricket shot's information is in the swing, which is in the
middle. The off-by-one in `linspace`-style index picking, which is the classic
bug here. And the case that matters most: a 49-frame clip and a 24-frame clip
must both come out as 15 frames covering **the whole shot**, so the step size
has to come from the clip, never from a constant.

**Two things have to happen before the sampling**, both from section 5:

1. **Drop duplicate frames.** 14–17% of the frames in 28- and 29-frame clips
   are repeats from frame-rate conversion. Sampling across them spends slots on
   identical pictures. After de-duplication almost the whole dataset is 24
   unique frames, which also makes the clips far more uniform than they look.
2. **Trim at the camera cut.** 29% of clips end with footage that is not the
   shot. Cuts never occur before the halfway point, so trimming is safe; the
   detector is the same frame-difference measure used in section 5, about ten
   lines. A fixed "use the first half" rule would also be safe but throws away
   the follow-through on the 71% of clips that need no trimming at all.

Order matters: de-duplicate, then trim, then sample 15 from what is left.

**Run.** Sample a 24-frame clip, a 28-frame clip, a 49-frame clip and one with
a camera cut (`Sweep/vid306_8.avi`, cut at frame 16 of 24); print the chosen
indices for each.

**Check.** 15 indices every time; first and last of the *kept* frames always
included; no index lands on a duplicate or past a cut. For `vid306_8` every
index must be below 16.

**Done 2026-10-09.** `src/preprocess/sample.py`, 26 tests.

The duplicate threshold was measured, not chosen. Nothing is bit-identical --
the clips were re-encoded -- but over 150 clips there is an **empty gap**:
duplicate pairs top out at a mean absolute difference of **0.927**, real motion
starts at **1.045**. A threshold of 1.0 sits in the gap. Duplicates by length:
24-frame 0.4%, 28-frame **15.3%**, 49-frame 0.0%, reproducing Day 2 from a
different direction.

End to end, read -> dedupe -> detect -> crop -> sample:

```
clip            raw  dedup  kept  out
vid306_8.avi     24     24    16   15   every index < 16, the measured cut
vid1_0.avi       29     26    17   15   3 duplicates removed
vid368_14.avi    28     23    23   15   5 duplicates removed (18%)
vid317_9.avi     49     49    49   15   spread 0..48, not truncated
vid385_27.avi    49     49    32   15   trimmed 17 at a cut
```

**4.2 is already done** -- segmentation lives in `crop_clip(segmented=True)`,
so Day 4 is 4.1, 4.3 and the run.

**Storage, decided.** Raw uint8 `.npy`, one file per clip per variant.
Measured on real crops over 10,091 clips: raw **42.4 GB**, PNG 11.3 GB
(lossless), JPEG q95 3.0 GB, WebP q90 1.4 GB. Raw costs disk but the expensive
thing is the run, not the bytes -- raw and PNG convert to each other for free,
JPEG is one way. The smoke run over 20 real clips projects to 42.4 GB, within
1% of the estimate.

**4.3 done 2026-10-09.** `build_dataset.py`, 143 lines, 9 tests, most of them
about interruption rather than cropping: a second run redoes nothing, a run
stopped early resumes, a manifest line cut in half mid-write is skipped and
that clip redone, and one unreadable clip records its reason and the run
carries on. `_save` writes `.tmp` then `os.replace`.

**Measured rate: 0.5 clips/s, about 5.6 h** for the full dataset -- slower than
Day 3's 4 h because masks and a second crop pass were added since.

**Takeaway.** Resampling to a fixed length is also what *removes* the
frame-count difference between matches. After this step, clip length carries no
information about which match a clip came from — which is one shortcut the
model can no longer take.

### 4.2 `segment.py` — colour the striker and the bat (≈45 min)

**Why.** This is the second half of the authors' 14 points. It tells the model
where the batter *is*, instead of making it work that out from scratch.

**Build.** `src/preprocess/segment.py`, using YOLOv11x-seg: striker blue, bat
green, blended at alpha 0.5.

**Code to understand.** The difference between a **box** (four numbers) and a
**mask** (one yes/no per pixel), and why segmentation helps beyond cropping —
the crop removes the crowd, the mask separates the batter from the pitch inside
the crop. Then what alpha blending is, and why 0.5 rather than painting the
pixels solid: the texture underneath still carries information.

**Run.** Segment one cropped clip and save the grid.

**Check.** Blue on the batter, green on the bat, and the underlying image still
visible through both.

**Takeaway.** The authors did not explain why this works, so you can: it is a
cheap way to hand the model an attention map instead of asking it to learn one.

### 4.3 `build_dataset.py` — all of it, resumably (≈45 min)

**Why.** This is the hours-long job, and load shedding will interrupt it. It has
to continue rather than restart.

**Build.** `src/preprocess/build_dataset.py`: for each clip, crop → sample →
segment → save, **skipping any clip whose output already exists**.

**Code to understand.** The skip check and why it goes *first*, before any model
runs. Why failures are logged and counted instead of raising (one clip with no
detection must not kill a 6-hour job). And why the output is written atomically
— a half-written clip that *exists* would be skipped forever, which is worse
than one that is missing.

**Run.** First on 30 clips. Then kill it halfway with Ctrl-C and run it again,
and watch it skip.

**Check.** The second run reports skipping what the first one finished. This is
the single most important check of the week.

**Takeaway.** "Resumable" means the completion test is cheap and exact. Here it
is "does the output file exist", which costs nothing and cannot be wrong —
provided the write was atomic.

### 4.4 Launch the full run (≈20 min to start, then 4–8 h)

**Run.**

```bash
python -m src.preprocess.build_dataset --split all
```

**Check.** `data/processed/` fills up. Spot-check finished clips *while it runs*.
Expect a few hundred failures out of 10,091; read the failure log and make sure
they are not all from one class.

**Done 2026-10-10. 10,091 clips, 9,880 usable (97.9%), 211 failed, 4.53 h.**
Zero `read_clip` retries. 9,880 `.npy` files in each variant against 9,880 `ok`
rows, **no `.tmp` left behind**, 42 GB -- within 1% of the projection.

The failure rate *was* uneven and the cause turned out to be the footage.
Scoop 8.6%, Reverse Sweep 5.2%, Straight Drive 5.0%, against Hook 0.0% and
Pull 0.8%. Scoop's failures are 20 "too short" to 4 "too little detected", and
surviving Scoop clips keep a median of 19 frames against 22-23 elsewhere: a
scoop sends the ball over the keeper, the broadcast cuts away sooner, more of
the clip is correctly trimmed. The trim is right.

Effect on the imbalance, which is the thing that would matter:

```
before : 1123:252 = 4.46:1
after  : 1108:239 = 4.64:1
```

Negligible. One sentence in the thesis, not a re-run.

> **Day 5 depends on this.** The split CSVs still name all 10,091 clips, 211
> of which have no output. `dataset.py` must filter the splits against
> `data/processed/manifest.jsonl` or it will fail on a missing file.

**Takeaway.** While it runs, read the authors' notebook in
`data/unnecessary/cricshot10k-models.zip` and compare their preprocessing to
yours. Differences you find now are cheap.

---

## Day 5 — The dataset and the model

### 5.1 `dataset.py` — feed the GPU (≈40 min)

**Why.** A `Dataset` is the one object standing between files on disk and
tensors on the GPU.

**Build.** `src/data/dataset.py` reading the split CSVs from Day 1.

**Code to understand.** The three methods a `Dataset` needs and what each is
for; where normalisation belongs and why ImageNet mean/std specifically (the
pretrained weights were trained on inputs scaled that way, so anything else
shifts every feature); and the shape contract, stated out loud:
`(T, H, W, 3) uint8 -> (T, 3, 224, 224) float32`, which the DataLoader then
batches to `(B, T, 3, 224, 224)`.

**Run.**

```bash
python -c "
from src.data.dataset import CricShotDataset
d = CricShotDataset('data/splits_grouped/train.csv')
x, y = d[0]; print(len(d), x.shape, x.dtype, y)"
```

**Check.** `(15, 3, 224, 224) torch.float32`, a label in 0–14, and a length
matching the CSV.

**Takeaway.** Check a Dataset by indexing it directly, before a DataLoader is
anywhere near it. One `d[0]` finds more bugs than an epoch of training.

**Done 2026-10-10.** `src/data/dataset.py`, 104 lines, 14 tests. The check
passes: `6341 torch.Size([15, 3, 224, 224]) torch.float32 0`, values running
-2.12 to 2.39 with mean +0.121, which is what ImageNet normalisation looks
like (plain 0-1 scaling would be 0 to 1).

The 4.4 dependency is handled -- missing clips are dropped at construction and
listed in `.missing`, or `strict=True` raises. The arithmetic confirms it:

```
grouped  train 121 + val 43 + test 47 = 211
author   train 146 + val 30 + test 35 = 211
```

211 from both split sets, which partition the same clips differently, so that
is a check rather than a coincidence. Both variants hold the same clips
(6341/1569/1970), so box vs seg is a comparison on identical data.

### 5.2 `encoder.py` — the same CNN on all 15 frames (≈40 min)

**Why.** Each frame needs turning into a feature vector before anything temporal
happens.

**Build.** `src/models/encoder.py`: EfficientNetV2-S, ImageNet weights, applied
to every frame.

**Code to understand.** **TimeDistributed**, which sounds harder than it is: you
have `(B, T, 3, 224, 224)`, the CNN wants `(B, 3, 224, 224)`, so you fold time
into the batch — `(B*T, 3, 224, 224)` — run the CNN once, then unfold back to
`(B, T, features)`. One `reshape` each way, and the same weights see every
frame. Also why a loop over `T` would be far slower: it hands the GPU 15 small
jobs instead of one large one.

**Run.** Push one batch through and print the shape at each stage.

**Check.** `(2, 15, 3, 224, 224) -> (30, 3, 224, 224) -> (30, 1280) -> (2, 15, 1280)`.

**Takeaway.** "Apply a 2D model across time" is a reshape, not a new
architecture. That one idea covers most video models built on image backbones.

**Done 2026-10-10.** `src/models/encoder.py`, 76 lines, 15 tests. The check
passes on the Arc: `(2, 15, 3, 224, 224) -> (30, 3, 224, 224) -> (30, 1280) ->
(2, 15, 1280)`, 20.2 M trainable parameters.

Two things beyond the reshape. **The backbone is a parameter**, since Phase 3
swaps it ten times; torchvision puts the ImageNet head on `.classifier` for
efficientnet and `.fc` for resnet, so `_strip_classifier` finds whichever,
records its `in_features` and replaces it with Identity. Downstream reads
`encoder.out_dim`, never 1280. **A frozen backbone is kept in eval**, because
BatchNorm updates its running statistics in train mode with or without
gradients -- freezing the weights alone would still let the features drift,
which defeats Phase 3's frozen-feature runs.

### 5.3 `head.py` — GRU-128 and the classifier (≈30 min)

**Why.** 15 feature vectors have to become one prediction. The order matters:
that is what makes it a shot rather than a pose.

**Build.** `src/models/head.py`: GRU(128) → BatchNorm → Dense(1024, ReLU) →
Dense(15). **No dropout** — the authors have none, and we are reproducing them
first.

**Code to understand.** What a GRU does, in one sentence: it walks along the 15
steps keeping a running summary, and learns what to keep and what to forget.
Why we take the **last** hidden state (it has seen the whole shot). What
`logits` are — the 15 raw scores before softmax — and why the loss function
wants logits, not probabilities (softmax then log is numerically worse than
doing both at once).

**Run.** One batch through the head alone.

**Check.** `(2, 15, 1280) -> (2, 15)`.

**Takeaway.** Keep the encoder and the head as separate modules. Phase 3 swaps
one or the other, never both at once, and that is only easy if they were
separate from the start.

**Done 2026-10-10.** `src/models/head.py`, 68 lines, 17 tests. The check
passes on the Arc: `(2, 15, 1280) -> (2, 15)`, and end to end
`(2, 15, 3, 224, 224) -> (2, 15, 1280) -> (2, 15)`.

0.69 M parameters in the head against 20.2 M in the encoder -- nearly all the
capacity is in the frame encoder.

Three tests are worth copying elsewhere, because a broken head still returns
the right shape: reversing the clip must change the prediction, changing only
the *first* frame must change it, and the output must **not** sum to 1 (it is
logits; softmax belongs inside the loss).

> **Constraint for Day 6.** `BatchNorm1d` raises on a batch of one in training
> mode. Either drop the last batch or keep the batch size above 1, or a run
> dies at the end of an epoch whose last batch has a single clip.

### 5.4 One batch through the whole model (≈20 min)

**Why.** Before any training, confirm data flows end to end and gradients reach
the first layer.

**Build.** Nothing — wire encoder and head together.

**Run.** One forward pass, compute the loss, call `.backward()`, and print the
gradient norm of the first convolution.

**Check.** Loss near `ln(15) = 2.71` on an untrained model — that is what
"guessing uniformly among 15 classes" looks like, and a loss far from it means
something is wrong before training even starts. The gradient norm must be
non-zero.

**Takeaway.** `ln(num_classes)` is the number to expect from an untrained
classifier. Knowing it turns the first printed loss into a test.

**Done 2026-10-10.** Loss **2.6406** against ln(15) = 2.7081, gradient norm at
the first convolution **2.82**, and **0 of 460 parameter tensors without a
gradient**. `src/models/model.py` (36 lines, 8 tests) wires the halves
together -- slightly more than "build nothing", because Day 6 needs it and
`build_model()` sizes the head from `encoder.out_dim`.

> **The memory wall, found here.** The encoder sees **B x T images at once**,
> so a batch of 8 clips is 120 images through EfficientNetV2-S with
> activations kept for the backward pass. Batch 8 in fp32 is an OOM.
>
> | batch | images | fp32 | bf16 | bf16 step |
> |---|---|---|---|---|
> | 2 | 30 | 4.44 GB | 2.43 GB | 0.160 s |
> | 4 | 60 | 8.93 GB | 4.67 GB | 0.218 s |
> | 6 | 90 | 13.19 GB | **6.82 GB** | 0.353 s |
> | 8 | 120 | **OOM** | | |
>
> An XPU OOM also **poisons the context**: every later allocation in that
> process failed with `UR_RESULT_ERROR_OUT_OF_RESOURCES`. Day 6 must treat OOM
> as fatal -- checkpoint and exit, let resume restart it.

> **Phase 3 arithmetic.** At batch 6, 0.353 s a step, 6,341 train clips is
> ~1,057 steps, about **6 minutes an epoch**: 3 hours for a 30-epoch run and
> **~11 days of GPU for 92 runs**. Fine-tuning end to end every time is not
> affordable. The frozen-encoder rows are how most of that table gets filled.

---

## Day 6 — The training loop

### 6.1 `train_epoch` (≈40 min)

**Why.** This is the five lines everything else serves.

**Build.** One epoch over the DataLoader: forward, loss, backward, step.

**Code to understand.** Why `optimizer.zero_grad()` comes first (PyTorch
*accumulates* gradients by default, so skipping it silently sums across
batches); what `model.train()` changes and why it matters for BatchNorm; the
autocast block from Day 1.2 in its real place; and **gradient accumulation** —
how to get the effect of a larger batch than the GPU can hold, by stepping the
optimizer only every N batches.

**Run.** One epoch on 50 clips.

**Check.** The loss goes down. That is all you need today.

**Takeaway.** Train on a tiny subset first and make sure the model can
**overfit** it to near 100%. A model that cannot memorise 50 clips has a bug,
and you will find it in two minutes instead of two hours.

**Done 2026-10-10.** `src/engine.py`, 118 lines, 17 tests.

> **Take the subset from every class.** The first attempt used the first 50
> rows of the split CSV, which is ordered by class -- all 50 were Cover Drive,
> and "100% accuracy" only meant the model had learned to say one word. Four
> clips from each of the 15 classes instead:
>
> ```
> epoch 1   loss 2.7234   top-1  15.0%
> epoch 3   loss 2.1745   top-1  75.0%
> epoch 6   loss 1.5433   top-1 100.0%
> epoch 20  loss 0.2248   top-1 100.0%
> ```
>
> Epoch 1 at 2.7234 against ln(15) = 2.7081 is 5.4's anchor appearing again
> unprompted. 2.5 s an epoch at batch 6.

### 6.2 `evaluate` (≈30 min)

**Why.** Validation accuracy decides everything in Phase 3, so it must be
exactly right.

**Build.** The eval pass, returning top-1, top-2, top-3 and the loss.

**Code to understand.** `model.eval()` and `torch.no_grad()` — what each one
does and why you need both (one changes layer behaviour, the other stops
building the graph); and what top-k accuracy is, plus why top-3 is worth
recording for 15 confusable classes.

**Run.** Evaluate the untrained model on val.

**Check.** About 6.7% top-1 — that is 1/15, random guessing. Materially above
it, on an untrained model, means labels are leaking in somewhere.

**Takeaway.** Both of your "is this right?" anchors are now in place:
`ln(15)` for the loss and `1/15` for the accuracy.

**Done 2026-10-10.** `evaluate` and `topk_correct` in `src/engine.py`, 27
tests. On the real val set, untrained:

```
loss    2.7369   ln(15) = 2.7081
top-1    2.42%   chance  6.67%
top-2    6.37%   top-3  12.81%
```

> **Below chance is the right answer here, and the plan's "about 6.7%" is
> loose.** An untrained net is biased, not uniform: this one puts 65.6% of its
> predictions on Scoop and 27.6% on Reverse Sweep, the two *rarest* classes
> (2.6% and 2.4% of val). Expected top-1 for that bias is **2.80%**; measured
> 2.42%, within one standard error. The loss agrees -- 2.7369 is slightly
> above ln(15), which is what concentration costs.
>
> The test is **materially above chance**, which would mean a leak. A
> symmetric tolerance around 6.7% is the wrong check.

### 6.3 The loop, with resume wired in (≈40 min)

**Why.** This is where Day 2's crash safety earns its place.

**Build.** `src/train.py`: the epoch loop, best-checkpoint tracking, early
stopping, `save_resume` every epoch, and one ledger row at the end.

**Code to understand.** Where `save_resume` goes and why it is *before* any
`break`; why `resume.pt` is deleted when the run completes, so its presence
always means "interrupted"; and why resuming is the default rather than a flag
— when the power returns you want to retype the same command, not remember a
flag.

**Run.** Train 3 epochs on 50 clips. Kill it with Ctrl-C during epoch 2. Run the
same command again.

**Check.** It prints `RESUMED` and continues from epoch 2. **Do this
deliberately now**, while a lost run costs seconds.

**Takeaway.** Rehearse the recovery before you need it. An untested backup is
not a backup.

**Done 2026-10-10.** `src/train.py`, 260 lines, 11 tests. Rehearsed with a
**hard kill**, not Ctrl-C, since a power cut gives no chance to save on the
way out:

```
killed during epoch 4
resume.pt says epoch 3, best 10.00%, 3 epochs of history

... same command again ...
RESUMED from experiments\smoke2
esume.pt -- starting at epoch 4 with best 10.00%

epochs in history: [1, 2, 3, 4, 5, 6]
resume.pt after a clean finish: False
```

No gap, no repeated epoch, file cleaned up.

Two things folded in from earlier days: `--limit_clips` takes its subset
**across all classes** (6.1's lesson), and the processed-data directory is a
config key rather than a constant, so a second preprocessing run can be
trained against without editing code.

### 6.4 `configs/mimic_author.yaml` (≈30 min)

**Why.** Reproducing the authors means their settings, not ones that look
sensible.

**Build.** The config, matching their notebook exactly:

> EfficientNetV2-S (ImageNet, TimeDistributed) → Flatten → GRU(128) →
> BatchNorm → Dense(1024, ReLU) → Dense(15, softmax). **No dropout.** Adam
> lr=1e-4, batch 4, ReduceLROnPlateau(factor 0.1, patience 4), early stop
> patience 10, 100 epochs, horizontal flip only.

**Code to understand.** Why the config is data and not code; and three places
where their *paper* and their *notebook* disagree — batch 4 not 8, 100 epochs
not 50 — with the notebook winning, because that is what produced the weights.

**Check.** Every value in the config traces to their notebook. Note any you had
to guess, in the journal.

**Takeaway.** "Reproduce" means their numbers, including the ones you would have
chosen differently. Your own choices come after the baseline lands.

**Done 2026-10-11.** `configs/mimic_author.yaml`.

> **There is no notebook in either archive.** The evidence is their shipped
> `.keras` file, which is better -- it is what produced the weights. Its
> `config.json` confirms no dropout, a **trainable** (not frozen) encoder,
> GRU(128), Dense(1024), Adam with no weight decay.
>
> **The saved learning rate is 1e-9**, which nobody starts from. It is where
> `ReduceLROnPlateau(factor=0.1)` lands after five cuts from 1e-4 -- so both
> numbers are corroborated by one they never wrote down.

> **The finding that made this real work.** Their GRU kernel is
> **(62720, 384)**: 384 is 3 gates x 128, and **62720 is 7 x 7 x 1280**. They
> do not pool -- the Flatten hands the GRU the whole spatial map. We were
> average-pooling to 1280, which discards *where* the bat is relative to the
> body.
>
> | | GRU params | model |
> |---|---|---|
> | theirs (flatten) | 24.13 M | 44.5 M |
> | ours (avg) | 0.54 M | 20.9 M |
>
> `FrameEncoder` now takes `pool="avg"|"flatten"`, and `out_dim` is measured
> by a dummy forward rather than derived. It returns 62720, matching their
> kernel exactly.

Also added for this config: `src/data/augment.py` (`ClipFlip`, one decision
per clip) and `ReduceLROnPlateau` stepped on validation top-1.

**Re-measured, because the architecture changed:** flatten at batch 4 is
**5.60 GB and 0.264 s a step**, so their batch fits and an epoch is about
**7 minutes**. 100 epochs is ~12 h before early stopping.

**Five values could not be read from the file** and say so in the config:
batch 4, 100 epochs, early-stop patience 10, plateau patience 4, and the flip.

---

## Day 7 — Train it, and hit the number

### 7.1 Launch the author replication (≈30 min to start, then 3–6 h)

**Run.**

```bash
python -m src.train --config configs/mimic_author.yaml
```

`python src/train.py` does **not** work -- the module imports `src.*`, which
needs the package on the path. Run it with `-m`.

**Measured before launching, 2026-10-11:** 6,320 train / 1,584 val clips at
batch 4 is 1,580 steps, **~9 minutes an epoch**, so 100 epochs is **15.5 h**
worst case. Early stopping with patience 10 should cut it well short.

**Check at epoch 5.** Val accuracy should already be past 60%. If it is stuck
near 7% the labels are shuffled.

**Measured, for comparison:** the real run reached **56.00% at epoch 1 and
62.06% at epoch 5**, peaking at 79.23% (epoch 35) after four lr reductions.
Roughly 7 minutes an epoch, 45 epochs before early stop.

> **The "75% means uncropped" rule has been withdrawn.** The cropped run
> passes straight through 75% around epoch 20 on its way to 79%, so that
> check now fires on healthy runs. Use the epoch-1 value instead: **56% at
> epoch 1 is right; ~15% means something is broken**, and what it usually
> means is below.

**The two checks that actually matter, both added after they cost real runs:**

| column | healthy | broken |
|---|---|---|
| `pin` | 0–2% | 40%+ means the GRU is saturating — BatchNorm eps or the init |
| `gn` | 15–30, drifting down | hundreds, or climbing, means the backbone is blowing up |

`pin` is the fraction of GRU units pinned at tanh's rail, read free off the
BatchNorm running stats. It sat at 41% for the whole of a six-hour run that
should have been killed at epoch 2.

### 7.2 Horizontal-flip augmentation (≈1 h)

**Why.** It is in the authors' pipeline and worth about 1.5 points.

**Build.** Flip the **training split only**, *after* the split.

**Code to understand.** Why flipping after the split is essential: flip first
and a clip's mirror image can land in test while the original is in train — a
leak that is very hard to spot later. Also why flipping is safe for cricket
shots at all (it turns a right-hander into a left-hander, which is a real thing
that happens) while a vertical flip would not be.

**Done 2026-10-11, online rather than offline.** `src/data/augment.py`'s
`ClipFlip` is attached to the training `Dataset` only, so the leak the plan
warns about is impossible **by construction** -- there is no flipped file that
could be split into test, because there are no flipped files.

Three reasons it beats writing the doubled set to disk:

| | offline | online |
|---|---|---|
| extra disk | +21 GB per variant | none |
| steps an epoch | 3,160 | 1,580 |
| 100 epochs | ~30 h | **~15.5 h** |
| what the model sees | the same two copies every epoch | a fresh coin-flip each epoch |

The doubling is what costs the time, and it buys a *fixed* pair rather than
variety. One decision per clip, never per frame -- flipping frame 7 and not
frame 8 invents the camera jump Day 3 spent its time removing.

**Check.** `val_loader.dataset.transform is None` while the training one is a
`ClipFlip`. There is a test for exactly that.

**Takeaway.** Every augmentation needs this question asked: does it produce an
image that could really occur?

### 7.3 `evaluate.py`, with a checkpoint hash (≈45 min)

**Why.** A stale evaluation report already cost this project one wrong diagnosis
(see *The second bug*). Make it impossible.

**Build.** `src/evaluate.py`: top-1/2/3, precision, recall, macro-F1, the 15x15
confusion matrix — and the checkpoint's **SHA-256 and mtime written into the
report**, with a refusal to trust a report whose hash does not match.

**Code to understand.** Why a hash and not just a timestamp (a file can be
rewritten with the same mtime, and clocks move); and why the guard refuses
loudly rather than warning — a warning in a 6-hour log is a warning nobody
reads.

**Run.** Evaluate `best.pt` on val first, and check it reproduces the number in
`history.json`. Only then touch test.

Then evaluate the **same checkpoint on both test sets**. It was trained on
`splits_author/`, so the grouped test set contains matches it has seen — the
grouped number here is therefore not a clean measurement, only a first look at
the size of the effect. The clean version comes in Phase 3, where each
experiment is trained and tested within one split.

**Check.** Val matches history to within half a point. **Target on the author
test set: 89.09% ± 1.5%.** Record both numbers and their difference in the
journal; that difference is the first estimate of what the protocol is worth.

> **Actual: val 79.23%, test 78.14%**, val reproducing `history.json` exactly.
> The hash guard earned its place twice — once catching reports written for a
> checkpoint that a later run had replaced. Keep using `src/evaluate.py` rather
> than reading any JSON by hand.

The authors' per-class sanity checks matter as much as the headline:

- **Late Cut** is their worst class (F1 ≈ 0.745) — it looks like Upper Cut and
  Square Cut.
- **Scoop** is their best (recall ≈ 1.0) — the most visually distinct shot.

If your confusion matrix shows the same pattern, you have genuinely reproduced
their model. If it does not, something differs even when the headline matches.

**Takeaway.** Always reproduce a known number on val before computing an unknown
one on test. Test is touched once, at the end.

**Built 2026-10-11.** `src/evaluate.py`, 15 tests. Every report carries the
checkpoint's SHA-256, size and mtime; `load_report` re-hashes and raises
`StaleReport` on a mismatch.

A hash rather than a timestamp: there is a test that `os.utime`s the file
without changing a byte and asserts the report still loads, which a timestamp
check would fail. And it raises rather than warns, because a warning in a
six-hour log is a warning nobody reads.

Proved end to end on CPU -- trained a small model, scored it (`report top-1
0.3889` against `history best 0.3889`, difference 0.0000), then retrained
underneath the report and watched it be refused.

Run it when the real training finishes:

```bash
python -m src.evaluate experiments/mimic_author/best.pt     data/splits_author/val.csv --out experiments/mimic_author/val_report.json
```

Val first and only then test, as the takeaway says.

### 7.4 If you are short of 89% (≈2–4 h)

In order of likelihood:

1. Crops are wrong → back to Day 3.4 and look at the images again.
2. Flip augmentation missing, or applied before the split.
3. Too few epochs — they converged around epoch 30 of 100.
4. Backbone accidentally frozen — the authors fine-tune the whole thing.
5. Wrong normalisation (must be ImageNet mean/std).
6. BGR/RGB swapped in `video.py` — costs a few points, silently.

#### Outcome: all six eliminated, 2026-10-10 to 2026-10-13

This step was budgeted at 2–4 h and took three days. Every cause above was
tested and ruled out by measurement:

| # | how it was ruled out |
|---|---|
| 1 | padding swept 0→0.5 and both square rules; see `tools/crop_sweep.py` |
| 2 | present, and worth **+3.28** measured (their own ablation says +2.37) |
| 3 | early stop fired at epochs 45 and 54, after four lr reductions |
| 4 | tested — freezing the backbone is **28 points worse**, not better (47.98% against 75.95%, both converged; the 12 points first written here compared epoch 11 to epoch 5 and was not a converged comparison) |
| 5 | ImageNet mean/std, correct for torchvision weights |
| 6 | RGB beats BGR on the authors' own model, so the order is right |

Three more were tested and killed that this list does not name: backbone
drift under a single learning rate, blackout segmentation, and frame
de-duplication.

**The decisive tool was the authors' own trained model.** `CricShoot10kModels/`
ships their full Keras weights, and `tools/authors_model.py` scores them on our
crops. That partitions the gap in 13 minutes of CPU instead of a 6-hour run:

```
their model on our box crops              34.46%
their model on our paper-spec seg crops   43.22%
our model on the same crops               78.14%
their model, their crops (claimed)        89.09%
```

Their own network reads our frames at 43% where ours reads them at 78%. The
signal is in our crops; the framing is not theirs. **Use this ruler before
spending a GPU-day on any preprocessing idea** — it responds to real changes
(the segmentation fix moved it +8.76, over five standard errors) and it costs
minutes.

Two loading traps, both already handled in the tool: their `.keras` needs
**Keras 3.8.0** (metadata.json; 3.12 raises "Cannot convert '15' to a shape"),
and their weights file names backbone layers generically while their config
names them canonically, so `load_weights` falls back to positional order and
silently scores 16%. The tool refuses that path unless asked.

### 7.5 Write the baseline row and tag it (≈30 min)

```bash
git add -A && git commit -m "Reproduce CricShotNet baseline: 78.14% top-1 (authors report 89.09%)"
git tag baseline-reproduced
```

**This is the single most valuable artifact of the whole project.** Everything after it
is measured against it, and you built every line of it yourself.

**The baseline is 78.14%, not 89.09%, and that is the number to use.** It does
not have to equal theirs; it has to be fixed, honest and reproducible. Yours
comes with something their paper does not have: a measured account of where the
difference lives.

| run | val top-1 | test top-1 | test macro-F1 |
|---|---|---|---|
| **mimic_author (crop + flip)** | **79.23%** | **78.14%** | **0.7605** |
| mimic_author_seg, paper spec | 76.07% | 76.97% | 0.7499 |
| mimic_author_seg, merged grey | 74.94% | 75.51% | 0.7328 |
| mimic_author_noflip | 75.95% | — | — |
| authors, reported | — | 89.09% | 0.8847 |

Per-class behaviour matches theirs even though the level does not: **Late Cut
is the worst class on both splits** (F1 0.562 / 0.586), as in their paper, and
the confusions are cricket rather than noise — Lofted Offside↔Lofted Legside,
Pull↔Hook.

**What is known about the remaining 11 points**, so nobody re-runs these:

- Not the model. Our training takes this architecture to 78.14% on crops their
  own network can only read at 43%.
- Not cropping geometry, flip, de-duplication, normalisation, channel order,
  epochs, or backbone freezing — all measured, all ruled out in 7.4.
- Segmentation *is* part of it: fixing it to the paper's written spec moved
  their model +8.76. But ours gains nothing from it, most likely because only
  **65% of frames get a bat mask** (`tools/mask_quality.py`), so the green-bat
  cue the paper depends on is missing from a third of the data.
- The honest reading is that **their released weights do not correspond to the
  preprocessing their paper describes**. That is a defensible finding, backed
  by an unusually complete elimination, and it belongs in the write-up.

---

# PHASE 2 — Make 92 runs affordable (Days 8–9)

Forty-six experiments on two splits is **92 runs**. The baseline took
**199 minutes** (`mimic_author`, 45 epochs, `experiments/results.jsonl`), and
the segmentation run took **380**. Eighty-eight of those is roughly **300
hours** — twelve days of uninterrupted GPU on a desktop that loses power most
afternoons. You are not going to get there by being patient.

You get there by noticing that the backbone runs forty-five times for nothing.

In `mimic_author` the encoder is trainable, so its features change every epoch
and must be recomputed. But **most of the 46 experiments never touch the
encoder.** Family A swaps the temporal head — GRU → LSTM → BiLSTM →
transformer → TCN → attention pooling. Family D swaps the classifier — SVM,
XGBoost, logistic regression. For all of those the encoder is *frozen*, which
means its output for a given clip is **the same tensor on epoch 1 and on epoch
45**. Computing it 45 times is 44 wasted passes over 21 GB of video frames.

So compute it once, write it to disk, and let the experiments read tensors.

That is the whole of Phase 2. Day 8 builds the cache and proves it is correct.
Day 9 builds the harness that turns one experiment ID into two ledger rows
without you touching anything.

---

## Day 8 — Cache the CNN features

### 8.1 Do the arithmetic before writing the extractor (≈20 min)

**Why.** An earlier draft of this plan said the cache would be
`(15, 1280)` per clip and about 600 MB. That number is wrong by a factor of
forty-six, and building it would have produced a cache that **cannot reproduce
a single result you have.** Twenty minutes of arithmetic now prevents a wasted
day later.

**Build.** Nothing. A calculation, on paper, and then checked.

**The thing to work out.** What does one clip's feature tensor actually
contain? Look at `configs/mimic_author.yaml`:

```yaml
pool: flatten                      # 62720 features, from the GRU kernel shape
```

`pool: avg` gives 1280 numbers per frame — a global average over the spatial
grid. `pool: flatten` keeps the grid: EfficientNetV2-S ends at a
`(1280, 7, 7)` feature map, and `7 × 7 × 1280 = 62720`. **Every number in this
project was produced with `flatten`**, because that is what the authors' GRU
kernel shape `(62720, 384)` proves they used. A 1280-wide cache describes a
model you have never trained.

So predict, before running anything: how big is the cache?

**Run.**

```bash
python -c "
n = 9880
for name, shape, width in (
        ('avg   (15, 1280)',       (15, 1280),       2),
        ('grid  (15, 1280, 7, 7)', (15, 1280, 7, 7), 2),
        ('grid  as float32',       (15, 1280, 7, 7), 4)):
    per = width
    for d in shape:
        per *= d
    print('  %-24s %8.1f KB/clip  %6.2f GB total' % (
        name, per / 1024, per * n / 1024**3))
print('  7*7*1280 =', 7 * 7 * 1280)
"
```

**Check.** You should get exactly this:

```
  avg   (15, 1280)              37.5 KB/clip    0.35 GB total
  grid  (15, 1280, 7, 7)      1837.5 KB/clip   17.31 GB total
  grid  as float32            3675.0 KB/clip   34.63 GB total
  7*7*1280 = 62720
```

Then check it against the disk you actually have:

```bash
powershell -c "Get-PSDrive D | Select-Object @{n='FreeGB';e={[math]::Round(\$_.Free/1GB,1)}}"
```

**Measured 2026-10-15: 73.4 GB free on `D:`.** So the grid cache fits twice
over, and the decision is not close enough to need agonising about.

**The decision, and why.** Cache the **grid**, in **float16**, 17.3 GB.

- **Why the grid and not the pooled view.** The grid is a *superset*. Average
  pooling is a mean over the last two axes, so the 1280-wide view can be
  derived from the grid on the CPU in about nine minutes for both halves,
  with **no GPU and no second pass over the video** (step 8.8 proves this, to 3.6e-07). The reverse
  is impossible: once you have averaged, the spatial layout is gone forever and
  recovering it costs another 14-minute extraction. Cache the thing you cannot
  recompute cheaply.
- **Why float16 and not float32.** Half the bytes for the same features to
  within fp16 rounding, and the only consumer is a network that trains under
  bf16 autocast anyway — it has *less* precision than the file. 34.6 GB to
  store numbers the model immediately rounds away is not a trade.
- **Why not JPEG-style compression.** These are not pictures. The values are
  post-ReLU activations with no spatial smoothness to exploit, so lossy codecs
  destroy them and lossless ones barely compress.

**Takeaway.** Before building a cache, work out what one element of it
contains and multiply. The question "how big is it?" is also the question "did
I understand what I am storing?", and getting it wrong by 46× is what happens
when you answer the second one from memory.

> **Budget for the rest of Phase 2.** Write this in the journal:
>
> ```
> free now                     73.4 GB
> grid, un-mirrored  -17.3  -> 56.1 GB
> grid, mirrored     -17.3  -> 38.8 GB   (step 8.7)
> avg views, both     -0.7  -> 38.1 GB   (step 8.8)
> other extractors    -0.4  -> 37.7 GB   (step 8.9)
> ```
>
> If it ever gets tight, `data/processed/seg` is **21 GB** and is the lower-value
> variant (76.97% against box's 78.14%). It rebuilds from the videos in 4.5 h.
> The raw videos rebuild from nothing. Delete in that order, and never the
> other way round.

---

### 8.2 `extract.py` — one clip to one tensor (≈45 min)

**Why.** One file, one function: read a processed clip, run the frozen encoder,
write the feature tensor. Everything else in Phase 2 depends on this being
exactly right, so it gets built and checked on its own before it is let near
9,880 clips.

**Build.** `src/extract.py`, with this layout on disk:

```
data/features/effnetv2s/grid/<class>/<clip>.pt          (15, 1280, 7, 7) fp16
data/features/effnetv2s/manifest.jsonl                  one row per clip
```

**Code to understand.**

**The shape contract**, stated out loud because the whole cache is worthless if
it is wrong by a transpose:

```
on disk (input)   (15, 224, 224, 3) uint8      what build_dataset.py wrote
normalised        (15, 3, 224, 224) float32    ImageNet mean/std
encoder output    (15, 62720)       float32    pool="flatten"
reshaped          (15, 1280, 7, 7)  float32    channel-major
saved             (15, 1280, 7, 7)  float16
```

**That reshape is the one line that can silently poison everything.**
`FrameEncoder(pool="flatten")` hands back 62,720 numbers in a flat row, and
there are two plausible ways to fold them back into a grid: `(1280, 7, 7)` or
`(7, 7, 1280)`. Only one is right. torchvision's EfficientNet does
`features -> avgpool -> flatten(1) -> classifier`, and `_strip_pooling` makes
`avgpool` an `Identity`, so the tensor reaching `flatten(1)` is
`(N, 1280, 7, 7)` and C-order flattening makes it **channel-major**. So
`(1280, 7, 7)`. Do not take that on trust — 8.5 proves it with a number.

**Why the path has `grid/` in it but not `train/`.** The *view* (grid, avg) is
part of the tensor's identity: a different view is a different file. The
*split* is not — a frozen encoder's output depends only on the clip, so one
cache serves `splits_author` and `splits_grouped` both, and the split is just a
list of filenames over it. Writing `train/val/test/` into the path is the trap:
it would force a full re-extraction every time a split changed, and every
experiment in Phase 3 runs on two splits.

**Why `eval()` and `no_grad()` are both needed, and neither is enough alone.**
`no_grad()` stops the autograd graph being built — that is memory. `eval()`
stops BatchNorm updating its running statistics — that is correctness.
`FrameEncoder.train()` already forces `eval` on a frozen backbone for exactly
this reason (see `src/models/encoder.py`), but `freeze=True` must still be
passed, or the parameters get gradients they do not need.

**Why the write goes through `atomic_save`.** Same reason as Day 2: the power
goes off, and a half-written `.pt` that still loads as a tensor of the right
shape is the worst possible failure — it would poison every experiment
downstream and never raise. `src/utils/checkpoint.py:atomic_save` writes to
`.tmp`, fsyncs, then renames.

**Why it skips what the manifest already lists.** Fourteen minutes is short
enough to restart, but the manifest costs nothing and makes the habit uniform
with `build_dataset.py`. Same `append_result` / `read_results` pair.

**Why normalisation happens here and not at load time.** The ImageNet
mean/std shift is part of what the encoder *is* — the pretrained weights were
fitted on inputs scaled that way. Baking it in means the cache holds the
features of a correctly-normalised clip, and `FeatureDataset` does no arithmetic
at all. One place for the constants, not two that can drift apart.

**Run.** One clip, nothing more.

```bash
python -m src.extract --limit 1 --out data/features/effnetv2s
python -c "
import torch
t = torch.load('data/features/effnetv2s/grid/Cover Drive/vid100_41.pt')
print(t.shape, t.dtype, float(t.min()), float(t.max()), t.numel() * 2 / 1024, 'KB')"
```

**Check.** `torch.Size([15, 1280, 7, 7]) torch.float16`, `1837.5 KB`, and a
minimum of **−0.2793**, not zero.

**That floor is the real check, and it is sharper than "no negatives" would
be.** EfficientNetV2-S ends in **SiLU**, not ReLU — `features[-1]` is
`Conv2d → BatchNorm2d → SiLU` — and `SiLU(x) = x · sigmoid(x)` dips below zero,
bottoming out at **−0.278464556** at `x = −1.2788`. Under bf16 autocast the
nearest representable value to that is **exactly −0.279296875**, which is what
a correct cache reports. So the check is:

```
min  ≥ -0.28   and sitting right at it      -> correct layer, autocast as documented
min  =  0.0                                   -> wrong: that is a ReLU backbone, not this one
min ≪ -0.28                                   -> wrong layer, probably a pre-activation conv
```

An earlier draft of this step asserted `min == 0.0` on the grounds that the
backbone ends in "a ReLU-family activation". It does not, the smoke test said
so on 2026-10-15, and the measured value turned out to pin **two** things at
once: the right layer, and that autocast really is bf16.

**Takeaway.** A cache is a contract with your future self, and the shape is the
contract. Write it in the module docstring before writing the code, the way
`dataset.py` does — then any later reader can check the file against the
docstring instead of re-deriving it.

---

### 8.3 Launch the extraction (≈15 min to start, then ≈9 min)

*A launch step: no new code, so only Run, Check and Takeaway apply.*

**Run.**

```bash
python -m src.extract --variant box --out data/features/effnetv2s --workers 4
```

**Predict first.** Before you press enter: how long? The baseline trains an
epoch over 6,320 clips in 4.4 minutes *with* a backward pass. A forward pass
costs roughly a third of a forward-plus-backward, so guess, write the guess in
the journal, and then compare.

**Predicted 2026-10-15, before the extractor existed**, by timing
`FrameEncoder(pool="flatten")` in a bare synchronous loop on the Arc B580:

```
  batch 4: 32 clips in 2.6s = 12.1 clips/s -> 9880 clips in 13.6 min
  batch 8: 32 clips in 2.7s = 11.9 clips/s -> 9880 clips in 13.8 min
  disk:    1.79 MB/clip at 1068 MB/s       -> 17.3 GB in 0.3 min
```

**Measured 2026-10-15, the real run**, `--workers 4`:

```
  9,880 clips, 0 failed, 8.5 min, 19.4 clips/s
  17.3 GB written, 0 .tmp left, 57 GB free afterwards
  largest activation 32.8 -> 2000x fp16 headroom
```

**19.4 against a predicted 12.1 — 60% faster than the bare loop**, and that
difference is the whole argument for `--workers`. Batch 8 was *not* faster than
batch 4 in the synchronous loop, which said the GPU was never the bottleneck:
the `.npy` read and the float32 normalisation on the CPU were. Four worker
processes do that work while the GPU is busy, and the job drops from 13.6
minutes to 8.5. Raising the batch size would still not help.

The disk write is 0.3 minutes of an 8.5-minute job, so the fp16-versus-fp32
decision in 8.1 was about disk *space*, never about time.

**Check.** Three numbers, all of which must agree:

```bash
python -c "
import os, json
root = os.path.join('data', 'features', 'effnetv2s', 'grid')
n = 0
for folder, _, files in os.walk(root):
    for f in files:
        n += f.endswith('.pt')
print('files      ', n)
print('manifest ok', sum(
    1 for line in open(os.path.join('data','features','effnetv2s','manifest.jsonl'),
                       encoding='utf-8') if line.strip() and json.loads(line)['ok']))
print('tmp left   ', sum(f.endswith('.tmp') for _, _, fs in os.walk(root) for f in fs))
"
du -sh data/features/effnetv2s/grid
```

**9,880** files, **9,880** `ok` rows, **0** `.tmp` files, **17 GB**. 9,880 and
not 10,091, because 211 clips failed preprocessing on Day 4 and have no `.npy`
to read — the same 211 that `dataset.py` already drops.

**Takeaway.** When a job has a compute part and an I/O part, measure which one
is in charge before tuning either. Fourteen minutes of GPU and twenty seconds
of disk means every minute spent on the storage format was a minute spent on
the wrong thing.

---

### 8.4 `FeatureDataset` — read the cache back (≈45 min)

**Why.** `CricShotDataset` reads `.npy` frames and hands the model
`(T, 3, 224, 224)`. Nothing can use the cache until there is an object that
reads `.pt` features and hands the model `(T, D)`. It is the mirror image of
Day 5.1, and the same rule applies: index it directly before a DataLoader is
anywhere near it.

**Build.** `src/data/features.py` with `FeatureDataset`, taking the same split
CSVs as `CricShotDataset` so the two are drop-in alternatives.

**Code to understand.**

**`view` selects the shape handed to the head, and each view is one reshape:**

```
view="flat"   read grid/  (15, 1280, 7, 7) -> (15, 62720)   reproduces the authors
view="avg"    read avg/   (15, 1280)       -> (15, 1280)    Family A and D
view="grid"   read grid/  (15, 1280, 7, 7) -> unchanged     spatial-attention heads
```

One directory and one reshape per view, with no fallbacks. A fallback — "read
`avg/` if it exists, else reduce `grid/`" — is the kind of convenience that
makes two experiments silently read different numbers, and you would never find
out which.

**Where the float16 becomes float32, and why there.** On disk it is fp16; the
model wants fp32 (autocast casts down again inside the forward pass, which is
its job, not yours). So `__getitem__` does `.float()` on the way out. Do it in
the Dataset, not in the training loop: the loop should not know what precision
the cache happens to use.

**Why it reuses the split CSVs and `classes.json` unchanged.** Rule 3 — same
splits for everything. A `FeatureDataset` that built its own class ordering
would produce label 7 for a different shot than `CricShotDataset` does, and
every comparison between a cached and an end-to-end run would be quietly
meaningless. Import `_classes_for` rather than re-implementing it.

**Why `strict` and `.missing` come across from `CricShotDataset`.** Same
problem, one level down: `splits_author/train.csv` names 6,466 clips, 146 of
which have no `.npy` and therefore no `.pt`. Drop them loudly at construction, exactly as
Day 5.1 does, so a run cannot die on epoch 3.

**Run.**

```bash
python -c "
from src.data.features import FeatureDataset
for view, want in (('flat', 62720), ('grid', None)):
    d = FeatureDataset('data/splits_author/train.csv', view=view)
    x, y = d[0]
    print('%-5s %-6d clips  %-22s %s  label %d' % (
        view, len(d), tuple(x.shape), x.dtype, y))
"
```

**Check.** `flat` gives `(15, 62720) torch.float32`; `grid` gives
`(15, 1280, 7, 7)`. Both report **6,320** clips — 6,466 in the CSV minus the
146 that failed preprocessing — which must equal what
`CricShotDataset('data/splits_author/train.csv')` reports. If those two numbers
differ, the cache is incomplete and 8.3's check lied.

**Takeaway.** When a second Dataset reads the same splits as the first, the
class ordering, the missing-clip handling and the label arithmetic must be
*shared code*, not parallel code. Two implementations of "which shot is label
7" is a bug that shows up as an inexplicable accuracy difference six weeks
later.

---

### 8.5 Prove the cache holds the same numbers (≈30 min)

**Why.** This is the most important check in Phase 2, and the reason is in the
plan's own rules: **a silently wrong cache poisons all 60 rows after it.** The
accuracy check in 8.6 is too blunt to catch a subtle error — a transposed grid
or a half-written file would still train to *some* number, and you would spend
Phase 3 comparing experiments to each other and never notice they were all
wrong together.

So check the tensors, not the accuracy.

**Build.** `tests/test_extract.py`, plus the two one-off measurements below.

**Code to understand.** Three checks, each catching something the others
cannot.

**Check 1 — round-trip against the extraction path, bit-exact.** Call the same
`encode_clip()` that extraction called, on 20 clips, and compare to the files.
Not with a tolerance — `torch.equal`. The forward pass is deterministic, so a
correct cache reproduces **exactly**, and any gap at all means the cache was
written by different code or different settings. What can actually break here
is serialisation and layout, so this tests that and nothing else.

Getting bit-exactness required finding a real bug first, and it is the subtlest
one of Day 8.

> **The cache depended on `--workers`, a flag that is only supposed to affect
> speed.** Measured 2026-10-15: identical input tensors, deterministic forward,
> and yet `--workers 0` and `--workers 4` wrote files differing by **0.39** on
> values reaching 24 — reproducibly, 50× bf16's spacing at that magnitude.
>
> The cause is two steps removed from either flag. `dataset.py` builds its
> tensor with `.permute(0, 3, 1, 2)`, which makes it **channels_last**, and
> that memory format propagates through the elementwise normalisation. The
> DataLoader then silently decides it: at `--workers 0` the collated batch
> stays channels_last; at `--workers 4` the shared-memory collate rewrites it
> to NCHW. oneDNN picks a different kernel for each, and bf16 rounding diverges
> and compounds over forty layers.
>
> Neither result is *wrong* — both are valid bf16 computations. But a cache
> whose contents depend on a performance knob is a cache you cannot verify, and
> it would have made 8.6's epoch-1-loss comparison meaningless.
>
> **`.contiguous()` in `encode_clip` fixes it. `.clone()` does not**, because
> clone preserves memory format — which is exactly why the first fix attempt
> failed. Two regression tests pin it: one that the two worker counts now
> agree, and one that `dataset.py` really does hand over a channels_last
> tensor, so that if it ever stops, the `.contiguous()` can be reconsidered
> on evidence rather than removed on a guess.
>
> The cache already on disk was built with `--workers 4`, which is the NCHW
> path, so it needed **no re-extraction** — both worker counts now reproduce
> it bit-for-bit.

**Check 2 — the reshape order, against an independent route.** If
`(15, 1280, 7, 7)` is the correct folding, then averaging over the last two
axes must equal what `FrameEncoder(pool="avg")` produces, because that is
precisely what its average-pool layer computes. Two different code paths, one
answer. **Measured 2026-10-15:**

```
  vid100_41.npy    max|derived - real| = 3.576e-07
  vid100_60.npy    max|derived - real| = 4.768e-07
  vid100_8.npy     max|derived - real| = 3.576e-07
```

Float32 rounding noise — `1e-7`, not `1e-3`. The reshape is right, and the avg
view is a genuine derivative of the grid rather than an approximation of it.
Had the order been `(7, 7, 1280)` this would have come out around `1e-1`.

**Check 3 — the activation floor, everywhere.** `min() >= -0.28` on every
cached tensor, and no tensor sitting at a hard `0.0`. The backbone's last
operation is **SiLU**, whose global minimum is −0.278464556 (−0.279296875 in
bf16), so a correct cache presses against that floor and never goes under it.
One pass over the cache, and it catches a wrong layer, a wrong backbone and a
corrupt file at once — more tightly than `>= 0` would, because it pins the
value to a known constant instead of a half-plane. See 8.2.

**Run.**

```bash
python -m pytest tests/test_extract.py -q
python -m src.extract --verify --clips 20
```

**Check.** Measured 2026-10-15 — 22 new tests, and:

```
verifying 20 clips from dataeatures\effnetv2s
  1 round-trip against encode_clip   PASS   20/20 bit-exact  max abs gap 0.0000  (values reach 27.2)
  2 reshape order, fp32, vs pool=avg PASS   max abs 4.768e-07  (float32 noise is ~1e-7)
  3 activation floor                 PASS   min -0.279297  max 27.25  (SiLU floor -0.278465; 0 at a hard 0.0)

all 3 passed
```

474 tests in the suite, up from 452.

**Takeaway.** Verify a cache by comparing *tensors against an independent route
to the same tensors*, never by checking that a model trained on it still works.
"It still works" cannot distinguish "correct" from "wrong in a way the
optimiser routed around", and only one of those is recoverable.

---

### 8.6 Prove the cache trains to the same number (≈10 min to start, then ≈25 min)

*A launch step. The code already exists — this run changes one config key.*

**Why.** 8.5 proved the numbers are right. This proves the *pipeline* is right:
that `FeatureDataset` + the existing `GRUHead` + the existing loop reach the
same accuracy as reading video. It is the end-to-end check, and it has a
precise target.

**Which target — and this is the part the first draft of this step got
wrong.**

> **A feature cache *is* a frozen backbone.** Written on 2026-10-15 after
> running it. `mimic_author_noflip` has `freeze: false`: its backbone updates
> during epoch 1, so the features its head sees at step 1,500 are not the
> features it saw at step 1. A cache is fixed by construction. **No cached run
> can ever reproduce a trainable-backbone run**, and setting 75.95% as the
> target made a correct cache look broken.
>
> Measured, epoch 1:
>
> | run | backbone | train top-1 | val top-1 | pin |
> |---|---|---|---|---|
> | `author_frozen` (video) | frozen | 23.07% | 30.37% | 41% |
> | `cached_gru_noflip` | frozen | **23.47%** | 26.20% | **39%** |
> | `mimic_author_noflip` | trainable | 35.21% | 53.98% | 0% |
>
> The cached run lands on the **frozen** row, 0.4 points from it, with the same
> 40%-ish pinning. That agreement is the evidence the cache is correct. The
> 28-point gap to the trainable row is the price of freezing, and it is the
> thing to understand about all of Phase 3's cheap rows — not a defect in
> Phase 2.
>
> **Converged, measured 2026-10-15:**
>
> | run | backbone | train | val | gap | pin |
> |---|---|---|---|---|---|
> | `mimic_author` | trainable + flip | 83.51% | **79.23%** | 4.3 | 0% |
> | `mimic_author_noflip` | trainable | 86.27% | **75.95%** | 10.3 | 0% |
> | `cached_gru_noflip` | **frozen** | 86.90% | **51.20%** | **35.8** | **45%** |

> Final, 78 epochs, 93.6 min. Freezing costs **24.75 points**.
>
> Three things cause it, and they compound. **ImageNet features are not cricket
> features** — frozen, the encoder describes "a person in white on grass"
> identically for a Cover Drive and a Square Cut, and fine-tuning is what
> teaches it bat angle and body rotation. **So the head memorises instead of
> generalising**: a 22.8-point train/val gap against the baseline's 4.3. And
> **45% of the GRU is dead throughout**, not transiently — 39% at epoch 1,
> still 45% at epoch 32.
>
> That last one is a threat to the screening premise, not just a lost result.
> The cheap sweep is meant to rank heads so the expensive 32 runs go to the
> right ones — but it would be ranking them **inside a pathology that does not
> exist in the fine-tuned setting**, where pinning is 0%. Hence A13, which is
> now a validity fix rather than an optimisation, and 9.7, which measures
> whether the ranking transfers before 28 rows are built on the assumption
> that it does.

So the comparison is a **frozen video run**, and flip is held at 0 because the
un-mirrored cache cannot flip (8.7 restores it). This also means the 60 cached
rows of Families A, D and V are comparable **to each other** and not to
78.14% — which is what Days 28–30 exist for: take the best heads from the
cheap frozen sweep and re-run those few end to end.

**Run.**

```bash
python -m src.train --config configs/cached_gru_noflip.yaml
```

where the config is `mimic_author_noflip` with three keys changed — `features`
replacing `variant`, `view: flat`, `freeze: true` — and nothing else. One
variable at a time means the *only* difference from the 224-minute run is where
the features come from.

**Check.** Epoch 1, against a **frozen video run**, and in this order:

1. **Train top-1 and `pin`.** Both are nearly pure functions of the inputs at
   epoch 1, so a wrong cache shows up here in two minutes rather than after
   thirty epochs. Measured: 23.47% against the frozen video run's 23.07%, and
   pin 39% against 41%.
2. **The definitive version**, worth the 15 minutes: run the same frozen
   config on **video** with today's code, two epochs, and compare epoch 1
   directly. `author_frozen` predates the BatchNorm fix, so the table above is
   suggestive rather than conclusive.
3. **Final val top-1.** Whatever it is, it *is* the frozen baseline — the
   reference every Family A row gets measured against. `author_frozen` was
   interrupted at epoch 12 and never produced one, so this run is filling a
   real hole rather than confirming a known number.

**The number that justifies the whole day.** Also record the wall clock. The
projection, to be replaced by the measurement: the GRU's input layer is
`62720 × 384 = 24.1 M` parameters, so a `flat` epoch is a real matmul plus
11.1 GB of disk read — call it 15–20 s against the end-to-end 4.4 minutes.
Roughly a **13× speedup** on the faithful configuration, and far more than that
on the `avg` view, which reads 48× less. Measure both and write them down;
every scheduling decision in Phase 3 rests on them.

**Takeaway.** When you replace one component and want to know whether the
replacement is faithful, compare the *earliest* observable, not the final one.
Epoch 1's loss is nearly a pure function of the inputs; the final accuracy has
45 epochs of optimiser dynamics layered on top, and those will happily hide a
real defect.

---

### 8.7 The mirrored half, because flip cannot be faked (≈20 min, then ≈14 min)

**Why.** Horizontal flip is the single best-measured augmentation in this
project: **+3.28 points**, and removing it opens a 15-point train/val gap.
Every cached experiment needs it, and a cache of un-flipped features cannot
provide it.

**The obvious shortcut, and why it is wrong.** A horizontal flip of the image
ought to be a horizontal flip of the `7 × 7` feature grid. That would be free:
one `torch.flip` at load time, no extra disk. It is also false, and not
marginally false. Convolution is not flip-equivariant —
`conv(flip(x), w) = flip(conv(x, flip(w)))`, and `flip(w)` is a *different
filter*. **Measured 2026-10-15** on three clips, cosine similarity between
`flip(grid(x))` and `grid(flip(x))`:

```
  vid100_41.npy    0.3776
  vid100_60.npy    0.4480
  vid100_8.npy     0.4148
  mean             0.4135
```

0.41 is not an approximation, it is a different tensor. So the mirrored
features have to be computed by the encoder, like any others.

**Build.** `--mirror` on `src/extract.py`: flip the normalised clip on the last
axis before encoding, write to `grid_mirror/`. Four lines, and it reuses
`ClipFlip`'s own `torch.flip(clip, dims=[-1])` so there is one definition of
"mirrored" in the codebase rather than two.

**Why this is exactly equivalent to flipping at train time, not an
approximation of it.** `ClipFlip` is a *binary* choice made once per clip:
either the clip or its mirror, probability `p`. There are exactly two possible
inputs, so caching both is complete. `FeatureDataset(mirror=0.5)` then reads
from `grid/` or `grid_mirror/` with probability 0.5, and the distribution of
tensors the model sees is identical to the end-to-end run's. This is the rare
case where a cache loses nothing at all.

**Run.**

```bash
python -m src.extract --variant box --mirror --out data/features/effnetv2s --workers 4
python -m src.train --config configs/cached_gru.yaml
```

**Check.** 9,880 files in `grid_mirror/`, another 17 GB (**measured: 39 GB
free, against 8.1's budgeted 38.8**), and the cached flip run beats the
un-mirrored one.

**Measured 2026-10-15, and it refutes the prediction first written here.** I
expected flip to be worth *more* frozen than trainable, because the frozen
run's train/val gap is 35.8 points against the baseline's 10.3. It is worth
less:

| | best val | overfit gap at the end | flip worth |
|---|---|---|---|
| frozen, noflip | 51.20% | 35.8 | |
| frozen, flip | **53.66%** | 15.1 | **+2.46** |
| trainable, noflip | 75.95% | 10.3 | |
| trainable, flip | 79.23% | 4.3 | **+3.28** |

```
frozen     removed 20.7 points of overfitting, converted 2.46  ->  12%
trainable  removed  6.0 points of overfitting, converted 3.28  ->  55%
```

**Flip suppressed the memorising almost entirely and the accuracy barely
moved.** That is an information ceiling, not a regularisation problem: ImageNet
features do not hold the discriminative cricket signal, and stopping the head
from memorising does not create it. The frozen head is not failing because it
overfits -- it overfits *because* it is failing.

So **Tier 2's regularisers should be expected to underdeliver on every cached
row** (12% conversion), while still earning their estimates on the trainable
ones (55%). And `pin` did not move at all -- 45% without flip, 46% with -- so
the saturation is a separate problem needing a separate fix, which is A13.

Sanity check worth the ten seconds: the mirrored tensor must not equal the
un-mirrored one for any clip, and must have the same `min() == 0`.

**Takeaway.** Before caching a *transformed* input, ask how many distinct
transformed inputs there are. Two, as here, means caching all of them is exact.
Continuous augmentation — colour jitter, random resized crop, speed
perturbation — has infinitely many, cannot be cached, and therefore cannot run
on frozen features at all. That is a real constraint on Phase 3, and Tier 2
items 9, 10 and 11 are the ones it bites.

---

### 8.8 Derive the pooled views without touching the GPU (≈30 min)

**Why.** Family A's interesting heads — a transformer, a TCN, attention
pooling — do not want 62,720 numbers per frame. A transformer over 62,720-wide
tokens spends 24 M parameters on the input projection alone before it has
learned anything. They want the 1,280-wide view. And at train time, reading
1.79 MB per clip in order to average it down to 37.5 KB means 11.1 GB of disk
traffic per epoch instead of 231 MB.

So materialise the small view once. 8.5 already proved it is derivable to
`3.6e-07`, so this costs no GPU and no second pass over the video.

**Build.** `src/extract.py --derive avg`: walk `grid/` and `grid_mirror/`,
`mean(dim=(-1, -2))`, write `avg/` and `avg_mirror/`.

**Code to understand.** Two small things, both easy to get wrong.

**Average in float32, store in float16.** The input is fp16 and you are summing
49 values. Summing fp16 in fp16 accumulates rounding error that the measured
`3.6e-07` does not include, because that measurement started from fp32. So
`.float().mean(...)` then `.half()`. One `.float()` is the difference between a
derived view that matches and one that nearly matches.

**Reduce over the right axes.** `dim=(-1, -2)` on `(15, 1280, 7, 7)` gives
`(15, 1280)`. `dim=(1, 2)` would give `(15, 7)` — still a valid tensor, still
saves without complaint, and completely meaningless. Negative axis indices here
because they stay correct whether or not a batch dimension is present.

**Run.**

```bash
python -m src.extract --derive avg
python -c "
import torch
g = torch.load('data/features/effnetv2s/grid/Cover Drive/vid100_41.pt').float()
a = torch.load('data/features/effnetv2s/avg/Cover Drive/vid100_41.pt').float()
print(tuple(a.shape), 'max diff', float((g.mean(dim=(-1, -2)) - a).abs().max()))"
du -sh data/features/effnetv2s/avg data/features/effnetv2s/avg_mirror
```

**Check.** `(15, 1280)`, a max difference at fp16 resolution (`~1e-3`, the
storage floor, not the `1e-7` of the arithmetic), and about **350 MB** each.

**Measured 2026-10-15:**

```
grid/ -> avg/                9,880 clips, 0 failed, 5.1 min, 32 clips/s
grid_mirror/ -> avg_mirror/  9,880 clips, 0 failed, 3.8 min, 43 clips/s
each file 39,977 bytes       389 MB per half, 0 .tmp, 38 GB free afterwards
vs the float32 mean          max diff 6.3e-04 -- fp16 rounding, nothing else
```

**The time estimate was wrong by 4x, and the reason is worth knowing.** It came
from timing 300 files that had just been read, so they were in the OS cache,
and it timed only read-and-average. Per clip, measured afterwards on files not
recently touched:

```
read + average       10.6 ms
atomic_save fsync     2.3 ms
manifest-row fsync    7.1 ms
total                20.0 ms   -> ~50 clips/s
```

Half the time is the two fsyncs -- the price of a power cut costing one clip
rather than a corrupt cache. Worth paying for a job that runs once per
extractor, and the reason not to "optimise" them away.

**Takeaway.** A cache in two widths is not duplication, it is a *lossy
derivative of a lossless archive*. Keep the archive for the thing that needs
it, keep the derivative for the thing that runs forty times, and be able to
rebuild the derivative from the archive in minutes with no GPU. That property
is what makes the 17.3 GB defensible.

---

### 8.9 Commit the cache tooling (≈10 min)

*A commit step: no new code, no check.*

```bash
git add -A && git commit -m "cache EfficientNetV2-S grid features: 17.3 GB, 14 min, reproduces the baseline from tensors"
```

`.gitignore` must exclude `data/features/` — it is 35 GB and it is
reproducible from `src/extract.py` plus `data/processed/`. What gets committed
is the extractor, `FeatureDataset`, the tests, the two configs, and the
measured numbers in the journal.

> **Journal entry for Day 8.** Every claim with a number: the grid is
> `(15, 1280, 7, 7)` fp16 at 1.79 MB/clip; extraction ran at 12.1 clips/s,
> loader-bound not GPU-bound; the avg view is derivable to 3.6e-07; the
> flip-the-grid shortcut has cosine 0.41 and is therefore wrong; the cached
> baseline is X% against the end-to-end 79.23%; an epoch now takes Y s against
> 264 s.

---

### 8.10 Launch the other extractors, and leave them running (≈45 min to start, then ≈2 h unattended)

*A launch step. Start this at the end of Day 8 and let it run through Day 9 —
it is pure compute with no decisions in it.*

**Why.** The cache is a *format*, not one model: anything that turns a clip
into a feature tensor plugs into the same `FeatureDataset`. Caching several
extractors now means Families A, D and V are all unblocked on the same morning,
and it buys an unusually cheap piece of information — a **linear probe** on
each cache tells you whether motion pretraining matters here before you have
spent one hour fine-tuning.

**Build.** `--backbone` on `src/extract.py`, writing to
`data/features/<name>/`. In this order:

| # | extractor | pretrained on | per clip | all 9,880 |
|---|---|---|---|---|
| 1 | **ConvNeXt-Tiny**, per frame | ImageNet | `(15, 768)` | 227 MB |
| 2 | **r3d_18** | Kinetics-400 | `(512,)` | 10 MB |
| 3 | **VideoMAE-Base** | Kinetics-400 | `(768,)` | 15 MB |
| 4 | **CLIP ViT-B/16**, per frame | WIT-400M | `(15, 768)` | 243 MB |

About 400 MB in total — these are all pooled views, so the 17.3 GB problem does
not recur. Roughly 20–40 minutes each on the Arc. *(Measured: 1.09 GB with
both halves, since CLIP is 768 wide rather than 512; 3–11 minutes per half.
See the summary at the end of 8.10.)*

**Two things to get right.** The video models (2 and 3) return **one vector per
clip**, not one per frame, so `FeatureDataset` must accept a `(D,)` tensor and
the head must be a plain classifier rather than a temporal one — there is no
time axis left to model. And CLIP does **not** use ImageNet normalisation; it
has its own mean and std, so the constants in `extract.py` have to come from
the extractor rather than from `dataset.py`.

**Built in four parts, one backbone each, because the four need different
code** (checked 2026-10-15). Only ConvNeXt goes through the existing
`FrameEncoder`. r3d_18 is in torchvision but takes the whole clip at once and
needs its own encoder path; VideoMAE and CLIP need `transformers`, which is not
installed. The video models also expect 16 frames, and the clips have 15.

**Part 1, ConvNeXt-Tiny: built 2026-10-15.** It did not "just work" with
`--backbone convnext_tiny`. Three fixes:

1. **Its last layer is hidden in its classifier.** ConvNeXt's head is
   `LayerNorm -> Flatten -> Linear`, and `_strip_classifier` threw the whole
   head away, so the encoder returned `(N, 768, 1, 1)` with no LayerNorm and
   crashed. It now drops only the Linear (and any Dropout). EfficientNet's
   head is `Dropout -> Linear`, so it still becomes `Identity`; the baseline
   is unchanged.
2. **Its avg view cannot be derived from a grid.** The LayerNorm runs after
   the pool, and LN(mean) is not mean(LN). So `--pool avg` writes `avg/`
   straight from the encoder. That is also the smaller job: 768 numbers per
   frame, about 225 MB per half.
3. **The cache folder defaulted to `effnetv2s` for every backbone.** A second
   backbone would have found that manifest full and quietly written nothing,
   the same trap as `build_dataset --variants seg` on Day 4. The folder now
   defaults to the backbone's name, and `source.json` records backbone and
   variant on the first write. A mismatch stops the run before the model
   loads.

**Its check is different from the grid's.** There is no grid to fold, and the
SiLU floor is EfficientNet's. So `--verify --pool avg` runs two checks: the
bit-exact round trip, and **the ImageNet Linear applied to the cached vectors
must reproduce the whole model's logits**. That proves the file holds the
vector the pretrained head reads. A 4-clip smoke run gave logit cosine 0.9999
and top-1 agreement on 45 of 45 frames. With the LayerNorm left out it gives
**0.87, and top-1 still agrees on 11-12 of 15 frames**, which is why an
accuracy-style check would have passed the bug.

```bash
python -m src.extract --backbone convnext_tiny --pool avg --workers 4
python -m src.extract --backbone convnext_tiny --pool avg --workers 4 --mirror
python -m src.extract --verify --backbone convnext_tiny --pool avg
```

**Measured 2026-10-16:**

```
avg/           9,880 clips, 0 failed, 9.2 min, 17.9 clips/s
avg_mirror/    9,880 clips, 0 failed, 10.9 min, 15.0 clips/s
each file      24,617 bytes (15 x 768 fp16 + header), 243 MB per half
largest value  10.9 -- fp16 headroom 5,985x
verify         20/20 bit-exact; logit cosine 0.9999, top-1 43/45 frames
```

Same speed as EfficientNet's 19.4 clips/s although ConvNeXt-Tiny is the
lighter network: the loader is the bottleneck, as on Day 8. The two top-1
misses are near-tied logits flipping under bf16; the cosine is what tests
the layer.

**Part 2, r3d_18: built 2026-10-16.** A new `src/models/video_encoder.py`:
the whole clip goes in at once, `(B, T, 3, H, W) -> (B, 3, T, H, W)`, and
one 512-vector per clip comes out. What it needed:

1. **Its own input size and colours, read from the weights.**
   torchvision ships them: 112x112 and the Kinetics mean/std. `_Clips` now
   takes size and colours from the encoder; the ImageNet 224 path is
   byte-for-byte unchanged, so the EfficientNet cache still verifies.
2. **A `video/` view of `(D,)` tensors.** `FeatureDataset` used to compute
   the width as the product of every axis after time, which is **1** for a
   `(512,)` tensor (the empty product). It now reads 512, and each view
   checks the number of axes it expects, so a `(512,)` file read as `avg`
   raises instead of being misread.
3. **The pool comes from the backbone.** `--backbone r3d_18` alone means
   `--pool video`; `--pool avg` on it, or `--pool video` on a frame model,
   stops before any download.

15 frames is fine: r3d_18 pools time adaptively (15 and 16 both give
`(1, 512)`). Only VideoMAE needs exactly 16.

**Measured before the run:**

```
smoke, 8 clips       8/8 bit-exact; logit cosine 1.0000, top-1 8/8 clips
its own guess        'playing cricket' on 38 of 40 clips across all classes,
                     mean probability 0.92
same, wrong input    (ImageNet colours at 224) still 38/40 top-1, but
                     mean probability 0.62
```

Kinetics-400 has a "playing cricket" class, so this backbone has seen the
sport, though not the difference between a drive and a pull. And the
wrong-input row is why the input spec comes from the weights: it fails
quietly, with nothing to see except a third of the confidence gone.

```bash
python -m src.extract --backbone r3d_18 --workers 4
python -m src.extract --backbone r3d_18 --workers 4 --mirror
python -m src.extract --verify --backbone r3d_18
```

**Measured 2026-10-16:**

```
video/          9,880 clips, 0 failed, ~3.5 min of work, ~60 clips/s
video_mirror/   9,880 clips, 0 failed, 2.7 min, 61 clips/s
each file       2,601 bytes, 25.7 MB per half
verify          20/20 bit-exact; logit cosine 1.0000, top-1 20/20 clips;
                'playing cricket' on 19 of 20
```

The first half was cut after 1,664 clips (00:29:32) and restarted at
00:54:39; the rerun did exactly the other 8,216. Resume works on a real cut.

**Three times faster than the frame models** (60 against 18-19 clips/s),
although the loader does more work here -- it also resizes. What it moves is
less: 112x112 is a quarter of the bytes of 224x224 through the worker
queue, the collate and the copy to the GPU. So Day 8's "the loader is the
bottleneck" was too simple; moving 9 MB per clip is at least part of it.
Not worth chasing for a job that runs once.

**Part 3, VideoMAE-Base: built 2026-10-16.** `VideoMAEEncoder` in the same
file, from `MCG-NJU/videomae-base-finetuned-kinetics` (transformers 5.20,
now in `requirements.txt`). The Kinetics-finetuned checkpoint, not the
self-supervised one: a frozen probe needs features that already learned
labels, and the classifier is what verify checks against.

1. **The vector is what its classifier reads**: mean over all 1,568 tokens,
   then `fc_norm`. Only `classifier` becomes `Identity` -- the ConvNeXt rule
   again.
2. **16 frames from 15**: `resample_frames` takes 16 evenly spaced picks,
   which repeats frame 7. Measured against repeating the last frame on 40
   clips: 'playing cricket' top-1 40/40 both ways, mean probability 0.625
   against 0.629. **The choice does not matter here**; even picks stay
   because they keep the timing even.
3. **HF_HUB_OFFLINE=1 is set in this machine's `.bashrc`** for another
   project. The model was downloaded once with the flag lifted for that one
   command; everything after runs offline from `D:/huggingface_cache`. If
   the model is missing, the encoder now says so with the download command
   rather than failing deep inside transformers.

ImageNet colours at 224 -- VideoMAE's own config says so -- so `_Clips` does
no resizing for it.

**Measured before the run:**

```
smoke, 8 clips       8/8 bit-exact; logit cosine 1.0000; top-1 8/8 clips
its own guess        'playing cricket' 8/8 (and 40/40 on the spread sample,
                     mean probability 0.63 against r3d_18's 0.92)
speed, 200 clips     18.0 clips/s -> ~9 min per half
```

```bash
python -m src.extract --backbone videomae_base --workers 4
python -m src.extract --backbone videomae_base --workers 4 --mirror
python -m src.extract --verify --backbone videomae_base
```

**Measured 2026-10-16:**

```
video/          9,880 clips, 0 failed (6,860 before a cut, 3,020 after)
video_mirror/   9,880 clips, 0 failed, 6.1 min, 27.1 clips/s
each file       3,113 bytes, 30.8 MB per half
verify          20/20 bit-exact; logit cosine 1.0000; top-1 20/20 clips;
                'playing cricket' 20/20
```

Faster than the 200-clip timing (27 against 18 clips/s): the short run
paid model load and worker start-up over too few clips. Second real resume
of a cache, again exact.

**Part 4, CLIP ViT-B/16: built 2026-10-16.** `src/models/clip_encoder.py`,
from `openai/clip-vit-base-patch16`. A frame model like ConvNeXt: every
frame on its own, `(15, 768)` per clip, written to `avg/`.

1. **768, not the table's 512.** 512 is CLIP's shared image-text space, one
   bias-free Linear after the 768. The CLIP paper's own linear probes read
   the 768, and a probe on 768 can learn anything a probe on 512 can. The
   table row is corrected.
2. **The vector is `pooler_output`**, the CLS token after the final
   LayerNorm: what the projection reads. A cache holding the wrong layer
   would read, against the true embedding,

   ```
   post-LayerNorm CLS (correct)    cosine 1.0000
   CLS before the final LayerNorm  cosine 0.4764
   mean of the patch tokens        cosine 0.4140
   ```

3. **CLIP's own colours**, from its config: mean 0.481/0.458/0.408 against
   ImageNet's 0.485/0.456/0.406.
4. **Check 2 has no classifier to use**, so it scores the projected
   embedding against the text "a photo of a person playing <sport>" for
   eight sports. Cosine on the embeddings tests the layer; the zero-shot
   guess is a sanity print, not a result.
5. **The official repo has no safetensors**, only `pytorch_model.bin`
   (599 MB), which transformers loads with `weights_only`. The shared HF
   plumbing (cache lookup with the download command in the error, quiet
   loading) moved to `src/models/hf.py`, used by both HF encoders. Quiet
   loading matters here: loading only CLIP's image half otherwise prints a
   25-line table of the text tower's "UNEXPECTED" keys.

**Measured before the run:**

```
smoke, 8 clips       8/8 bit-exact; embedding cosine 0.9998; top-1 45/45
its own guess        'cricket' on 45 of 45 frames, zero-shot
speed, 400 clips     19.8 clips/s -> ~8 min per half (probably less)
```

```bash
python -m src.extract --backbone clip_vit_b16 --workers 4
python -m src.extract --backbone clip_vit_b16 --workers 4 --mirror
python -m src.extract --verify --backbone clip_vit_b16
```

**Measured 2026-10-16 (CLIP):**

```
avg/           9,880 clips, 0 failed, 7.5 min, 21.9 clips/s
avg_mirror/    9,880 clips, 0 failed, 7.5 min, 22.1 clips/s
each file      24,617 bytes, 243 MB per half
verify         20/20 bit-exact; embedding cosine 0.9999; zero-shot
               'cricket' 45/45 frames
```

**8.10 done, 2026-10-16. Five caches, all verified, all logged.**

| cache | what it saw | per clip | both halves | per half | its own guess |
|---|---|---|---|---|---|
| `effnetv2s` | ImageNet photos | `(15, 1280, 7, 7)` + avg | 34.6 GB + 778 MB | 8.5 min | — |
| `convnext_tiny` | ImageNet photos | `(15, 768)` | 486 MB | 9–11 min | 'ballplayer' 21/45 frames |
| `r3d_18` | Kinetics actions | `(512,)` | 51 MB | ~3 min | 'playing cricket' 19/20 |
| `videomae_base` | Kinetics actions | `(768,)` | 62 MB | ~6 min | 'playing cricket' 20/20 |
| `clip_vit_b16` | 400M captioned photos | `(15, 768)` | 486 MB | 7.5 min | 'cricket' 45/45 (zero-shot) |

The last column is a sanity print, not a ranking: every model knows it is
looking at a bat-and-ball sport, and **none of that says it can tell a
drive from a pull.** That is the probe's job, on Day 9.

Two power cuts hit these runs (r3d_18 at 1,664 clips, VideoMAE at 6,860).
Both resumed exactly. From part 4 on, every command-line run writes a row to
`experiments/caches.jsonl` (in git), so these numbers no longer depend on
being copied out of a terminal.

The probe that ranks the caches needs the Day 9 runner, so it waits until
then. The extraction does not have to wait.

**Family V needs un-segmented frames, and they already exist.** That is
`data/processed/box`, built alongside `seg` in one pass on Day 4. No second
`build_dataset.py` run — point the extractor at `--variant box`.

**Only `box`, for now.** On the author split `box` scores 78.14% against `seg`'s
76.97%, so the segmented cache is the lower-value one and it doubles every
number in the table above. Extract it when an experiment calls for it.

**Run.**

```bash
for b in convnext_tiny r3d_18 videomae_base clip_vit_b16; do
  python -m src.extract --backbone $b --variant box --workers 4
done
```

**Check.** A file count and a size per cache, and then the probe:

```bash
python -m src.experiments.run --family probe
```

one linear head per cache, two minutes each. **If the Kinetics features beat
the ImageNet ones on a linear probe alone, that is your evidence that motion
pretraining is the lever** — recorded before a single hour of fine-tuning, and
it tells you which of Family B and Family V deserves the GPU time.

**Takeaway.** A frozen feature cache plus a linear probe is the cheapest
model-selection tool in deep learning. It ranks ten backbones in twenty minutes
and the ranking usually survives fine-tuning. Do it before committing GPU
hours, not after.

---

## Day 9 — The experiment harness

The cache makes the runs cheap. The harness makes them *safe*: 92 runs over
eighteen days with power cuts is a bookkeeping problem long before it is a
compute problem, and bookkeeping done by hand is how you lose six results in
week three and cannot tell which six.

### 9.1 The registry: 92 runs as data, not 92 files (≈1 h)

**Why.** There are 46 experiment IDs and two splits. Writing 92 YAML files
guarantees that experiment A07 and A08 differ in some way you did not intend,
because you edited a copy of a copy. Rule 2 — one variable at a time — has to
be enforced by the *structure*, not by remembering.

**Build.** `src/experiments/registry.py`: a list of entries, each naming a base
config and a **delta** of at most a couple of keys.

```python
{"id": "A07_bilstm64", "base": "cached_head",
 "cfg": {"head": "lstm", "hidden": 64, "bidirectional": True},
 "why": "the authors' second best at 87.65%; does it hold up?"}
```

**Code to understand.**

**The delta is the experiment.** `base` names one of a handful of base configs;
`cfg` says what changes. "One variable at a time" becomes visible — anyone
reading the registry can see that A05 differs from the base in exactly one key
— and it becomes *testable*, which is the point.

**A test asserts every delta key exists in `DEFAULTS`.** This is not pedantry,
it is a defect this project has already had: `build_dataset --variants seg`
would have been a **silent no-op**, writing nothing and reporting success,
because nothing validated the argument. `load_config` already raises on unknown
keys for YAML; the registry needs the same guard, in a test, so a typo in
`"droupout": 0.2` fails at collection time instead of producing a row that
looks like a result.

**The run name is derived, never written.** `A07_bilstm64` plus `splits_author`
gives `A07_bilstm64__author`. Hand-written names drift, and the ledger is keyed
on them.

**`why` is a required field.** One sentence, written before the run. An
experiment you cannot state a question for is an experiment whose result you
cannot interpret, and in eighteen days you will not remember what A07 was for.
It also becomes the caption in the thesis table.

**Run.**

```bash
pytest tests/test_registry.py -q
python -m src.experiments.run --list
```

**Check.** 51 IDs (the 46 committed plus the 5-ID probe), 102 run names,
every family represented, and the test suite green on three properties: every
delta key is in `DEFAULTS`, every ID is unique, every entry has a non-empty
`why`.

**Built 2026-10-16.** `src/experiments/registry.py`, `run.py --list`,
`tests/test_registry.py` (20 tests).

1. **Six families, 51 IDs.** A 14, B 8, C 8, D 6, V 10 from Appendix A, plus
   **P, the probe**: one `Linear` on each of the five 8.10 caches, so the only
   variable is the features. P01 is D01's probe fitted by gradient descent
   instead of sklearn; the two should agree within a point, which makes the
   overlap a check rather than a duplicate.
2. **Six bases**, each a YAML plus a few keys: `cached_head` (A),
   `cached_classical` (D), `frozen_mlp` (V), `frozen_linear` (P),
   `mimic_author` (B), `video3d` (C). No delta has more than 3 keys.
3. **Five new `DEFAULTS` keys** for the heads Day 10 builds: `head`,
   `bidirectional`, `layers`, `attn_heads`, `feature_norm`.
4. **The silent no-op, one level deeper.** A key being in `DEFAULTS` is not
   enough: `cached_gru` never reads `backbone`, so a delta setting
   `backbone: resnet50` on it would train on EfficientNet features and write
   `resnet50` into the row. Each method now lists the keys it acts on
   (`READS`), `train.py` lists its own (`LOOP_READS`), and a test fails if a
   delta key is read by neither.
5. **`--list` says what blocks each ID**: `needs method cached_head`,
   `needs cache dinov2_base`, `torchvision has no xception`, or `ready`.
   On 2026-10-16: 6 ready (B01, B04–B08), 45 waiting on code or caches.
6. **B01 resolves to `mimic_author.yaml` exactly** (a test checks every key),
   so its author row can be the 78.14% reproduction instead of a 200-minute
   re-run. A03 is `cached_gru.yaml` with only the method changed.
7. **Paths use `/`**, as the YAMLs and ledger rows do, so the same strings
   work on Windows and Linux.

> **Name the test file `tests/test_registry.py`.**
> `tests/test_experiments.py` is already taken on this branch by the
> `cricketmind` package, and overwriting it would delete a working suite.

**Takeaway.** When a plan has 46 of something, make the 46 *data* and write the
logic once. The alternative — 46 near-identical files — is a design where every
mistake is invisible because there is nothing to compare against.

---

### 9.2 `run.py` — one ID, both splits, resumable (≈1.5 h)

**Why.** Rule 1 says every experiment appends one row to the ledger, and rule 5
says test is touched once at the end. Both are rules about *discipline*, and
discipline is what you lose at 11 p.m. on day fourteen when the power has just
come back. So the harness does it, every time, with no manual step.

**Build.** `src/experiments/run.py`.

```bash
python -m src.experiments.run --exp A07_bilstm64        # both splits
python -m src.experiments.run --exp B01_effnetv2s --split grouped
python -m src.experiments.run --family A                # the whole family
python -m src.experiments.run --family A --dry_run      # what would run
```

**Code to understand.** Five responsibilities, and the ordering between them is
the whole design.

**It runs both splits for one ID**, in sequence, producing two rows. This is
why the registry carries no split — the harness owns that axis, so an
experiment cannot accidentally be run on one split and compared with something
run on the other.

**It evaluates test automatically**, by calling `src/evaluate.py` on the best
checkpoint right after training. Not as a separate command you remember. The
thing to understand: this does not break rule 5. Rule 5 says never *choose*
using test, and the choosing happens in Days 28–30 from the ledger. Scoring it
once, recording it, and not looking at it is what the rule asks for — and
automating it is what makes "once" true.

**It skips an ID whose ledger row already exists.** This is the load-shedding
feature and the reason the harness exists at all. After an outage you press
up-arrow on `--family A` and it continues from the first unfinished experiment.
Two subtleties: the skip is keyed on `(id, split)` not `id`, because an outage
lands between the two splits more often than not; and it must **not** skip an
experiment whose `experiments/<name>/resume.pt` exists — that one gets resumed,
because `resume.pt` present means exactly "interrupted" (`src/train.py` deletes
it on a clean finish, which is what makes the signal trustworthy).

**It writes a short `split` field.** Rule 1 asks for `"author"` or
`"grouped"`; the rows written so far carry the full path
`data/splits_author`, which is fine for a human and useless for grouping a
table. Add the short field, keep the path.

**It never catches an exception to keep going.** A crashed experiment must stop
the family, loudly. A harness that logs the failure and moves on gives you 87
rows and one silent hole, and you will find the hole in week four.

**Run.**

```bash
python -m src.experiments.run --exp P03_r3d_18 --dry_run
python -m src.experiments.run --exp P03_r3d_18
```

**The probe needs a method first.** Every P, A and V entry names
`cached_head`, which does not exist yet. Build it here with three heads,
`linear`, `mlp` and `gru`, so P01–P05 run on Day 9 as 8.10 promised; Day 10
adds the other heads to the same file.

**Check.** `--dry_run` prints two run names, two resolved configs and two split
paths, and writes nothing. The real run produces two ledger rows, two
`report_val.json`, two `report_test.json`, two training-curve figures and
two report cards (9.5) in `notes/figures/runs/`, and no `resume.pt` anywhere.

**Takeaway.** Automate the rules you wrote down, because rules you have to
remember are rules you break under load shedding. Every one of rule 1, 4 and 5
is now a line of code rather than a line of good intentions.

---

### 9.3 Prove the harness is idempotent and deterministic (≈30 min)

**Why.** You are about to trust this thing with eighteen days of work. Two
properties make that safe, and both are one test each.

**Build.** `tests/test_run_experiment.py`, using `limit_clips` and
`max_train_batches` so the whole suite runs in seconds.

**Code to understand.**

**Idempotence**: run an experiment, then run it again. The second call is a
**no-op** — no new ledger row, no re-training. That is the property that makes
`--family A` safe to re-run after every outage, and without a test it is a
property you merely hope for.

**Determinism**: run the same experiment twice with `--force` and the same
seed. Two identical rows. If they differ, something in the pipeline is reading
an unseeded random source, and every comparison in Phase 3 would be measuring
that noise rather than the experiment. `set_seed`, `make_generator` and
`seed_worker` already exist from Day 1 — this test proves they are actually
wired into this path.

**The third test, which is about honesty rather than mechanics**: a row's
recorded `cfg` must reproduce that row. Load the ledger row, rebuild the config
from it, and assert it equals what the registry resolves. A ledger whose `cfg`
does not match what ran is worse than no ledger.

**Run.**

```bash
pytest tests/test_run_experiment.py -q
```

**Check.** Green, in under a minute. Then once by hand, because the test uses a
subset:

```bash
python -m src.experiments.run --exp P03_r3d_18          # should say: already done, skipping
```

**Takeaway.** Idempotence is the property that makes interruption cheap, and it
is worth a test of its own. "Re-running costs nothing" turns a power cut from a
problem into a pause.

---

### 9.4 The leaderboard (≈45 min)

**Why.** `results.jsonl` will have 92 rows in it. A ledger you cannot read is a
ledger you stop updating.

**Build.** `tools/leaderboard.py`, on `read_results`.

**Code to understand.**

**The two splits go side by side and are never averaged.** Section 6 is
explicit: the numbers are not comparable to each other, they answer different
questions. One table with an `author` column and a `grouped` column, and the
*difference between them* as a third — that difference is itself a result,
because it measures how much the authors' protocol flatters a model.

**It prints what is missing, not only what is done.** `92 − len(rows)` and the
IDs behind it. The most useful thing this tool can tell you on day nineteen is
which three experiments never finished.

**It skips the four `smoke_*` rows.** `smoke_frozen`, `smoke_cached`,
`smoke_cached_gru` and `smoke_mimic_author_noflip` are pipeline checks on a
handful of clips, not results. Filter on the `smoke_` prefix, and print how many
were skipped so the filter is never silent.

**It draws no figures.** That is 9.5's job, in one place, so there is one chart
style rather than two.

**Run.**

```bash
python -m tools.leaderboard
python -m tools.leaderboard --family A
```

**Check.** Every row currently in the ledger appears, `mimic_author` at 79.23%
val, and the pending count is right. Re-run it at the end of every day in
Phase 3 — that is the habit the tool exists to support.

**Takeaway.** A results table that shows the gaps is more useful than one that
shows the wins, because the gaps are the thing you can still act on.

---

### 9.5 Figures for every experiment (≈1.5 h)

**Why.** 92 runs is too many to judge from a table, and a figure made by hand
at the end is a figure made from whatever you remember. So every run draws its
own figures the moment it finishes, and every comparison figure is rebuilt from
the ledger by one command. Nothing is drawn by hand, nothing is drawn twice, and
Day 33's package already exists by then.

**Build.** Two levels.

**Per run, automatically.** `run.py` calls these right after evaluation, for
each split:

| figure | what it shows | made by |
|---|---|---|
| training curves | loss, accuracy, lr, gradient norm per epoch, lr cuts marked | `plot_run` (exists, Day 7) |
| report card | 15×15 confusion matrix (each row sums to 100%) beside per-class F1, sorted worst to best | `plot_report` (new, in `src/plot.py`) |

**Across all runs, on demand.** `tools/figures.py` reads `results.jsonl` and
each run's `report_val.json`, and writes to `notes/figures/compare/`:

| figure | the question it answers |
|---|---|
| one per family (A, B, C, D, V) | did each change beat its base? One dot per ID, the base as a line, author and grouped as two panels |
| split gap | how much does the authors' protocol flatter each model? Author vs grouped as a dumbbell per ID |
| per-class F1 heatmap | which change helped which shot? IDs × the 15 classes |
| cost vs accuracy | which runs earn their GPU minutes? `minutes` against val top-1 |
| probe ranking | which of the five caches carries the most shot information? |

Plus `notes/figures/index.md`, written by the same command: every figure with
its run name and the registry's `why` line as its caption.

**Code to understand.**

**Validation only — rule 5 reaches the figures too.** A test confusion matrix
on every run is test looked at 92 times. Every figure here is drawn from
`report_val.json`. Test figures are drawn once, for the chosen models, on
Days 32–33. `report_test.json` is still written by 9.2; it is just never
plotted before then.

**Row-normalised, with the counts kept.** Classes differ a lot in size: on the
author val split Lofted Legside has 178 clips and Reverse Sweep 39. In raw
counts the big classes' mistakes swamp the picture. Each row is shown as a
percentage of that true class, and the support goes in the row label,
`Reverse Sweep (n=39)`, so a small class is never mistaken for a reliable one.

**One style, already validated.** Import `THEMES` and the palette from
`src/plot.py`: one axis per panel, never a twin axis. The confusion matrix uses
a single-hue sequential scale (light to dark in one colour), not a rainbow, so
darker always means more. Two splits are two panels, never averaged.

**Rebuilt, never edited.** `tools/figures.py` overwrites everything it owns on
every run. If a figure is wrong you fix the code and rerun; you never touch the
PNG. That makes every figure in the thesis reproducible by one command.

**Git keeps the light version of per-run figures.** 92 runs × 2 figures ×
2 themes is about 60 MB of PNGs. The dark copy is still drawn into
`experiments/<name>/`; only the light one is mirrored into git. Comparison
figures are few, so both themes are kept.

**Backfill.** The runs that already exist get report cards too. Those with a
`report_val.json` (`mimic_author`, `mimic_author_seg`) are drawn straight away.
The other three (`mimic_author_noflip`, `cached_gru`, `cached_gru_noflip`) need
`src/evaluate.py` run on their `best.pt` first — one command each, a minute or
two.

**Run.**

```bash
pytest tests/test_plot.py tests/test_figures.py -q
python -m src.plot --report experiments/mimic_author
python -m tools.figures
```

**Check.** A report card for every run with a `report_val.json`; one figure per
family that has rows, the rest reported as "no rows yet", never an empty chart;
`index.md` lists every figure; and rerunning `tools.figures` gives byte-equal
PNGs (the same check as 9.3's determinism, for figures).

**Takeaway.** Make the figure part of the run, not part of the write-up. A
figure drawn automatically from the ledger cannot drift from the number it
shows, and on Day 33 you are choosing figures rather than making them.

---

### 9.6 The first real row: the grouped-split baseline (≈10 min to start, then ≈3–4 h)

*A launch step, and the most valuable single run left in the project.*

**Why.** Every number this project has produced — all seven rows in
`experiments/results.jsonl` — is on `splits_author`. The grouped split was
built on Day 1, committed, and never trained on. That is the largest missing
number in the whole plan, and it is missing at the exact point where it would
change how the results are framed.

**What it is worth, measured 2026-10-15.** Source-video overlap between train
and test:

```
splits_author    382 of 384 test source videos also appear in train   (99.5%)
splits_grouped     0 of 102 test source videos appear in train        (0.0%)
```

No `(video, class)` pair is in both — their rule does hold, and the label
itself is not leaked. But 99.5% of the test *footage* was seen during training:
same ground, same batter, same kit, same camera, same broadcast compression.
**Their 89.09% is an in-domain measurement.** It answers "can you classify a
shot from a match you have already studied", not "from a match you have never
seen". Both are legitimate questions. Only one of them is the question the
paper claims to answer.

You cannot write that sentence in a thesis without the second number.

**Run.**

```bash
python -m src.experiments.run --exp B01_effnetv2s --split grouped
```

End-to-end, not cached — it has to match `mimic_author`'s configuration exactly
to be comparable, and that configuration has a trainable backbone. The splits
are nearly the same size (6,341 usable train clips against 6,320), so budget ~200
minutes, and expect one or two power cuts inside it. `resume.pt` handles them.

**Check.** One ledger row with `split: "grouped"`, and a number **below**
78.14%. If it comes out *above*, stop and check the split files before
believing it — rule 7.

**What to do with it.** This is the headline of the write-up, and it is worth
more than beating 89% on a protocol with 99.5% footage overlap:

> We reproduce 78.14% under the authors' protocol and reach X% with Y. We also
> report the first grouped-split result on CricShot10k (Z%), and show that
> 99.5% of test source videos appear in the authors' training set, making their
> reported figure an in-domain measurement.

**Takeaway.** When a published protocol has a gap, the valuable contribution is
usually not a bigger number under the same protocol — it is the number nobody
has measured, next to theirs, with the difference explained.

---

### 9.7 Commit and tag Phase 2 (≈15 min)

*A commit step.*

```bash
git add -A && git commit -m "Phase 2: feature cache (13x), experiment harness, figures for every run, first grouped-split baseline"
git tag phase2-complete
```

**What exists at this point, and it is worth listing because Phase 3 assumes
all of it:**

| | |
|---|---|
| `data/features/effnetv2s/` | 35 GB, four views, reproduces 79.23% from tensors |
| four more caches | ConvNeXt-T, r3d_18, VideoMAE-B, CLIP; ~400 MB; linear-probe ranking recorded |
| `src/extract.py` | resumable, verified to fp16, `--mirror` and `--derive` |
| `src/data/features.py` | `FeatureDataset`, three views, shares `classes.json` |
| `src/experiments/` | 46 IDs as data, both splits per ID, idempotent, test-eval automatic |
| `tools/leaderboard.py` | both splits side by side, pending count, smoke rows skipped |
| `tools/figures.py` | per-run report cards, family / gap / heatmap / cost / probe figures, `index.md` |
| the grouped baseline | the number the authors never reported |

**What Phase 2 does not solve, so Phase 3 plans around it.** The cache only
helps experiments with a **frozen** encoder. Families A, D and V are 28 of the
46 IDs, so **60 of the 92 runs are now minutes each**. Families B and C are
end-to-end by definition — 16 IDs, **32 runs at the full ~200 minutes** — and
that is where every remaining GPU-hour goes. The table in
[section 6](#why-running-all-46-twice-is-affordable) costs it out: 126 hours in
total, of which 120 belong to those 32 runs.

Continuous augmentation also stays out of reach. Tier 2's colour jitter, speed
perturbation and random resized crop cannot run on cached features, for the
reason in 8.7: infinitely many transformed inputs, and a cache can only hold
finitely many. Those rows have to be end-to-end or not at all.

---

# PHASE 3 — The 46 experiments, twice (Days 10–27)

Full list with IDs in **Appendix A**. Order is deliberate: cheap and informative
first.

Every experiment runs on **both splits**, so each ID produces two ledger rows.

| Day | Family | Experiments | Rows | GPU time |
|---|---|---|---|---|
| 10 | **A** — temporal heads | A01–A14 (14) | 28 | 4.2 h |
| 11 | **D** — features + classical ML | D01–D06 (6) | 12 | 1.0 h |
| 12 | **V** — video-pretrained features (frozen) | V01–V10 (10) | 20 | 3.3 h extract + 1.0 h |
| 13–18 | **B** — CNN backbones | B01–B08 (8) | 16 | 48 h |
| 19–27 | **C** — 3D / video models (fine-tuned) | C01–C08 (8) | 16 | 72 h |
| stretch | **E** — pose and graph | E01–E04 (4) | 8 | 24 h |
| stretch | **F** — multi-scale and hybrid | F01–F05 (5) | 10 | 30 h |
| stretch | **W** — cricket-specific motion | W01–W05 (5) | 10 | 35 h |

**60 listed, 46 committed, 92 rows.** Days 10–12 alone give you **60 rows** for
about 10 GPU-hours, because they reuse the cached features. Families **E, F and W
remain stretch** — another 89 GPU-hours between them, and the weakest expected
payoff of the eight.

The day ranges assume roughly 8 usable GPU-hours a day. Count the GPU-hours
column, not the days, and re-divide for your own power situation.

**Do the cached families first**, for the reason in section 6: the 28 cheap
experiments tell you whether the leak gap depends on the architecture, and that
answer decides how much the 60 extra hours on B and C are really buying.

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

# PHASE 4 — Combine and improve (Days 28–32, ~41 GPU-hours)

Only now do you look at the leaderboard and pick.

## Days 28–30 — Choose honestly, then tune the winner

**~39 GPU-hours.** Re-running the top 3 with 3 seeds is 9 trainings, and if the
winner is an end-to-end model that alone is ~27 hours. Budget three days, not
one.

**First, choose.** Take the top 5 by **validation** accuracy. For each, ask: is
it genuinely better, or just lucky? Re-run the top 3 with **three different
seeds** and report mean ± std.

> A 0.4% gap with ±0.8% std is not an improvement. This is the single most common
> way student papers get rejected.

**Then tune**, only the winner, one axis at a time:

- learning rate: 3e-5, 1e-4, 3e-4
- frames: 15, 20, 25
- backbone LR multiplier: 0.1, 0.5, 1.0
- label smoothing: 0.0, 0.05, 0.1
- class-balanced loss (see Appendix B — your minority classes are 4× smaller)

**Check:** every tuning run is also a ledger row.

## Day 31 — Ensembles and test-time augmentation

- Soft-vote the top 3 (different backbones ensemble better than similar ones)
- TTA: horizontal flip + 2 temporal crops
- Weighted vote, weights chosen on **val**

**Check:** the ensemble beats its best member on val before you touch test.

## Day 32 — Error analysis

- Full 15×15 confusion matrix for the best model
- The 20 worst-misclassified clips — **watch them**
- Is Late Cut still the worst class? Did your model fix it or move the problem?
- Grad-CAM on 5 correct and 5 incorrect clips: is it looking at the bat or the
  crowd?

This section is what turns a results table into a paper.

---

# Day 33 — The package for your supervisor

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
*Answers: how much does the temporal model matter? Minutes each on the `avg`
view; **70 min each on `flat`**, measured 2026-10-15 — a `62720 x 384` GRU
kernel is 24.1 M parameters and an epoch reads 11.0 GB. A13 and A14 are there
to find out which of these heads can drop to `avg` without losing anything.*

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
| A13 | **LayerNorm on the features, then GRU-128** | **39% of the GRU is dead at epoch 1** on frozen `flat` features (measured 8.6; 41% on Day 7's video run). The Day 7 diagnosis was specific: 62,720 non-centred post-SiLU values summed into a tanh gate. Centring the input is the direct fix, and a third of the model's capacity is the prize. |
| A14 | **GRU-128 on the `avg` view** (1,280, not 62,720) | The other half of the same question. `flatten` keeps the 7x7 grid -- bat position relative to the body, which `encoder.py` argues is much of the signal -- but it is also *what saturates the GRU*: 49x more inputs to sum is ~7x larger pre-activations under random init. **Information preserved against capacity wasted, and nobody has measured which wins.** Also the view Family A's time budget depends on: `flat` measured 70 min a run against the 3.6 min budgeted. |

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

## Family E — Pose and graph (stretch)
*2–4 hours each. Answers: does explicit body structure help?*

**Stretch.** Moved out of the committed set when Phase 1 grew to seven days.
Do these only if Phase 1 finished early; otherwise they are future work, and
you say so in the write-up.

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

## Family G — Ensembles (Day 31)

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

| # | What | Why | Predicted | **Measured** |
|---|---|---|---|---|
| 1 | **Crop to striker + bat** (YOLOv11) | Removes crowd, boards, grass | +10% | **+3.2%** (74.93 → 78.14) |
| 2 | **Segmentation overlay** (striker blue, bat green, α=0.5) | Removes jersey-colour noise across formats | +3% | **−1.2%** (76.97) |
| 3 | **Horizontal flip**, training split only, after the split | Simulates left/right-handed batters | +1.5% | **+3.3%** |
| 4 | **Grouped splits by `vidNNN`** | Same match in train and test is a leak | — | not yet run |

Measured 2026-10-13 on `splits_author`, test top-1. Three notes:

- **#1 under-delivered** — but it was also entangled with the two framework
  bugs in section 0. Cropping was applied at the same time as the broken GRU
  init, which is why the first cropped run scored 32.39%.
- **#2 did not work for us at all.** The spec is right — it is written out in
  the paper's method section, and implementing it exactly moved the authors'
  own model +8.76 — but our masks give only 65% bat coverage, so the cue it
  depends on is absent from a third of frames. Correct recipe, insufficient
  masks.
- **#3 over-delivered**, and matches their own ablation direction (+2.37).
  Keep it. Removing it costs 3.28 points and opens a 15-point train/val gap.

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
| 13 | **Dropout 0.2–0.3 before the classifier** | The authors use 0.0, and the baseline ends at train 86.88% / val 79.23% — an **8.5-point gap** with no regulariser but flip. The most likely single-run gain on the list. |
| 14 | **Tighter crops** (`pad` 0 or 0.125 instead of 0.25) | The authors' model reads our crops monotonically better as padding tightens: 37.6% at pad 0 against 33.0% at 0.25. Needs a rebuild, so pair it with another change. |

## Tier 3 — Try, but expect nothing

| # | What | Caution |
|---|---|---|
| 15 | MixUp / CutMix | **You already found this hurts.** Mixing Pull with Hook makes an ambiguous label. Keep the negative result — it is publishable. |
| 16 | RandAugment | Aggressive policies destroy fine-grained stroke differences |
| 17 | Heavy rotation / vertical flip | A batter never plays upside down; this is nonsense augmentation |

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
| The two splits give nearly identical accuracy | Either the leak is genuinely small, or `splits_author/` was built grouped by mistake. Check the straddling-match count from step 1.6 before believing it |
| `splits_grouped/` scores much lower | Expected. That is the finding, not a bug — provided step 1.6's leak check passed |

---

## The schedule in one table

| Days | Phase | What you build | Output |
|---|---|---|---|
| 1 | Foundations | `device.py`, `seed.py`, `splits.py` | grouped splits, committed |
| 2 | Crash safety + video | `checkpoint.py` read, `video.py` | clips readable, runs resumable |
| 3 | Finding the striker | `detect.py`, `crop.py` | 30 crops you have examined |
| 4 | Pipeline + launch | `sample.py`, `segment.py`, `build_dataset.py` | `data/processed/` |
| 5 | Data + model | `dataset.py`, `encoder.py`, `head.py` | one batch, loss ≈ ln 15 |
| 6 | Training loop | `train.py`, `mimic_author.yaml` | overfits 50 clips; resume rehearsed |
| 7 | Reproduce | `evaluate.py` with hash guard | **78.14% baseline, tagged** (89.09% not reachable; see 7.4) |
| 8–9 | Feature cache + harness | `extract.py`, `features.py`, `experiments/run.py`, `leaderboard.py` | 35 GB cache serving both splits; 92 runs as data; **first grouped-split baseline** |
| 10–12 | A, D, V on both splits | — | **60 rows for ~10 GPU-h**; is the gap architecture-dependent? |
| 13–18 | B — CNN backbones, both splits | — | 16 rows, 48 GPU-h |
| 19–27 | C — 3D / video models, both splits | — | 16 rows, 72 GPU-h |
| 28–30 | Choose (3 seeds × top 3) and tune the winner | — | best model + mean ± std, ~39 GPU-h |
| 31 | Ensembles + TTA | — | reuses existing checkpoints, ~2 GPU-h |
| 32 | Error analysis | — | per-class failures, the leak gap explained |
| 33 | Package for supervisor | — | leaderboard, figures, summary |

---

*Plan for the CricShot10k accuracy work. The CricketMind / LangGraph work is
paused while this runs.*
