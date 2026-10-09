# CricShot10k — The Plan

**Goal.** First reproduce what Dihan et al. achieved (89.09% top-1), from code
you wrote. Then run 44 different approaches **on two different data splits** and
record all 88 results. Then combine and tune the best ones.

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
  preprocess/
    detect.py        Day 3  YOLO: find the striker and the bat
    crop.py          Day 3  one clip -> cropped frames
    sample.py        Day 4  any length -> 15 frames
    segment.py       Day 4  colour the striker and bat
    build_dataset.py Day 4  run all of it over 10,091 clips, resumably
  models/
    encoder.py       Day 5  EfficientNetV2-S over each frame
    head.py          Day 5  GRU-128 -> Dense -> 15 classes
  train.py           Day 6  the training loop
  evaluate.py        Day 7  test-set metrics, with a checkpoint hash
configs/
  mimic_author.yaml  Day 6  the authors' exact settings
tests/               every step that can be tested gets a test here
notes/journal.md     one entry a day
```

Twelve modules, none of them long. The order matters: each one is only built
once the thing before it works, so nothing is written against an interface that
might still change.

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

### Why running all 44 twice is affordable

**A frozen backbone's features depend only on the clip, not on the split.** One
feature cache serves both; changing the split is a different list of filenames
over the same files.

| family | n | h each | 1 split | both splits |
|---|---|---|---|---|
| A temporal heads | 12 | 0.06 | 0.7 h | 1.4 h |
| D classical ML | 6 | 0.08 | 0.5 h | 1.0 h |
| V video-pretrained | 10 | 0.05 | 0.5 h | 1.0 h |
| B CNN backbones | 8 | 3.0 | 24 h | 48 h |
| C 3D / video | 8 | 4.5 | 36 h | 72 h |
| **total** | **44** | | **62 h** | **123 h** |

The 28 cached experiments cost **1.7 extra hours** for 28 extra rows. The 16
end-to-end ones cost 60 extra hours, which is the real price of this decision.

### Run the cached families first, and here is why

It is not just about getting easy rows early. The 28 cached experiments
**measure whether the leak gap depends on the architecture**:

- If the gap is stable across all 28 -- say 6.2% ± 0.8% whatever the head -- it
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

Two days longer than the original plan, because building from nothing with the
code explained is slower than editing what was there. The two days come out of
Phase 3, where Family E joins F and W as stretch work — that still leaves **44
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
cannot tell a real improvement from luck. With 44 experiments coming, that
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

**Why.** 44 experiments are coming. Finding out in week three that rows were
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
> **~11 days of GPU for 88 runs**. Fine-tuning end to end every time is not
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

---

## Day 7 — Train it, and hit the number

### 7.1 Launch the author replication (≈30 min to start, then 3–6 h)

**Run.**

```bash
python src/train.py --config configs/mimic_author.yaml
```

**Check at epoch 5.** Val accuracy should already be past 60%. If it is stuck
near 7% the labels are shuffled. **If it is near 75%, you are reading uncropped
data** — check the processed path, because that is exactly the old failure.

**Takeaway.** The 75% number is now a diagnostic. If you ever see it again, it
means the crops are not being read.

### 7.2 Horizontal-flip augmentation (≈1 h)

**Why.** It is in the authors' pipeline and worth about 1.5 points.

**Build.** Offline flip, **training split only**, applied *after* the split.

**Code to understand.** Why flipping after the split is essential: flip first
and a clip's mirror image can land in test while the original is in train — a
leak that is very hard to spot later. Also why flipping is safe for cricket
shots at all (it turns a right-hander into a left-hander, which is a real thing
that happens) while a vertical flip would not be.

**Check.** The training set roughly doubles; val and test are untouched.

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

The authors' per-class sanity checks matter as much as the headline:

- **Late Cut** is their worst class (F1 ≈ 0.745) — it looks like Upper Cut and
  Square Cut.
- **Scoop** is their best (recall ≈ 1.0) — the most visually distinct shot.

If your confusion matrix shows the same pattern, you have genuinely reproduced
their model. If it does not, something differs even when the headline matches.

**Takeaway.** Always reproduce a known number on val before computing an unknown
one on test. Test is touched once, at the end.

### 7.4 If you are short of 89% (≈2–4 h)

In order of likelihood:

1. Crops are wrong → back to Day 3.4 and look at the images again.
2. Flip augmentation missing, or applied before the split.
3. Too few epochs — they converged around epoch 30 of 100.
4. Backbone accidentally frozen — the authors fine-tune the whole thing.
5. Wrong normalisation (must be ImageNet mean/std).
6. BGR/RGB swapped in `video.py` — costs a few points, silently.

### 7.5 Write the baseline row and tag it (≈30 min)

```bash
git add -A && git commit -m "Reproduce CricShotNet baseline: <your> % top-1"
git tag baseline-reproduced
```

**This is the single most valuable artifact of the whole project.** Everything after it
is measured against it, and you built every line of it yourself.

---

# PHASE 2 — Make 88 runs affordable (Days 8–9)

Thirty experiments at 4 hours each is 120 hours. You do not have that. The trick
is that most of them do not need to touch the video at all.

## Day 8 — Cache the CNN features

### 8.1 Extract once (≈2 h + 1 h compute)

Run the frozen EfficientNetV2-S over every processed clip and save a
**(15, 1280)** feature tensor per clip.

```
data/features/effnetv2s/<class>/<clip>.pt
```

**Flat, not split-organised.** A frozen backbone's features depend only on the
clip, so one cache serves both splits — the split is just a list of filenames
over these files. Writing `{train,val,test}/` into the path would force a full
re-extraction every time the split changed, which is exactly the trap, since
every experiment now runs on two splits.

**Check:** 10,091 files, each (15, 1280) float16; about 600 MB. Then confirm the
same cache loads correctly under both `splits_author/` and `splits_grouped/`.

### 8.2 Why this changes everything

Any experiment that only replaces the **temporal head** — GRU, LSTM, BiLSTM,
transformer, TCN, attention pooling, SVM, XGBoost — now trains on these features
in **1–3 minutes instead of 4 hours**, because the CNN never runs again.

That is Family A and Family D below: roughly 17 of the 30 experiments, finishable
in a single day.

**Check:** retrain GRU-128 on the cached features alone. It should land within
~1 point of Day 5's number. If it is far off, the cache is wrong.

> **You will learn:** the difference between end-to-end training and
> feature-then-head, and why the second is the right tool for an ablation sweep.

### 8.3 Cache more extractors (≈2 h)

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

## Day 9 — The experiment harness

### 9.1 One command per experiment (≈3 h)

```bash
python -m src.experiments.run --family A --exp A03_bilstm
```

It must: load the config, set the seed, train, evaluate, append one row to
`experiments/results.jsonl` via `append_result`, and save the confusion matrix
PNG. No manual steps — manual steps are how you lose six results in week three,
and an outage mid-experiment is how you lose the seventh.

**Check:** run the same experiment twice. Two identical rows (same seed) means
the harness is deterministic.

### 9.2 A results notebook (≈1 h)

Reads `results.jsonl` with `read_results`, prints the leaderboard sorted by val
accuracy, and plots accuracy per family. Re-run it at the end of every day.

---

# PHASE 3 — The 44 experiments, twice (Days 10–27)

Full list with IDs in **Appendix A**. Order is deliberate: cheap and informative
first.

Every experiment runs on **both splits**, so each ID produces two ledger rows.

| Day | Family | Experiments | Rows | GPU time |
|---|---|---|---|---|
| 10 | **A** — temporal heads | A01–A12 (12) | 24 | 1.4 h |
| 11 | **D** — features + classical ML | D01–D06 (6) | 12 | 1.0 h |
| 12 | **V** — video-pretrained features (frozen) | V01–V10 (10) | 20 | 3.3 h extract + 1.0 h |
| 13–18 | **B** — CNN backbones | B01–B08 (8) | 16 | 48 h |
| 19–27 | **C** — 3D / video models (fine-tuned) | C01–C08 (8) | 16 | 72 h |
| stretch | **E** — pose and graph | E01–E04 (4) | 8 | 24 h |
| stretch | **F** — multi-scale and hybrid | F01–F05 (5) | 10 | 30 h |
| stretch | **W** — cricket-specific motion | W01–W05 (5) | 10 | 35 h |

**58 listed, 44 committed, 88 rows.** Days 10–12 alone give you **56 rows** for
about 7 GPU-hours, because they reuse the cached features. Families **E, F and W
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
| 7 | Reproduce | `evaluate.py` with hash guard | **89% baseline, tagged** |
| 8–9 | Feature cache + harness | `extract.py`, `run.py` | one cache, serving both splits |
| 10–12 | A, D, V on both splits | — | **56 rows for ~7 GPU-h**; is the gap architecture-dependent? |
| 13–18 | B — CNN backbones, both splits | — | 16 rows, 48 GPU-h |
| 19–27 | C — 3D / video models, both splits | — | 16 rows, 72 GPU-h |
| 28–30 | Choose (3 seeds × top 3) and tune the winner | — | best model + mean ± std, ~39 GPU-h |
| 31 | Ensembles + TTA | — | reuses existing checkpoints, ~2 GPU-h |
| 32 | Error analysis | — | per-class failures, the leak gap explained |
| 33 | Package for supervisor | — | leaderboard, figures, summary |

---

*Plan for the CricShot10k accuracy work. The CricketMind / LangGraph work is
paused while this runs.*
