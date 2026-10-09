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

---

## 2026-10-09 (3.1) — the detector was running at 224px

**Goal.** Step 3.1: run the authors' YOLO on one frame and see what comes back.
Also settle, with the detector rather than by eye, whether the frames after a
camera cut contain a batter.

**Broke.** The detector barely worked. Striker found in a mean of **48%** of
frames across 40 random clips, ranging from **0%** to 96%. On
`Cover Drive/vid302_23.avi` it found the batter in **0 of 24 frames**, in a clip
where the batter is plainly visible and reasonably large.

**Cause.** Ultralytics takes `imgsz` from the checkpoint, and this checkpoint
says **224**. So every 896x540 broadcast frame was being squashed to 224px
before detection, and the batter became a few pixels. Nothing warns you; it
just quietly finds less.

```
clip            imgsz=224 (default)    640      896 (native)
vid302_23            0/24              9/24       17/24
vid148_8             1/24             18/24       24/24
vid412_12            1/24              8/24       20/24
vid1_0              11/29             24/29       27/29
```

**Fixed.** `src/preprocess/detect.py` pins `IMGSZ = 896`. Striker detection over
25 clips went from mean 48% (min 0%) to **mean 92%, median 96%, min 71%**.

Two smaller fixes fell out of it. At 896 the segmentation head exhausted the
Arc mid-run (`UR_RESULT_ERROR_OUT_OF_HOST_MEMORY` inside `process_mask`), so I
dropped the batch to 4 and emptied the cache after each clip. **Both halves of
that were wrong**, see the next entry. And ultralytics reads numpy arrays as BGR,
so `detect()` converts from the project's RGB convention first -- handing it RGB
gives 22 striker detections over 16 frames at mean confidence 0.522 instead of
17 at 0.767, i.e. more boxes, less often right.

**I was also wrong about the camera cuts.** With a working detector, over 30
clips with a cut and 20 without:

| | striker found in |
|---|---|
| no-cut control | 94.4% of frames |
| before the cut | 90.4% |
| after the cut | **71.0%** |

Only 4 of 30 cut clips (13%) have no striker at all after the cut. I had
generalised "the post-cut frames contain no batter" from two clips I happened to
look at, and that is false — most cuts go to another angle that still shows the
batter. The reason to trim at a cut is the viewpoint discontinuity, not an
absent batter. The plan has been corrected.

**Open.** Nothing blocking. But note how this one was found: not by a failing
test, but by rendering a clip the detector had scored 0% on and seeing a batter
in it. Numbers that disagree with a picture are worth chasing.

---

## 2026-10-09 (3.1, corrected) — the OOM was another model, and my fix cost 3.4x speed

**Goal.** Check a suggestion: that the Arc running out of memory was caused by
another project's model using the GPU at the same time, not by my batch size.

**Broke.** My own fix. I had reacted to one OOM by dropping the batch to 4 *and*
calling `torch.xpu.empty_cache()` after every clip, without measuring either.

**Cause.** The GPU was idle when I checked: **10.88 GB free of 11.67**. So the
earlier crash was contention, as suggested. Benchmarked on the idle card, 77
frames at imgsz=896:

| batch | empty_cache | fps | peak GPU |
|---|---|---|---|
| 4 | True | 6.3 | 1.03 GB |
| 4 | **False** | **21.6** | 1.02 GB |
| 8 | False | 20.1 | 1.81 GB |
| 16 | False | 21.5 | 4.05 GB |
| 24 | either | fails | — |

`empty_cache` was costing **3.4x speed and saving nothing** — peak memory is
identical with and without it, because torch reuses its cached blocks anyway.
Batch size barely affects speed at all; only memory. And batch 24 fails even
with the card to itself, so that limit is real.

**Fixed.** Removed the `empty_cache` call. Kept batch 4, now for a reason that
holds up: it is the cheapest of the equally-fast options, which leaves the card
free for whatever else is running.

Over 40 clips end to end: **17.5 fps, peak 1.03 GB, no accumulation**. That puts
Day 4's detection pass at about **4 hours** instead of roughly 11.5.

**Open.** Nothing. The lesson is the cost of fixing a symptom without measuring:
a single crash with an unexamined external cause produced a change that would
have added seven hours to the longest job in the project.

---

## 2026-10-09 (3.2, before writing it) — the plan's own selection rule picks a fielder

**Goal.** Look at real detections before writing the striker-selection logic,
instead of coding the rule the plan proposed and testing it afterwards.

**Broke.** The proposed rule, on the first frame I tried. `Sweep/vid306_8.avi`
frame 5 returns **6 detections for one batter and one bat** — 3 Strikers, 3 Bats:

```
Bat      0.55  centre ( 867, 165)   1.7% of frame
Bat      0.54  centre ( 211, 259)   1.6%
Striker  0.45  centre ( 390, 221)  10.0%   <- the real batter
Bat      0.39  centre ( 218, 261)   2.7%
Striker  0.30  centre ( 865, 128)   3.2%   <- a fielder, detected twice
Striker  0.30  centre ( 872, 112)   2.2%
```

"The striker is the person nearest the bat" picks the fielder. The strongest
detection in the whole frame is a Bat at 0.55 standing next to him, 480 px from
the batter:

| rule | picks | right? |
|---|---|---|
| largest striker box | batter | yes |
| highest-confidence striker | batter | yes |
| nearest the centre of frame | batter | yes |
| nearest the best bat | **fielder** | **no** |

Frame 10 of the same clip repeats it — bat 115 px from the wrong person, 147 px
from the right one — while area (41716 vs 25393) and confidence (0.71 vs 0.35)
both pick correctly.

**Cause.** The bat is 1.7% of the frame against the batter's 10%, so bat
detection is the *less* reliable of the two. Keying selection on it puts the
weak signal in front of the strong one and inherits its false positives.

**Fixed.** Not yet — 3.2 is deferred until the GPU is free. The plan's 3.2
section now carries this measurement and the rule it points to: score Strikers
on area, confidence and centrality, with bat overlap as confirmation only.

**Open.** Two frames is not a sample. Measure the candidate rules over a few
hundred frames across classes before settling, and keep the 20-frame eyeball
check as the acceptance test. Also unresolved: the keeper, who is large and
central in some angles and would beat the batter on those scores.

**Note.** The GPU was busy (**8.11 GB of 11.67 in use, 1.63 GB RAM free**) so
this ran on CPU for a single frame. Fine for looking; not for 10,091 clips.

---

## 2026-10-09 (3.2) — the batter is the one in the middle, and nothing else comes close

**Goal.** Pick the striker among the detector's candidates, choosing the rule
from measurement rather than from the two frames I happened to look at.

**Broke.** My own claim from yesterday. On two frames, largest-box and
highest-confidence both found the batter, so I said they worked. Over 34
hand-labelled frames they are the two **worst** rules tested.

**Cause.** Two frames is not a sample. Across 720 frames from 120 train clips:

| | |
|---|---|
| frames with 0 Strikers | 9.0% |
| frames with exactly 1 | 40.0% |
| frames with 2 or more | **51.0%** |

So selection is needed on half of all frames, and on those the rules disagree
wildly — largest-box and highest-confidence pick different people on **45.8%**
of them. At most one of them could have been right.

**Fixed.** Labelled 36 random multi-striker frames by eye (train split only;
eyeballing test clips to tune a rule is peeking), numbering every Striker box
and recording which one is the facing batter. Scored every candidate rule, a
pick counting as correct at IoU >= 0.5 so a duplicate box of the same person
is not punished:

| rule | correct |
|---|---|
| **nearest the frame centre** | **33/34 = 97%** |
| nearest the best bat | 30/34 = 88% |
| largest box | 26/34 = 76% |
| highest confidence | 23/34 = 68% |

Adding area and confidence on top of centrality made it *worse* (32/34).

The reason centrality wins is visible in the labels: the true batter's box
centre sits at **x = 0.44 to 0.54** of frame width in 80% of frames, median
0.50. The dataset is already built around the batter. That is a fact about
CricShot10k, not about cricket — the rule will not transfer to uncropped
broadcast footage, and `striker.py` says so in its docstring.

Shipped `src/preprocess/striker.py` (54 lines) + 13 tests. Verified the module
reproduces 33/34 rather than merely resembling the thing I measured.

**Open.**
- 34 frames is a small sample. The gap from 97% to 76% is far too large to be
  noise; the gap from 97% to the 94% combinations is not, so I did not pick
  between them on that evidence. Worth a second labelled batch before Day 4.
- Sweeping the anchor down to (w/2, 0.55h) scores 34/34. That is one frame, so
  I left the anchor at the true centre rather than tune a constant on 34
  samples.
- **F19 is the one failure and it is a near miss**: the batter is 131 px from
  centre, a fielder 113 px. 18 px decided it. Only 3 of 34 frames are decided
  by under 20 px (median margin 286 px), so most picks are not close calls.
- On 2 of 36 frames (6%) the detector never found the batter at all. No
  selection rule can help there; 3.3 has to handle it.

---

## 2026-10-09 (3.3) — cropping works, and it costs 7% of the dataset

**Goal.** Turn 24 independent striker picks into one steady cropped clip.

**Built.** `src/preprocess/crop.py` (148 lines, 19 tests). Four stages, in this
order because each changes what the next sees:

1. **Reject rogues.** A pick more than one body-height from the clip's median
   centre is a fielder in a crowd shot, not the batter. Threshold measured, not
   guessed: over 40 clips, strays are median 0.21 body-heights, 90th pct 0.76,
   99th 2.30. A cutoff of 1.0 rejects 8.0% — which independently matches the
   9.3% of frame-to-frame jumps over 150 px found the same day.
2. **Choose a span.** Gaps of 1–2 frames are the detector blinking and get
   interpolated; a longer gap is a camera cut and the clip stops there.
3. **Smooth.** Median filter (width 5) then mean (width 3).
4. **Square and pad.** One side for the whole clip — a per-frame side is what
   makes the crop pulse.

**Broke.** My first padding measurement, by using the highest-confidence bat.
It said the bat pokes outside the striker box in **96%** of frames by a median
of 35% of box size, with a 90th percentile of 174% — absurd. Cause: 3.2 had
already shown the strongest bat detection is often a false positive on a
hoarding, and I measured against it anyway. Using the **nearest** bat instead,
and discarding bats more than one body-height away (11% of them), gives a sane
distribution: median overhang 27%, 90th pct 61%.

Then measured against the geometry actually used — a *square* crop, since the
model wants 224x224 and squaring a tall thin person box adds width exactly
where the bat is:

| padding | bat stays whole | batter fills | crop needs nudging |
|---|---|---|---|
| 0% | 37% | 100% | 0% |
| 20% | 77% | 71% | 5% |
| **25%** | **84%** | **67%** | **8%** |
| 30% | 88% | 62% | 12% |
| 40% | 93% | 56% | 22% |

Took 25% as the knee.

**Validated.** Two clips whose cut positions were measured independently on Day
2 come out right without being told: `Cover Drive/vid1_0.avi` 29 -> 20 frames
(Day 2 found the cut at frame 20) and `Sweep/vid306_8.avi` 24 -> 16 (cut at
frame 16). Two separate methods, same answer.

**Open — and this one matters for Day 4.** Over 60 clips:

| | |
|---|---|
| median frames kept | 20 |
| 10th percentile | 10 |
| clips dropped entirely (under 8) | **4/60 = 7%** |
| clips with at least 15 frames left | **75%** |

7% is roughly **700 of 10,091 clips thrown away**, and a quarter of the rest
have fewer than the 15 frames Day 4 wants to sample. The cause is that
`MAX_GAP` treats any gap over 2 frames as a cut, when some are just the
detector failing on a continuous shot. The principled fix is to trim at an
**actually detected camera cut** (the frame-difference measure from Day 2,
already written) and interpolate across detector failures of any length inside
a continuous shot. Do that before Day 4 depends on these numbers.

**Also open.** Subject switching: the crop walks off the batter onto a fielder
in 0.4% of frame pairs, affecting 1/40 clips. Rare, but `Flick/vid213_40.avi`
shows it plainly — frames 12-17 are centred on a fielder with the batter sliced
off at the bottom. The fix is temporal: prefer the candidate nearest the
*previous* frame's box, not only the frame centre. Note I found this clip by
picking one already known to fail (it is F19 of the 3.2 label set), so 1/40 is
the honest rate, not my anecdote.

---

## 2026-10-09 (3.3b) — the gap fix: ask the pixels, not the gap length

**Goal.** Stop throwing away 7% of the dataset. `MAX_GAP = 2` called any gap
over two frames a camera cut; most of them are not.

**Measured first.** Over 60 clips, for every detection gap of 3+ frames, is
there a frame-difference cut within 2 frames of where it starts?

| | |
|---|---|
| gaps of 3+ frames | 47 |
| next to a camera cut | 10 (21%), median length 4 |
| **no cut nearby** | **37 (79%)**, median length 5 |
| frames binned by gaps with no cut | **230** |
| frames correctly trimmed at a cut | 48 |

Five good frames thrown away for every bad one.

**Built.** `src/preprocess/cuts.py` (`find_cuts`, `shots`) + 11 tests, and
rewired `crop_clip` to split the clip at its cuts, keep the shot holding the
most detections, and interpolate gaps inside it **whatever their length**.
A `MIN_DETECTED = 0.4` guard refuses a shot that is mostly invented.

**Broke.** My own test caught the cut detector's weak spot. On a synthetic clip
with one cut and no other motion, the threshold is `3 x median of the moving
differences` — and with only one moving difference, the cut *is* the median,
so it sets a bar it cannot clear. Fixed by dropping the top decile before
taking the median. On 120 real clips the new and old formulas give identical
cut lists for 117 (98%), and Day 2's findings still hold exactly: first cut at
median 88% through, **none in the first half**. Day 2's numbers stand.

Also broke: the synthetic test clips were random noise, which the new code
reads as a cut at every frame. Replaced with visually continuous frames and a
`cut_clip()` helper that joins two different shots.

**Result**, same 60 clips, same seed:

| | before | after |
|---|---|---|
| median frames kept | 20 | **22** |
| 10th percentile | 10 | **15** |
| clips dropped entirely | 4/60 (7%) | **1/60 (2%)** |
| clips with 15+ frames | 75% | **90%** |

About 700 clips saved, and Day 4 can sample 15 frames from 90% of the dataset
instead of 75%.

**Then eyeballed the recovered frames**, because interpolated frames are
invented and could be cropping grass. `Pull/vid436_6.avi` fills 12 of 24 and
the batter is in every frame. But `Flick/vid116_8.avi` exposed a different
bug: the broadcast **zooms in**, and a single clip-wide crop size cannot
follow it — the batter grew until the bat was clipped off the top.

**Fixed that too.** Size the crop from the box **height**, smoothed over 9
frames, not from `max(w, h)` held constant. Height swings x1.3 within a clip
against width's x2.3, because width is the bat leaving his outline and height
is the camera. Sizing on height follows a zoom without pulsing on the swing.
Re-rendered: the bat is now whole through the follow-through, and
`Cover Drive/vid1_0.avi` is unchanged.

**Open.** Subject switching (0.4% of frame pairs) is still unfixed; the
temporal link — prefer the candidate nearest the previous frame's box — is the
next thing. `MIN_DETECTED = 0.4` was chosen, not measured; it currently
refuses 1 clip in 60.

---

## 2026-10-09 (3.3c) — the temporal link, and knowing when to stop

**Goal.** Stop the crop walking off the batter onto a fielder.

**Broke first: my own framing.** I had reported the problem as "0.4% of frame
pairs", which made it sound negligible. That was frame *pairs*; a switch ruins
a run of frames. Measured at frame level over 60 clips: **2.2% of frames** crop
the wrong person, across **22% of clips** — 1 clip in 5, not 1 in 40. Worth
fixing, especially for a temporal model, where a mid-sequence jump to another
person is a false cut in exactly the signal the model reads.

**Built.** `candidates()` in `striker.py` (the runners-up, not just the winner),
and a two-pass `track_boxes` in `crop.py`:

1. pass 1 — centrality, as before, 97% per frame
2. build a reference track from the frames that *agree with each other*
3. pass 2 — re-pick each frame against that track, taking the runner-up where
   the most central box disagrees with the clip as a whole

Two passes rather than chaining off the previous frame, because a chain lets
one bad frame poison every frame after it. There is a test for exactly that.

**Broke: my tests, twice.** The first "fielder" I invented overlapped the
batter by 0.50 IoU — geometrically the same person, so nothing could
distinguish them. And I asserted `"swapped 4"` when the code says `"4 swapped"`.
Both were test bugs, not code bugs; fixed the tests.

**Result, and then knowing when to stop.** After the two passes: 1.5% of frames
wrong, 17% of clips. Better, not dramatic. So rather than keep tuning, I asked
whether the remainder was even fixable — for every still-wrong frame, did the
detector offer *any* box on the batter?

```
frames still cropping the wrong person: 21
  a correct box WAS available, we picked wrong : 0 (0%)
  the detector offered nothing on the batter   : 21 (100%)
```

**All of it.** Selection had nothing left to extract. What was left was the
crop being placed on whoever *was* detected — usually the bowler running in
before the batter is picked up.

**So the last fix was to stop guessing instead of to guess better.** The
reference track used to hold its end value outwards, which let those frames
pass. Now there is no reference outside the agreed stretch, so they are trimmed
rather than cropped onto the wrong man.

| | start | two passes | + no end-hold |
|---|---|---|---|
| frames cropping the wrong person | 2.2% | 1.5% | **0.6%** |
| clips affected | 22% | 17% | **8%** |
| clips with 15+ frames for Day 4 | 90% | 92% | **92%** |

The trim costs a frame or two on some clips (24-frame survivors 38% -> 30%)
but leaves the number Day 4 depends on untouched.

**Open.** The remaining 0.6% needs a better detector, not better selection, and
the authors already fine-tuned that model to mAP50 0.904 — see the next entry.
`TRACK_IOU = 0.2` is shared with the metric used to score this, so the headline
improvement is partly self-marking; the eyeball check on `Flick/vid360_55.avi`
is the independent evidence, and it now starts at the batter rather than the
bowler.

---

## 2026-10-09 (3.3d + 3.4) — the mask variant, and 30 crops judged by eye

**The finding that started it.** The checkpoints record their own training runs.
The authors **fine-tuned all three models** themselves: Player_Type from
`yolo11x-seg.pt`, 500 epochs, their own `SegmentData` on Colab, Dec 2024,
scoring precision 0.947 / recall 0.870 / **mAP50 0.904**. So the detector is
already a domain fine-tune, which is the answer to "should I fine-tune it" --
not without labels they did not ship, and not before proving it is the
bottleneck.

But the `task` field says **segment**, and their classifier is named
`Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras`. We were
throwing away masks we already pay for.

**Built.** `detect(masks=True)` returns silhouettes resized to the frame --
ultralytics pads them to a multiple of 32, so a 540-row frame comes back with
544-row masks and they need resizing. `crop_clip(segmented=True)` blacks out
everything that is not the batter **or his bat**: keeping only the person mask
would discard the one object that tells a Sweep from a Pull. Bats more than one
body-height away belong to somebody else. 6 new tests.

**Open, and the one decision left.** An interpolated frame has no mask, so it
keeps its background. Over 40 clips: **9.6% of frames, in 57% of clips**.
Inside a clip that is otherwise silhouettes, that is a flicker a temporal model
will see. Options: fall back to the interpolated box as a rectangle (background
gone, no invented silhouette), carry the nearest real mask (keeps the look,
may clip the pose), or drop the frame (breaks the frame-for-frame comparison
with the box variant). Not fixing it unmeasured.

## 3.4 — the eyeball pass

Two clips from each of the 15 classes, three moments each (start, middle, end
of the kept span), both variants. 90 frames.

**29 of 30 clips are right.** The batter is centred, the bat is in frame, and
the shot is readable in every one. The exception is `Defensive/vid239_7.avi`,
whose last frame zooms in far enough to show a fielder's legs above the
batter's head.

The pattern worth noting: **end frames are consistently tighter than start
frames**, because the crop size follows the box height and broadcasts zoom in
during a shot. That is the height-based sizing doing its job, and it
occasionally overshoots.

Segmented versions of the same 30: clean silhouettes, bat kept, background
gone -- apart from the unsegmented frames above, which stand out immediately
when you look at a sheet of them. Looking at the output is how that was found;
no number reported it.

---

## 2026-10-09 (3.3e + 4.1) — carry the mask, and any length down to 15

**The no-mask decision, settled by measurement.** An interpolated frame has no
silhouette (9.6% of frames, 57% of clips). Three options were on the table, so
I tested them on 463 frames that *do* have a mask, by pretending they do not:

| stand-in | IoU with the real silhouette |
|---|---|
| bounding box filled in as a rectangle | 0.51 |
| **carry the mask from 1 frame away** | **0.80** |
| carry from 2 frames away | 0.71 |
| carry from 3 frames away | 0.65 |

Carrying wins outright, and still beats the rectangle three frames out. So
`carried_mask()` takes the nearest real silhouette from whichever side is
nearer and shifts it onto the interpolated box. `_shift` rather than
`np.roll`, so nothing wraps round the frame edge.

**4.1 `sample.py`.** Drop duplicates, then sample 15 evenly.

The duplicate threshold is not a guess. Nothing in this dataset is
bit-identical -- the clips were re-encoded -- but over 150 clips there is an
**empty gap**: duplicate pairs top out at a mean absolute difference of
**0.927** and real motion starts at **1.045**. A threshold of 1.0 sits in the
gap with nothing near it. Duplicates by length: 24-frame 0.4%, 28-frame 15.3%,
49-frame 0.0% -- which reproduces Day 2's finding from a different direction.

Comparing each frame to the **last kept** frame, not the previous frame: a run
of sub-threshold steps would otherwise delete a slow pan entirely. There is a
test for that.

**End to end on real clips** (read -> dedupe -> detect -> crop -> sample):

```
clip            raw  dedup  kept  out
vid306_8.avi     24     24    16   15   all indices < 16, the measured cut
vid1_0.avi       29     26    17   15   3 duplicates removed
vid368_14.avi    28     23    23   15   5 duplicates removed (18%)
vid317_9.avi     49     49    49   15   spread 0..48, not truncated
vid385_27.avi    49     49    32   15   trimmed 17 at a cut
```

Exactly 15 out every time, first and last always included, never backwards.
The `vid306_8` check the plan asked for passes: no sampled index lands past
the camera cut.

**Open.** Storage format for Day 4.3. Measured on real crops: raw 42.4 GB for
both variants, PNG 11.3 GB lossless, JPEG q95 3.0 GB, WebP q90 1.4 GB. Staying
lossless keeps the decision reversible -- the 4-hour run is the expensive part
and raw <-> PNG transcodes for free, while JPEG is one-way.

---

## 2026-10-09 (4.3) — the driver, built around the power cuts

**Goal.** Walk 10,091 clips, write both variants, and survive an interruption.

**Built.** `src/preprocess/build_dataset.py` (143 lines, 9 tests). Per clip:
read -> drop duplicates -> detect -> crop box -> crop segmented -> sample 15 ->
save. Each clip is finished and recorded before the next starts, so a cut costs
**one clip**. Re-running the same command skips whatever the manifest lists.

Most of the tests are about interruption rather than about cropping:

- a second run redoes nothing and does not touch the files already written
- a run stopped after one clip resumes and finishes the other two
- a manifest line cut in half mid-write is skipped, and that clip is simply
  redone -- `read_results` was written for this on Day 1 and this is the first
  time it has been needed
- one unreadable clip records its reason and the run carries on

`_save` writes `.npy.tmp` then `os.replace`, so a cut can never leave a
half-written array that loads as a valid file with garbage in it.

**Measured on real clips.** 12 then 20, the second run picking up from the
first:

```
20 clips, 12 already done, 8 to go
box (15, 224, 224, 3) uint8   nonzero 100%
seg (15, 224, 224, 3) uint8   nonzero  14%
```

**0.5 clips/s, so about 5.6 hours** for the full dataset. That is slower than
Day 3's 4 h estimate because masks and a second crop pass were added since;
the 30% is the price of the segmented variant and it is paid once.

Disk: 86.1 MB for 20 clips over both variants, projecting to **42.4 GB** --
within 1% of the estimate made from 25 sample crops, which is reassuring about
the estimate rather than about the number.

**Broke.** Two small things. `except (ClipError, Exception)` -- redundant,
since `ClipError` is an `Exception`; the bare catch is deliberate and now says
so. And running the module with `python -I -m` failed: isolated mode strips the
working directory from `sys.path`. `-I` is the habit for scripts that read
untrusted data, not for the project's own modules.

**Open.** Nothing blocking. The run is ready to launch.
