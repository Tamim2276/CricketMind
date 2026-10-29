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

---

## 2026-10-10 (4.4) — the full run: 10,091 clips in 4.53 hours

**Done.** 10,091 clips, **9,880 usable (97.9%)**, 211 failed, 4.53 h at 0.6
clips/s. No `read_clip` retries at all, so no flaky decodes over the whole
dataset.

Output is consistent with the manifest, which is the check that matters:

```
box: 9880 .npy files, 0 leftover .tmp   21 GB
seg: 9880 .npy files, 0 leftover .tmp   21 GB
```

9,880 files in each variant against 9,880 `ok` rows, and **no `.tmp` files
left behind** -- the atomic save never got caught mid-write. 42 GB total,
within 1% of the 42.4 GB projected from 25 sample crops. Spot-checked 8 random
clips: all `(15, 224, 224, 3)` uint8, segmented versions 12-17% non-black.

**The failure rate is uneven, and I said to investigate that.** So:

| class | clips | failed | rate |
|---|---|---|---|
| Scoop | 279 | 24 | **8.6%** |
| Reverse Sweep | 252 | 13 | 5.2% |
| Straight Drive | 423 | 21 | 5.0% |
| ... | | | |
| Pull | 745 | 6 | 0.8% |
| Hook | 534 | 0 | **0.0%** |

**Cause found, and it is the footage, not a bug.** Scoop's failures are
overwhelmingly "too short" (20 of 24, against 4 "too little detected"), and
surviving Scoop clips keep a median of **19 frames against 22-23** for other
classes. A scoop sends the ball over the keeper, so the broadcast cuts away
sooner, so more of the clip is correctly trimmed, so more clips fall under
`MIN_FRAMES = 8`. The trim is right; the shots are just shorter on screen.

**Does it distort the dataset?** Barely. The two worst-hit classes were already
the smallest, so the worry was that preprocessing would amplify the imbalance:

```
imbalance before : 1123:252 = 4.46:1
imbalance after  : 1108:239 = 4.64:1
```

4.46 to 4.64. Not nothing, but not a distortion worth re-running 4.5 hours
for. Worth one sentence in the thesis: preprocessing costs the rarest classes
slightly more because their shots leave the screen sooner.

**Failure reasons overall:** 115 too short, 93 too little detected, 3 no
striker. All three are guards refusing to invent data rather than bugs.

**Next, and it is a real dependency.** The split CSVs still name all 10,091
clips, 211 of which have no output. Day 5's loader must filter the splits
against the manifest or it will fail on a missing file.

---

## 2026-10-10 (5.1) — the dataset, and the 211 clips that are not there

**Built.** `src/data/dataset.py` (104 lines, 14 tests). Reads a Day 1 split
CSV, loads the `.npy` build_dataset.py wrote, hands back a tensor and a label.

The shape contract, written into the docstring because everything downstream
assumes it:

```
on disk      (T, 224, 224, 3) uint8
__getitem__  (T, 3, 224, 224) float32, normalised
DataLoader   (B, T, 3, 224, 224)
```

ImageNet mean/std rather than plain 0-1, because the pretrained encoder was
fitted on inputs scaled that way and any other scaling shifts every feature it
has ever seen. Verified by eye on real data: the output runs -2.12 to 2.39
with mean +0.121, which is what ImageNet normalisation of ordinary photographs
looks like. Plain 0-1 would have come out 0 to 1.

**The dependency from 4.4 is handled.** The split CSVs still name all 10,091
clips. Missing ones are dropped at construction and listed in `.missing`, with
`strict=True` to raise instead. The arithmetic checks out exactly:

```
grouped  train 121 + val 43 + test  47 = 211 dropped
author   train 146 + val 30 + test  35 = 211 dropped
```

211 both times, from two independently built splits -- which is a real check,
not a coincidence, since the two CSVs partition the same clips differently.

**Both variants hold the same clips**: box and seg come out at 6341/1569/1970
each, so any box-vs-seg comparison is on identical data and identical labels.

**Broke.** A typo of mine in the tests -- `transform=torch.flip_ := None`, a
walrus operator used on an attribute, which is a syntax error. Caught at
collection, fixed in the test, nothing to do with the module.

**Check from the plan passes:** `6341 torch.Size([15, 3, 224, 224])
torch.float32 0`.

---

## 2026-10-10 (5.2) — one image model, fifteen frames

**Built.** `src/models/encoder.py` (76 lines, 15 tests). The plan's check
passes on real data on the Arc:

```
batch of clips      (2, 15, 3, 224, 224)
folded into photos  (30, 3, 224, 224)
CNN describes each  (30, 1280)
unfolded to clips   (2, 15, 1280)
```

20.2 M trainable parameters, features mean +0.021 std 0.310. The ImageNet
weights were already cached, so no download -- worth knowing, because a
download mid-loadshedding is a bad time to discover one is needed.

**Two decisions that are not just the reshape.**

*The backbone is a parameter.* Phase 3 swaps it ten times, so
`FrameEncoder("resnet18")` has to work too. Different torchvision models put
the ImageNet head on different attributes -- `.classifier` for efficientnet,
`.fc` for resnet -- so `_strip_classifier` tries each, finds the Linear inside
it, records `in_features` and replaces it with Identity. Everything downstream
reads `encoder.out_dim` rather than hardcoding 1280. There is a test that
builds a resnet18 and checks it comes out 512 wide.

*Freezing the weights is not enough to freeze a backbone.* BatchNorm updates
its running mean and variance in train mode whether or not it has gradients,
so a "frozen" encoder's features would drift anyway. `train()` is overridden
to keep a frozen backbone in eval. This matters for Phase 3's frozen-feature
experiments, where the whole point is that the features do not move.

**The tests worth having** are the ones that would catch a wrong reshape,
since that is the only real risk here and a wrong one still produces tensors
of the right shape:

- the same frame repeated four times gives four identical vectors
- a frame encoded inside a clip matches that frame encoded alone
- two clips in a batch do not leak into each other

A transpose instead of a flatten would pass a shape assertion and fail all
three.

---

## 2026-10-10 (5.3) — the GRU head

**Built.** `src/models/head.py` (68 lines, 17 tests). The authors' stack:
GRU(128) -> BatchNorm -> Dense(1024, ReLU) -> Dense(15), **no dropout**,
because they have none and this is the reproduction before it is an
improvement. Dropout is a parameter defaulting to 0, so Phase 3 can turn it on
without a rewrite.

Check passes on the Arc, encoder and head wired together:

```
encoder   (2, 15, 3, 224, 224) -> (2, 15, 1280)
head      (2, 15, 1280)        -> (2, 15)
```

0.69 M parameters in the head against 20.2 M in the encoder -- almost all the
capacity is in the frame encoder, which is worth remembering when Phase 3 asks
where the accuracy is coming from.

**The output is logits and that is deliberate.** Summed over classes the first
clip gives -0.02, not 1.00, and some entries are negative. `CrossEntropyLoss`
does softmax and log together in one numerically stable step; doing them
separately loses precision exactly where the model is most confident. A test
asserts the output is *not* a probability distribution, which is the kind of
thing that is easy to "fix" wrongly later.

**Tests worth having.** Two attack the thing that would make this not a
temporal model at all:

- reversing the clip changes the prediction
- changing only the **first** frame changes the prediction, so the summary
  really carries the whole sequence rather than collapsing to the final pose

Both would pass trivially if the head were broken in a way that still returned
the right shape -- for instance taking `x[:, -1]` and ignoring the GRU.

A third pins a real constraint: `BatchNorm1d` raises on a batch of one in
training mode. Better to fail loudly in a test than to discover it at the end
of an epoch when the last batch happens to have one clip left over. Day 6 must
either drop the last batch or keep batch size above 1.

**Also noted:** an untrained head scores cross-entropy 2.71 on random labels,
which is `ln(15)`. That is 5.4's check, and it already holds.

---

## 2026-10-10 (5.4) — one batch end to end, and the memory wall

**Check passes.**

```
clip batch      (6, 15, 3, 224, 224)
logits          (6, 15)
loss            2.6406
ln(15)          2.7081   difference 0.0674
first conv      (24, 3, 3, 3)   gradient norm 2.816625
tensors with no gradient: 0 of 460
```

Loss sits where an untrained 15-class model should, gradients reach the very
first convolution, and **no parameter is left unconnected** -- 0 of 460, which
is a stronger check than the plan asked for and would catch a head wired to
the wrong tensor.

**Built, slightly beyond the plan.** The plan says "build nothing", but
`src/models/model.py` (36 lines, 8 tests) makes this a one-liner and Day 6
needs it anyway. `ShotModel` just holds the two halves; `build_model()` sizes
the head from `encoder.out_dim`, so swapping to resnet18 resizes it to 512
with nothing edited by hand.

**Broke: out of memory, at the first attempt.** Batch 8 in fp32 died. The
reason is the reshape from 5.2: the encoder sees **B x T images at once**, so
8 clips is 120 images through EfficientNetV2-S with activations held for the
backward pass.

| batch | images | fp32 peak | bf16 peak | bf16 fwd+bwd |
|---|---|---|---|---|
| 2 | 30 | 4.44 GB | 2.43 GB | 0.160 s |
| 3 | 45 | 6.73 GB | 3.59 GB | 0.204 s |
| 4 | 60 | 8.93 GB | 4.67 GB | 0.218 s |
| 6 | 90 | 13.19 GB | **6.82 GB** | 0.353 s |
| 8 | 120 | **OOM** | not measured | |

bf16 autocast roughly halves it, which is what `get_amp_settings` was written
for on Day 1 and this is the first time it has mattered.

**Worth knowing: an XPU out-of-memory poisons the context.** After the batch-8
failure every later allocation in the same process died with
`UR_RESULT_ERROR_OUT_OF_RESOURCES`, including ones that had just succeeded.
The sweep had to be split across two processes. **Day 6 must treat OOM as
fatal** -- catch it, checkpoint, exit, and let the resume logic restart -- not
try to carry on with a smaller batch.

**Arithmetic Phase 3 needs to see now.** At batch 6 and 0.353 s a step, 6,341
train clips is 1,057 steps, about **6 minutes an epoch**. Thirty epochs is 3
hours a run, and 88 runs is roughly **11 days of GPU**. Fine-tuning everything
every time is not affordable; the frozen-encoder experiments are not a variant
to try but the only way most of that table gets filled.

**Also confirmed:** `BatchNorm1d` raised at batch 1 during the sweep, exactly
as the 5.3 test predicted. Day 6 drops the last batch or keeps batch >= 2.

**Follow-up, same day.** Suggested that the batch-8 OOM might have been other
software holding memory -- a good hypothesis, since exactly that caused three
failures on Day 2. Tested rather than argued: retried batch 8 fp32 on an idle
machine, **11.08 GB VRAM and 9.62 GB RAM free, Chrome closed**. Still out of
memory. The scaling says why -- 4.44, 6.73, 8.93, 13.19 GB for batches 2, 3,
4, 6, so batch 8 needs about 17.6 GB against a 11.6 GB card.

But the mechanism in the hypothesis is real and worth recording: **batch 6
fp32 reported 13.19 GB allocated on an 11.6 GB card**, so roughly 1.6 GB came
from shared system memory. Any configuration that spills does depend on free
RAM, and there other software matters. bf16 at batch 6 peaks at 6.82 GB and
stays inside the card entirely -- a second reason to use autocast beyond the
saving itself.

---

## 2026-10-10 (6.1) — train_epoch, and an overfit check that proved nothing

**Built.** `src/engine.py` (118 lines, 17 tests). One epoch: zero, forward,
loss, backward, step. Returns mean loss, top-1, clips and optimizer steps.

**Broke: my own check.** First run said "50 clips ... loss 2.62 -> 0.05,
**100% accuracy**", which looked like a pass. It was not. The line above it
said **"1 classes present"**. The split CSV is ordered by class, so the first
50 rows are all Cover Drive, and the model had learned to say one word. A
single-class subset makes 100% meaningless.

Re-ran on **4 clips from each of the 15 classes**:

```
  epoch    loss   top-1
      1  2.7234   15.0%
      3  2.1745   75.0%
      6  1.5433  100.0%
     20  0.2248  100.0%
```

That is the real check. Epoch 1 at **2.7234 against ln(15) = 2.7081** is 5.4's
anchor turning up again on its own, and 100% by epoch 6 means the whole chain
-- dataset, encoder, head, loss, optimizer -- can carry a gradient that
actually changes predictions.

**Tests are mostly about the silent failures**, since a broken epoch loop
still returns plausible numbers:

- *gradients are cleared between batches.* Run with lr=0 so the weights cannot
  move, and watch the gradient magnitude at each step. Without `zero_grad` it
  grows every batch; the test fails if the largest is more than 3x the
  smallest.
- *accumulation matches one big batch.* Four batches of 2 with `accum=4`
  produce the same gradient as one batch of 8, to 1e-5. This is what catches
  forgetting to divide the loss by N -- without it the gradient is 4x too
  large and training looks "fast" then diverges.
- *the scheduler steps per optimizer step, not per batch.* 6 batches with
  accum=2 must advance it 3 times.
- *the model is put into train mode*, because BatchNorm quietly uses stored
  statistics otherwise.

**Timing.** 2.5 s an epoch over 10 batches, so ~0.25 s a step at batch 6,
which matches 5.4's 0.353 s including its backward through a cold cache. The
first epoch costs 15 s extra for warm-up and weight loading.

**Flaky test found and fixed, same day.** The full suite failed on
`test_the_last_frame_is_not_the_only_one_that_counts`, which had passed
earlier -- it used an unseeded random clip and asserted the effect of changing
frame 0 exceeded 1e-4. Measured over 40 seeds: median **2.17e-04**, min
5.09e-05, and **1 in 40 falls below 1e-4**. So it failed about 2.5% of runs on
chance alone.

The measurement is worth keeping for its own sake: an **untrained** GRU feels
frame 0 about a thousand times more weakly than frame 14 (2.2e-04 against
2.7e-01). That is expected -- forgetting is the default and training is what
builds the memory -- but it means any tolerance in that test was testing the
seed. Now seeded, and asserting only that the effect is non-zero, which is the
property actually worth pinning.

---

## 2026-10-10 (6.2) — evaluate, and why below chance is the right answer

**Built.** `evaluate` and `topk_correct` in `src/engine.py` (221 lines now,
27 tests). Returns loss and top-1/2/3, optionally collecting predictions and
labels for a confusion matrix later.

**On the real validation set, untrained:**

```
loss    2.7369   ln(15) = 2.7081
top-1    2.42%   chance  6.67%
top-2    6.37%   chance 13.33%
top-3   12.81%   chance 20.00%
```

**Top-1 came out below chance and that needed explaining, not excusing.** An
untrained network is biased, not uniform. This one lands 65.6% of its
predictions on Scoop and 27.6% on Reverse Sweep -- and those are the two
*rarest* classes, 2.6% and 2.4% of the validation set.

So the arithmetic:

```
expected for a model with this bias   2.80%
measured                              2.42%
a uniform guesser                     6.67%
```

2.42% against 2.80% predicted is within one standard error on 1,569 clips
(0.42%). Below chance is exactly right for a model that always says Scoop.

The loss agrees independently: 2.7369 is slightly **above** ln(15), which is
what bias costs -- a uniform predictor scores exactly ln(15) and any
concentration raises it.

**My pass condition was badly written.** I coded `abs(top1 - chance) < 0.04`,
which flagged this as a CHECK. The plan says *materially above* chance means a
leak; below chance rules one out. Symmetric tolerance was the wrong test and
I should have written the one the plan states.

**Broke: my own hand-worked test.** `test_topk_counts_a_hit_at_each_depth` had
logits `[3.0, 5.0, 4.0]` with true label 2, which I called "top 3, not top 1".
It is rank 2. The code returned `[1, 3, 3]` and was right; my expected
`[1, 2, 3]` was wrong. Rewrote the fixture as the same ranking three times
with the true label at rank 1, 2 and 3 in turn, which is unambiguous.

**The useful test here** is a deliberately cheating model that reads the label
off its input and must score exactly 1.0. A metric that silently mis-indexes
still produces plausible numbers on a real model; it cannot produce 1.0 on
that one.

---

## 2026-10-10 (6.3) — the loop, and pulling the plug on purpose

**Built.** `src/train.py` (260 lines, 11 tests). Epoch loop, best-checkpoint
tracking, early stopping, `save_resume` every epoch, one ledger row at the
end.

**Three decisions, all about the power rather than the accuracy.**

`save_resume` runs **before** any `break`, so an early stop cannot lose the
epoch that triggered it. There is a test: the last epoch in `history.json`
must equal `epochs_run` in the ledger row.

`resume.pt` is **deleted on a clean finish**, so its presence means exactly
one thing -- interrupted -- and never "a finished run left a file lying
about". Also tested, from both directions.

**Resuming is the default**; `--fresh` is the flag. When the power comes back
you press up-arrow and enter.

**The rehearsal, done with a hard kill rather than Ctrl-C** -- no cleanup, no
exception handler, no chance to save on the way out, which is what a cut
actually is:

```
killed during epoch 4
files now: ['best.pt', 'resume.pt']
resume.pt says epoch 3, best 10.00%, 3 epochs of history

... the same command again ...

RESUMED from experiments\smoke2\resume.pt -- starting at epoch 4 with best 10.00%
  epoch   4  val 13.33%  *
  epoch   5  val 10.00%
  epoch   6  val 11.67%
  completed; best val top-1 13.33%

epochs in history: [1, 2, 3, 4, 5, 6]
resume.pt after a clean finish: False
```

No gap, no repeat, and the file cleaned itself up. That is Day 1's
`checkpoint.py` finally doing the job it was written for.

**Design change made while testing.** `CricShotDataset`'s processed directory
is now a config key rather than a constant, because the test needed to point
at a temporary dataset and monkeypatching the module was the alternative. A
config key is better than a patch, and it also means a second preprocessing
run can be trained against without editing code.

**Also folded in:** `_subset_across_classes` so `--limit_clips` takes clips
from every class. 6.1's lesson, now in the code rather than only in a note.

**`.gitignore`.** Per-run folders are regenerable and full of `.pt` files;
`experiments/results.jsonl` is the ledger of what was actually run and must
survive. Now `experiments/*/` ignored with the ledger negated.

---

## 2026-10-11 (6.4) — reading their model instead of their notebook

**Neither archive contains the notebook.** `cricshot10k-models.zip` holds five
model files and `drive-download-*.zip` holds the YOLO training data. So "every
value traces to their notebook" could not be done as written.

**Their shipped `.keras` file is better evidence anyway** -- it is what
actually produced the weights. `config.json` inside it gives the architecture
and the compile config:

```
Input (15, 224, 224, 3)
TimeDistributed(EfficientNetV2-S)   trainable=True
TimeDistributed(Flatten)
GRU(128)  dropout=0.0  recurrent_dropout=0.0
BatchNormalization -> Dense(1024, relu) -> Dense(15, softmax)
Adam  beta_1 0.9  beta_2 0.999  epsilon 1e-07  weight_decay null
loss  sparse_categorical_crossentropy
```

So: no dropout confirmed, encoder **not** frozen confirmed, 15 frames at 224
confirmed.

**The learning rate is a fingerprint.** The saved value is **1e-9**. Nobody
starts there. It is where `ReduceLROnPlateau(factor=0.1)` lands after five
reductions from 1e-4: 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9. Both the starting
rate and the factor are corroborated by a number neither was written down in.

**And the finding that made 6.4 real work.** Their GRU kernel has shape
**(62720, 384)**. 384 is 3 gates x 128. **62720 is 7 x 7 x 1280** -- their
EfficientNet stops at `top_activation` with no pooling, and the Flatten hands
the GRU the entire spatial map.

**We were average-pooling to 1280.** That throws away *where* things are, and
for a cricket shot the bat's position relative to the body is much of the
signal.

```
their GRU  24.13 M parameters      their model  44.5 M
our GRU     0.54 M                 our model    20.9 M
ratio        44.6x
```

**Built.** `pool="avg"|"flatten"` on `FrameEncoder`, with `out_dim` now
**measured by one dummy forward pass** rather than read off the classifier --
deriving it means knowing each backbone's stride and whether it pools, and a
forward pass just answers. It reproduces 62720 for EfficientNetV2-S and 25088
for resnet18 (512 x 7 x 7), and 62720 matching their kernel exactly is the
confirmation that the reading was right.

Also built: `src/data/augment.py` with `ClipFlip` (one decision per clip, not
per frame -- flipping frame 7 and not frame 8 invents the camera jump Day 3
spent its time removing), and `ReduceLROnPlateau` in `train.py`, stepped on
**validation top-1** after each epoch rather than inside `train_epoch`.

**Memory, re-measured because the architecture changed:**

| pool | batch | params | peak | step |
|---|---|---|---|---|
| avg | 4 | 20.9 M | 4.67 GB | 0.233 s |
| avg | 6 | 20.9 M | 6.83 GB | 0.378 s |
| flatten | 4 | **44.5 M** | **5.60 GB** | 0.264 s |
| flatten | 6 | 44.5 M | 7.76 GB | 0.410 s |

Their batch of 4 fits. At 0.264 s a step and 1,580 steps, that is about
**7 minutes an epoch**, so 100 epochs is ~12 hours -- early stopping with
patience 10 will almost certainly cut it short.

**Four values could not be sourced from the file** and are marked as such in
the config: batch 4, 100 epochs, early-stop patience 10, plateau patience 4,
and the flip itself. Those come from the paper, which the plan already warns
disagrees with the notebook in places.

---

## 2026-10-11 (7.1 launched, 7.2, 7.3) — evaluation that cannot go stale

**7.1 is running.** `mimic_author`: 6,320 train / 1,584 val, batch 4, pool
flatten, 100 epochs. Measured before launch: 1,580 steps an epoch at 0.264 s,
so ~9 min an epoch and 15.5 h worst case.

**A plan bug found at launch.** The plan said `python src/train.py --config
...`, which fails -- the module imports `src.*` and needs `-m`. Corrected.

## 7.2 — flip, online rather than offline

The plan called for an offline flip that doubles the training set on disk.
Built as a transform on the training `Dataset` instead. The leak the plan
warns about -- a clip's mirror landing in test while the original is in train
-- is impossible **by construction**, because no flipped files exist to be
split.

| | offline | online |
|---|---|---|
| extra disk | +21 GB per variant | none |
| steps an epoch | 3,160 | 1,580 |
| 100 epochs | ~30 h | **~15.5 h** |
| what it sees | the same two copies forever | a fresh coin-flip each epoch |

The doubling is what costs the fifteen hours, and it buys a *fixed* pair
rather than variety.

## 7.3 — `src/evaluate.py`

Top-1/2/3, per-class precision/recall/F1, macro averages, the 15x15 confusion
matrix, and the guard this project actually needs.

**Every report carries the checkpoint's SHA-256, size and mtime**, and
`load_report` re-hashes the file and raises `StaleReport` if it has moved.

*A hash rather than a timestamp*, because an mtime proves very little: a file
can be rewritten with the same one, clocks drift, and copying moves it. There
is a test that touches a file with `os.utime` without changing a byte and
asserts the report still loads -- a timestamp check would false-alarm there.

*It raises rather than warns*, because a warning inside a six-hour log is a
warning nobody reads. The message says what to do.

**Proved end to end on CPU** (the Arc is busy), by training a small model,
scoring it, then retraining underneath the report:

```
report top-1  0.3889
history best  0.3889
difference    0.0000          <- the plan's "val matches history" check

... retrain, then try to read the old report ...

  report : cc12f9ad0d10462b  2026-10-10 12:58:31
  on disk: a0f0d884d5b01f9d  2026-10-10 12:58:33
  Re-run the evaluation; do not read these numbers.
```

That is the 23.67% failure made impossible rather than merely unlikely.

**Metrics are hand-checked, not trusted.** `per_class` is nine lines of
arithmetic off the confusion matrix with a test that works the numbers out by
hand, including the case that matters: a class the model never predicts scores
0.0 rather than NaN, because one NaN silently poisons the macro average.

**Also:** the progress bar moved to `src/utils/progress.py` so training and
preprocessing share one implementation, and `evaluate` gained the `on_batch`
callback `train_epoch` already had.

---

## 2026-10-10 (7.1 result) — 32.39%, and the reason was one line of initialisation

**First, a misreading of my own.** I twice reported the run as "still inside
epoch 1". It had finished at 05:23. My check looked only for `resume.pt`, and
its absence means *either* "epoch 1 not done yet" *or* "completed cleanly" --
I had written that ambiguity into the design deliberately and then fell for
it. The right check is `history.json` existing, or the ledger row.

**The result.** Best val **32.39%** at epoch 4, early stop at 14, 1.56 h. The
target is 89.09% and the plan's epoch-5 gate is 60%; we had 30.93%.

```
  ep   train     val
   4  24.49%  32.39%   <- best
   8  25.49%  27.78%
   9  17.90%  12.44%   <- collapsed, lr unchanged at 1e-4
  14  21.74%  24.24%
```

**Train accuracy peaked at 26.72%.** That is the diagnostic. A 44.5 M model
fine-tuned end to end on 6,320 clips should be able to *memorise* far more
than a quarter of them. This was not a generalisation failure; it was barely
fitting.

**Cause: PyTorch initialises GRU weights as `uniform(+-1/sqrt(hidden))`,
ignoring fan-in.** Keras uses `glorot_uniform`, which divides by
`sqrt(fan_in + fan_out)`. At 1280 inputs the two differ by 1.5x and it does
not matter. At **62720** inputs -- which is what `pool: flatten` gives, the
setting 6.4 added to match the authors -- they differ by **9.1x**:

```
input width 62720
   pytorch   uniform(+-0.0884)   gate pre-activation std 13.06   64.9% saturated
   glorot    uniform(+-0.0098)   gate pre-activation std  1.33    0.0% saturated
```

A saturated sigmoid has gradient near zero, so **two thirds of the GRU's
gates were dead from the first step**.

This also explains why nothing caught it earlier: every smoke test through Day
6 ran on the *pooled* 1280-wide model, where the mismatch is harmless. The
flatten path was introduced in 6.4 and its first real exercise was this run.

**Fix.** `_init_gru_like_keras` in `head.py`: `xavier_uniform_` on the input
weights, orthogonal per gate-block on the recurrent weights, zero biases --
Keras's three defaults. On by default, with `keras_init=False` kept so the
broken behaviour stays reproducible, and a test that asserts it really is
broken.

**A/B on the same 60 clips, same seed, only the init differing:**

```
torch default   8% 10% 18% 28% 22% 22% 33% 30% 35% 42% 50% 67% 68% 75% 78%   loss 2.11
keras glorot   10% 45% 67% 82% 93% 98% 100% 100% ...                          loss 0.56
```

One memorises by epoch 7. The other is still climbing at epoch 15. That is the
32.39% explained.

**Still open, and deliberately not changed at the same time:** the config uses
`variant: box`, but the authors' classifier file is named
`..._NEEDS_CROPPED_SEGMENTED_SHOTS.keras`. Segmented input is the faithful
choice and is also a planned Phase 3 comparison. One variable at a time.

## Day 7 (10 Oct 2026) -- 55.11%, and the second way to saturate a GRU

The init fix worked: epoch 1 went from 15.40% to 51.45% val top-1, top-3 from
39.20% to 83.84%. Then the run peaked at epoch 5 and came apart.

```
epoch  train loss  train top1   val loss  val top1  val top3
  1      2.1163      30.59%      1.4693    51.45%    83.84%
  2      1.6170      46.50%      1.6096    49.12%    81.31%
  3      1.5985      46.27%      2.2296    49.31%    81.82%
  4      1.5776      47.75%      2.3103    53.79%    84.79%
  5      1.5683      47.50%      3.1031    55.11%    83.96%   <- best.pt
  6      1.4278      51.66%      4.8299    48.86%    81.63%
  7      1.5019      48.64%      2.7058    39.14%    71.72%
```

Val loss triples while val top-1 *rises*. Nothing in those columns says why,
so I read the checkpoint instead of guessing.

**Measured in `experiments/mimic_author/best.pt` (epoch 5):**

```
backbone drift from ImageNet, median over 670 tensors       18.65%
worst tensor, features.4.0.block.3.1.bias                  65x its norm
GRU units with |running_mean| > 0.99 (tanh's rail)          20 of 128
GRU units with running_var exactly 0.0                      15 of 128
BatchNorm's eval-time scale on those, 1/sqrt(0 + 1e-5)      316x
units amplified more than 50x                               18 of 128
```

**The chain.** One param group at `lr 1e-4` over 20M pretrained parameters at
batch 4 (`train.py:144`, as it was). The backbone moved a median 18.65% off
ImageNet in five epochs -- healthy fine-tuning moves under 2% -- so its
activations blew up, so the GRU's pre-activations blew up, so `tanh` saturated
and 20 units pinned at +-1 on every clip. A pinned unit has zero variance. In
training BatchNorm divides by the *batch* std, which is also ~0, so it cancels
and the train loss never notices. At eval it divides by `running_var = 0.0`,
i.e. multiplies by 316. Float-level noise becomes a huge activation, through
`fc1`, through `fc2`, into the logits: violently confident and wrong on a
minority of clips. Mean cross-entropy 4.83, argmax barely moved, top-3 still
81%. That is the whole shape of the table above.

The number that mattered most was not in the val column. **Train top-1 stuck
at 47%** -- 74 of 128 units had |mean| > 0.5, so most of the GRU sat in tanh's
flat region. The model could not fit its own training set. Not overfitting;
there was nothing to regularise away.

Same disease as the 32.39% run, different road. `_init_gru_like_keras` fixed
the starting point; it could not fix the dynamics.

**Fixed.**

- `backbone_lr` in `train.py` builds a second param group for the pretrained
  half (`_param_groups`). `null` keeps the single group, so every earlier
  result still reproduces exactly. `ReduceLROnPlateau` cuts both groups --
  tested, because rescuing only the head would have been a silent bug.
- `load_resume` now refuses a resume whose optimizer group count changed, and
  names `backbone_lr` as the cause. Torch's own error does not.
- `GRUHead.pinned_fraction()` / `dead_fraction()` read the saturation straight
  off the BatchNorm running stats. Free -- the stats are already there -- and
  printed every epoch as `pin`, so this shows up at epoch 2 instead of 7.
- The gradient norm is **always** measured and printed as `gn`; clipping stays
  off unless asked for. Measuring is what tells you a threshold is worth
  setting, so it must not be conditional on having set one. It goes through
  `_grad_norm`, not `clip_grad_norm_(inf)`, because that one multiplies every
  gradient by its coefficient on the way out and a single non-finite gradient
  makes the coefficient NaN.

**One thing I nearly got wrong.** I had `clip_grad: 1.0` written into the new
config before testing it. On the toy model the first epoch reported
`gn 422, clip 100%` -- every step clipped. A limit that fires on every step is
not a safety net, it is the optimizer, and it is a deviation the authors' compile
config (no clipnorm) does not have. So `clip_grad: 0.0` ships, `gn` gets
printed, and the threshold comes from that number when there is one. The same
mistake as `MIN_DETECTED = 0.4`: chosen, not measured.

New epoch line:

```
  epoch   1  train loss 1.1643 top1 22.22%   val loss 1.0968 top1 33.33% top3 100.00%   pin 0% gn 422  lr 1e-04/1e-05   0.0m  *
```

`configs/mimic_author.yaml` is unchanged -- it stays the faithful record of
what their `.keras` file specifies. The two deviations live in
`configs/author_backbone_lr.yaml` (head 1e-4, backbone 1e-5) and
`configs/author_frozen.yaml` (the isolation test: can the head alone, on
untouched ImageNet features, beat 55.11%?). Separate `name:`, so separate
`experiments/` directories, so epoch 5's `best.pt` is not clobbered.

405 tests pass, 19 of them new.

### Day 7, later -- the frozen run refuted me, and found the real bug

I predicted: freeze the backbone and `pin` stays at 0%, because frozen features
cannot blow up. **It was 41% at epoch 1.** The prediction was wrong, which is
the most useful thing the run produced.

```
epoch  train top1  val top1  val loss  pin   gn
  1      23.07%     30.37%    2.3739   41%  10.9
  4      33.94%     39.65%    3.1871   48%  14.6
  8      40.66%     42.61%    2.5205   40%  15.9
 11      44.30%     43.56%    3.2223   44%  16.6     <- best, then interrupted
```

Two things that kills:

1. **Backbone drift did not cause the saturation.** With the backbone frozen
   and its BatchNorm in eval, 41% of the GRU is pinned by the end of epoch 1.
   The cause is upstream of training entirely: 62720 non-centred post-SiLU
   features summed into a tanh GRU.
2. **Freezing is worse, not better.** 43.56% at epoch 11 against full
   fine-tuning's 55.11% at epoch 5. The backbone fine-tuning was helping. My
   "lr 1e-4 on the backbone is the bug" was wrong about causation; the 18.65%
   drift is real but is not what capped the run.

So I went back to their `.keras` file and read the BatchNormalization layer,
which I had skimmed the first time:

```
epsilon  = 0.001      torch defaults to 1e-5      100x smaller
momentum = 0.99       torch defaults to 0.1       10x more reactive
```

Keras keeps `momentum` of the old running value; torch takes `momentum` of the
new one. Their 0.99 is **0.01** here. And epsilon is the exact denominator from
the morning's measurement: a pinned unit has zero variance, so its eval-time
scale is `1/sqrt(0 + eps)`. Theirs caps at 31.6x. Ours was **316x** -- which is
the number I measured in `best.pt` and attributed to the backbone.

**The test, on the same weights, nothing retrained:**

```
eps 1e-05   loss 3.1031   top1 55.11%  top2 74.81%  top3 83.96%
eps 0.001   loss 1.4819   top1 56.50%  top2 76.39%  top3 86.43%
```

1e-5 reproduces the recorded 55.11% exactly, so the eval path is sound. One
constant halves the loss and adds 1.39 points of top-1 and 2.47 of top-3. The
"exploding val loss" was mostly a PyTorch default.

Same class of bug as the GRU init, and I should have found it at the same time:
both are Keras defaults that torch does not share. The lesson is that I read
their config for *architecture* and skipped the numeric defaults inside each
layer.

Also checked, and already correct: `reset_after=true` is what torch's GRU does
natively -- it has a separate `b_hn` for exactly that -- plus GlorotUniform
kernel, Orthogonal recurrent, zero bias. Tests now assert our BatchNorm against
their file directly, by reading the zip, so this cannot drift again.

**Fixed.**

- `BN_EPS = 1e-3`, `BN_MOMENTUM = 0.01` in `head.py`, as defaults, because they
  are *their* values -- this makes `mimic_author` more faithful, not less.
- `history.json` is now written every epoch instead of only at the end. The
  frozen run's twelve epochs existed nowhere but the terminal when it was
  killed at epoch 13. A power cut would have cost the same.
- `experiments/mimic_author` renamed to `mimic_author_bn1e-5` so the epoch-5
  checkpoint behind the table above survives the next run.

**Still open:** 41% of the GRU is pinned regardless, and correct BatchNorm
limits the damage rather than preventing it. If the re-run stalls again, the
62720-wide flatten into a tanh GRU is the next suspect -- but it is also
provably what the authors shipped, so that is an improvement, not a
reproduction. One variable at a time.

410 tests pass.

### Day 7, evening -- the BatchNorm fix in full, and a split theory that died

With the authors' BatchNorm constants, the same config that stalled at 47%
train top-1 ran as follows:

```
ep  train loss  trtop1   val loss  valtop1  valtop3   pin   lr
 1    2.0071   33.97%     1.3106   56.00%   87.63%    0%  1e-04
 4    1.2535   58.18%     1.0359   64.52%   90.91%    0%  1e-04
11    1.0949   64.24%     0.9355   68.69%   93.12%    1%  1e-04
15    1.0494   64.94%     0.9854   68.31%   91.29%    0%  1e-04
16    1.0701   64.79%     1.0104   66.29%   90.78%    2%  1e-05  <- plateau cut
19    0.8105   72.69%     0.7892   74.05%   94.26%    1%  1e-05
22    0.6990   76.69%     0.7578   75.06%   95.20%    1%  1e-05
24    0.6628   78.47%     0.7424   75.19%   95.45%    1%  1e-05
```

`pin` stayed at 0-2% for 24 epochs against 40-54% before the fix, which
retires the "62720-wide flatten saturates the GRU" theory from this morning.
The epsilon was causing the saturation, not merely amplifying it at eval:
BatchNorm's *backward* pass scales the gradient into the GRU by
`1/sqrt(batch_var + eps)`, and at batch 4 a unit's variance over four samples
is often near zero by chance. With eps 1e-5 that is a gradient spike of up to
316x into the GRU, which lifts its weights, which saturates it, which lowers
the variance further. eps 1e-3 floors the spike at 31.6x and the loop never
starts. Three symptoms -- the frozen run's 41% pinning, the exploding val loss,
the train-loss stall -- one constant.

The first lr cut was worth 6.5 points on its own (68.69% -> 75.19% over epochs
17-24), which is the thing to remember before declaring any future run capped.

**Train crossed above val at epoch 20.** Not a regime change; the train metric
is an epoch *average* while val is an end-of-epoch snapshot, so a fast-moving
model is scored against its own worse earlier self. At lr 1e-5 it barely moves
within an epoch and the handicap disappears. Flip-on-train-only and
BatchNorm's batch statistics account for the rest.

**A theory I had and discarded.** I suspected `splits_author` inflates 89.09%
by putting near-duplicate clips on both sides. Measured:

```
val clips whose source video also appears in train   1612 of 1614  (99.9%)
val clips with the same source video AND same class      0 of 1614  ( 0.0%)
```

So their split groups by **(video, class)**, not by clip. `vid447` sends its
Cover Drives to val and its Pulls to train. Shared stadium and broadcast style,
but the label is never given away, because the same video's training clips
belong to a different class. Not the leak that inflates a score. `splits_grouped`
(grouped by video alone, 0 shared ids) is still the stricter protocol and still
the honest number, but the gap between the two will be smaller than I assumed,
and the authors' 89.09% does not get explained away this cheaply.

**Remaining known deviations, ranked:** training length (epoch 24 of 100, one
of the five lr cuts their 1e-9 implies, train top-1 still climbing); then
`variant: seg`, which their filename names outright; then `flip: 0.5`, which is
my guess from the paper rather than anything in their config, and which is
train-only and so suppresses exactly the number that is currently limiting.

## Day 7 result (7.3) -- 78.14% test top-1 against the authors' 89.09%

Early stop at epoch 45, best at epoch 35, 199.2 min on the Arc.

`best.pt` scored **79.23%** on `splits_author/val.csv`, which reproduces
`history.json` to the digit -- the eval path is sound and the number is not a
stale report. Then, on the split that compares to the paper:

```
                    val (1584 clips)   test (1976 clips)
top-1                   79.23%             78.14%
top-2                   92.30%             91.30%
top-3                   95.96%             96.05%
macro-F1                0.7785             0.7605
loss                    0.6931             0.7152
```

Authors: **89.09%**. Short by **10.95 points**.

**The per-class shape matches theirs even though the level does not.** Late Cut
is the worst class on both splits -- F1 0.562 val, 0.586 test -- and the paper
also reports Late Cut worst. The confusions are physically sensible rather than
random:

```
Lofted Offside  -> Lofted Legside   28    the same shot, other side of the wicket
Pull            -> Hook             26    genuinely similar; differ in ball height
Upper Cut       -> Square Cut       15
Late Cut        -> Square Cut       12
```

That is a model that has learned cricket and is losing on the hard pairs, not
one with a wiring bug. It is the first run of this project whose errors are
worth reading.

65 clips had no preprocessed output across the two splits (30 val, 35 test), out
of the 211 that failed to build.

**Not done, and deliberately:** this checkpoint must never be scored on
`splits_grouped/test.csv`. The two protocols partition the same 10,091 clips, so
most of grouped's test set sits inside author's train set. The grouped number
needs its own training run, not a second evaluation of this one.

**Next:** `configs/mimic_author_seg.yaml`, which differs from `mimic_author` in
exactly two keys -- `variant` and `name` -- with a test asserting that, because
"one variable at a time" is worth more as an assertion than as an intention.
Their classifier is named `..._NEEDS_CROPPED_SEGMENTED_SHOTS.keras`, so this is
the deviation with their own words behind it.

411 tests pass.

## Day 8 (11 Oct 2026) -- seg loses by 13 points, and what their filename meant

Clean finish: 58 epochs, early stop, 391.0 min. Load shedding arrived after
11:27, which is when the ledger row was written, so nothing was lost.

**Checking whether a run finished, since I got this wrong once before.** The
row in `experiments/results.jsonl` is the only *positive* proof -- it is
appended last, after `history.json` and before `resume.pt` is removed.
`resume.pt` present means interrupted. `resume.pt` absent is **ambiguous**:
either a clean finish or a run that never completed epoch 1. Use the ledger.

```
                     box (mimic_author)   seg (mimic_author_seg)
val  top-1                 79.23%               62.37%
test top-1                 78.14%               65.08%
test top-3                 96.05%               91.60%
test macro-F1              0.7605               0.6303
best TRAIN top-1           86.88%               64.32%
epochs to early stop          45                   58
```

**box wins by 13.06 points on test.** Both checkpoints reproduced their
`history.json` value exactly on val before I looked at test.

The number that explains it is `best TRAIN top-1`: **64.32% against 86.88%**.
The segmented model could not fit its own training set either, 22 points below
what the same architecture manages on box crops. That is not an optimisation
failure -- the same lr schedule, the same init, the same everything -- it is
less information reaching the model. Masking out the pitch, the stumps, the
keeper and the fielders removes the context that tells a Lofted Offside from a
Lofted Legside. The shot is defined by where the ball goes, and the background
is what says where that is.

**So I had their filename wrong.**
`Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras` -- I read
"segmented" as pixel masking, and deferred it for days as the most likely
explanation of the 10.95-point gap. The likelier reading is **segmented in
time**: clips cut at camera boundaries so each one holds exactly one shot,
which is what `cuts.py` already does. On that reading `box` was the faithful
variant from the start, and this experiment did not test their deviation -- it
tested mine.

Worth the 6.5 hours anyway: it is now measured rather than assumed, it closes
the variable the plan had open, and the per-class table shows *why* background
matters, which the box run alone could not show.

**Still 10.95 points short of 89.09%, with seg eliminated as the explanation.**
Next: `configs/mimic_author_noflip.yaml`, differing from `mimic_author` in
`flip` and `name` only. `flip: 0.5` is my guess from the paper, is absent from
their config, applies to train only, and train-set fit is exactly what is
capping both runs. If it is costing points, this finds out in one run.

### Figures, from now on automatically

`src/plot.py` draws one figure per run from `history.json`, and `train.py`
calls it on a clean finish, so nobody has to remember. Both completed runs have
been drawn retroactively.

Four stacked panels sharing the epoch axis -- loss, accuracy, learning rate,
gradient norm. **Not a twin y-axis**, which was tempting for lr-over-accuracy
and would have been a lie about which curve belongs to which ruler. Four
scales, four panels.

The lr cuts get a vertical hairline across every panel. Every sharp bend in
this project's curves has turned out to be a plateau cut -- epoch 16 of the box
run was worth 6.5 points -- and unmarked, the bend reads as the model
spontaneously deciding to learn.

`pin` is in the subtitle, not a panel: on a healthy run it is a flat line at
zero and the only interesting thing is its peak. A fifth panel showing nothing
is worse than a number.

Colours are categorical slots 1/2/3 from the data-viz reference palette,
validated rather than eyeballed -- worst adjacent CVD deltaE 9.2 light / 9.4 dark
against a >=8 target. Aqua comes in at 2.74:1 on the light surface, under the
3:1 bar, so the top-3 series carries a visible endpoint label instead of
relying on its hue. Dark mode is its own stepping of the same three hues, not a
flipped copy, and a test asserts the two tuples differ.

Two bugs the render-and-look pass caught that no test would have: the title
overlapping its subtitle, and the "best 62.37% (epoch 48)" label running off
the right edge when the best epoch lands late. The label now flips inward past
two-thirds of the x-range.

A drawing failure cannot kill a run -- `plot_run` is wrapped, and a test
monkeypatches it to raise and asserts the run still finishes, writes its
history and clears `resume.pt`. Six hours of training will not be lost to
matplotlib.

424 tests pass.

### Tracking the evidence, not just the figures

I said `notes/figures/` was tracked. It is not -- `.gitignore:10` excludes it,
with a fair reason (frame grids, 7.7 MB, regenerable from `src/data/viz.py`).
Checking properly turned up a bigger hole than the figures:

```
experiments/*/history.json     IGNORED   the only per-epoch record
experiments/*/report_*.json    IGNORED   per-class scores + checkpoint sha256
experiments/*/best.pt          ignored   178 MB, correctly
```

80 KB of irreplaceable evidence was being thrown away while the checkpoints
were correctly excluded. `history.json` cannot be regenerated at all once the
run is gone, and `report_*.json` carries the sha256 that ties its numbers to a
specific checkpoint -- the whole point of `src/evaluate.py`.

`.gitignore` now keeps `history.json`, `report_*.json`, and
`notes/figures/runs/`, and still excludes every `.pt` and the frame grids.
1.7 MB tracked in total.

**A defect I introduced and then found.** Pointing the default figure output at
a tracked directory meant every test that calls `train()` dropped a PNG into
the repo -- `a.png`, `b.png`, `myrun.png`, `t.png` all turned up in
`notes/figures/runs/` the first time I previewed `git add`. Fixed with
`tests/conftest.py`: an autouse fixture that repoints `DEFAULTS["figures"]` at
a tmp directory for every test, plus explicit `mirror=` on the tests that call
`plot_run` directly, since a default argument is bound at def time and the
fixture cannot reach it. The whole suite now runs and leaves the directory with
exactly the four real figures in it.

**One inconsistency left alone deliberately.** `experiments/mimic_author_bn1e-5/`
holds `best.pt` from the epoch-5 run (55.11%) but a `history.json` from the
*earlier* 14-epoch 32.39% run, because that run was killed before history was
written per-epoch. Two different runs in one directory. Nothing reads it, but
it is precisely the stale-evidence trap `file_fingerprint` exists to catch, so
it is written down rather than quietly tidied.

428 tests pass.

### The mask measurement, which refuted me

I claimed from six pictures that the masks "cut off at the pad line" and
recommended fine-tuning YOLO *before* the 4.5-hour rebuild. `tools/mask_quality.py`
measured it on **2,662 frames from 120 clips sampled across all 15 classes**:

```
mask coverage of the striker's box     p10 40.7%   median 52.3%   p90 63.9%
how far down the box it reaches        p10 0.90    median 1.00    p90 1.00
  stops above 90% of the box           264 of 2662 frames (9.9%)
bat masked within reach of the striker 1735 of 2662 (65.2%)
no striker picked at all               475 of 3137 frames (15.1%)
```

**The masks reach the feet.** Median 1.00, and only 9.9% stop short. The pad
hypothesis is dead, and with it the argument for fine-tuning YOLO first.

The band profile does fall from 70.7% at the torso to 40.0% at the feet, and
that is exactly the number I would have quoted as proof if I had not stopped to
think: **it is confounded by anatomy.** Legs are narrower than a torso, so a
lower band has less of its width covered by any correct silhouette. The metric
cannot separate "mask missing" from "body narrower", so it cannot support the
claim. The figure now says so on its face, because a chart that invites the
wrong conclusion is worse than no chart.

**The real gap is the bat: masked in 65.2% of frames.** The paper applies the
overlay to "the batter and bat", so a third of frames get no bat tint. Less
damaging than it sounds -- the overlay exists to normalise *jersey colour and
lighting*, and a bat is pale wood in every clip, so there is little colour
variation there to remove. Worth recording, not worth blocking on.

**Verdict: rebuild now.** The masks are good enough, the overlay is worth
+3.33 points by the authors' own ablation, and fine-tuning YOLOv11 needs
labelled cricket data -- days of work, not hours. It stays on the list, behind
the thing that is already measured.

Settings, from `tools/overlay_sweep.py` across 6 clips at four alphas and three
tints: **grey (150,150,150) at alpha 0.4**. Green was the first guess and the
sweep killed it -- grass is green, so tinting the batter green destroys
figure/ground separation, visibly, at alpha >= 0.45. Between grey 150 and 200
the difference is cosmetic: the tint target shifts the mean but the variance
removed is set by alpha alone, so it is not worth an 8-hour run to settle.

## Day 9 (12 Oct 2026) -- the overlay lost too, and the authors' own weights

The paper's segmentation layer, implemented as the semi-transparent overlay it
describes, finished cleanly at 55 epochs in 373.4 min.

```
                    box      seg blackout   seg overlay   authors
val  top-1         79.23%       62.37%        74.94%         -
test top-1         78.14%       65.08%        75.51%      89.09%
test top-3         96.05%       91.60%        95.19%      99.25%
test macro-F1      0.7605       0.6303        0.7328      0.8847
peak TRAIN top-1   86.88%       64.32%        82.37%         -
```

The overlay recovers **10.4 points** over the blackout, so the misreading
diagnosis was right and the background does matter enormously. But it still
sits **2.6 points below plain cropping**, where the paper reports +3.33. Both
of my readings of "segmentation" have now lost.

Peak train top-1 says why: **82.37% against box's 86.88%.** The overlay model
fits its own training data *worse*, so it is destroying information rather
than normalising nuisance variation -- consistent with tinting over masks
whose boundaries move frame to frame.

**The stale-report guard earned its keep without being asked.** `report_*.json`
in that directory were still the blackout run's, written 10-11 against a
checkpoint replaced on 10-12. `load_report` refused them and printed both
hashes. That is the mechanism built after losing a day to exactly this.

Guessing at their preprocessing has now cost about 14 hours of compute across
two runs and produced two negative results. Time to stop guessing.

### Their shipped weights, and why the first number was a lie

`CricShoot10kModels/` holds not just the classifier but
`Striker_Bat_Segmentation_Model.pt` -- **their fine-tuned YOLO**, which
`detect.py` has been loading all along as `Player_Type_Detection_Model.pt`
(byte-identical). So "fine-tune YOLOv11", which I twice put at the top of the
remaining work, was already done. The masks measured yesterday are their masks.

`tools/authors_model.py` scores their trained classifier on our crops, which
partitions the remaining gap: near 89% means our preprocessing is faithful and
our training is the gap; near 78% means our crops are not their crops.

The first run returned **16.07%**, and the honest answer is that it means
nothing:

```
their config.json  backbone names:  stem_conv, stem_bn, block1a_project_conv
their weights.h5   backbone names:  conv2d, batch_normalization, activation
                   canonical names found in the h5:  0 of 513
```

Their own file is internally inconsistent -- architecture recorded with
canonical names, weights stored with generic ones -- so `load_weights` cannot
match by name and silently falls back to **positional order**. The head
verified clean against the h5 by name (0 mismatched); the 20M-parameter
backbone could not be verified at all. 16% against a 6.7% chance line is what
a misaligned backbone looks like.

The tool now **refuses to emit a number from that path** unless
`--allow_positional` is passed. A diagnostic that quietly reports an
untrustworthy figure is worse than one that fails.

The fix is the version that wrote the file: metadata.json says **Keras 3.8.0**,
installed is 3.12.4, and `TimeDistributed(Functional)` deserialisation broke
between them. An isolated venv with keras 3.8.0 and tensorflow 2.18 leaves the
working `.venv` untouched.

Also measured, and it kills a worry: TensorFlow has no GPU on native Windows
at 2.11+, and the card is an Arc so WSL2+CUDA would not help either -- but CPU
inference is **13 minutes for all 1,976 test clips**, so it does not matter.

## Day 10 (13 Oct 2026) -- the paper had the spec; I had not read it

The crop sweep closed: `max` scored 37.63%, identical to `height`, and
padding peaked at 37.63% with pad 0. Geometry is not the gap.

Then, reading the paper properly rather than for its numbers, section III:

> "the predicted masks were overlaid with semi-transparent colors
> (alpha = 0.5), highlighting the striker in blue and the bat in green.
> Semi-transparent color is chosen instead of full solid to avoid the loss of
> full information such as the position of the hands and rest of the body.
> **The striker is colored first, followed by the bat, ensuring that the bat
> remains fully visible even when it overlaps the striker.**"

Every parameter I had been guessing at is stated there. `CricShot10k_Plan.md`
Appendix B Tier 1 records it too -- "striker blue, bat green, alpha=0.5" --
written before any of this started.

What I built instead: `frame_mask` merged striker and bat into **one** mask and
painted it **one grey** at alpha 0.4. That discards all three specified
properties, and the third one is the damaging one: merging the masks is exactly
what the paper says not to do, because the bat must stay visible over the body.
The bat is the object that names the shot. That overlay scored 75.51% against
plain cropping's 78.14%.

I had searched the paper for "89.09" and the ablation table, read the overview
sentence at line 37 -- "a semi-transparent overlay is applied to the batter and
bat" -- and built from it, without reading the 150 lines further where the
method is written out.

**Rebuilt to specification.** `part_masks` returns the two silhouettes apart;
`overlay` paints the striker blue, then the bat green on top; `carried_mask`
carries both. ALPHA 0.5, BODY_TINT (0,0,255), BAT_TINT (0,255,0), **RGB**,
because `read_clip(rgb=True)` is what fills these arrays. Eight new tests,
including one that asserts the overlap goes green -- the paper's stated reason
for the ordering.

**A second bug, found while fixing the first.** `tools/overlay_sweep.py` wrote
RGB arrays with `cv2.imwrite`, which expects BGR, so every preview I judged
colours from had red and blue swapped. The pipeline data was always correct;
only the pictures lied. Grey hid it completely. Now converted on write, and
the TATA CAPITAL board renders blue instead of orange, which is how it was
caught.

452 tests pass.

**Also noted, and not added to the plan:** flip TTA was already in it twice --
Day 31 and Appendix B Tier 2 #12. Dropout is genuinely absent and is worth a
row, given the box run's 8.5-point train/val gap at `dropout: 0.0`.

### The ruler moved: 34.46% -> 43.22%

Rebuilt `seg` to the paper's specification (4.62 h, 211 failed as before), then
scored the authors' own model on it:

```
authors' model on our box crops            34.46%
authors' model on the old grey overlay    ~34.9%   (180-clip sample)
authors' model on the paper-spec overlay   43.22%   <- +8.76
permutation agreement on the fit sample    32.88% -> 44.44%
```

At 1,976 clips the standard error is about 1.1 points, so **+8.76 is over five
standard errors**. The first positive result in three days, and the first time
this instrument has read anything other than ~35% -- which also answers the
worry that it might be stuck there regardless of input. It is not; it responds
to a correct change.

So the segmentation specification was genuinely part of the gap, and the way
to find it was to read the method section rather than guess from the abstract.

It is still 43% against their 89%, so something else about their crops differs
too. But the ruler is now demonstrably calibrated upward, which makes it worth
more for the next thing we try than it was this morning.

## 2026-10-15 — Day 8 begins: the cache budget

### 8.1 The arithmetic, before any code

The old plan said the feature cache would be `(15, 1280)` and about 600 MB.
Both wrong, and wrong in a way that would have mattered: 1280 is `pool: avg`,
and **every result this project has came from `pool: flatten`**. A 600 MB cache
would have described a model I have never trained.

Where the real number comes from, checked rather than remembered:

```
m.features(zeros(1, 3, 224, 224))  ->  (1, 1280, 7, 7)      224 / 32 = 7
7 x 7 x 1280                       =   62,720 per frame
15 x 62,720 x 2 bytes (float16)    =   1.79 MB per clip
1.79 MB x 9,880 clips              =   17.31 GB
```

Off by 46x. Twenty minutes of multiplication against a day of extraction.

### The budget

```
free on D:                      73.4 GB   (df: 74G avail)
grid, normal              -17.3  -> 56.1 GB
grid, mirrored            -17.3  -> 38.8 GB   (8.7 -- flip cannot be faked)
avg views, derived         -0.7  -> 38.1 GB   (8.8 -- no GPU needed)
other extractors           -0.4  -> 37.7 GB   (8.10)
```

If it ever gets tight, `data/processed/seg` is 21 GB and is the weaker variant
(76.97% against box's 78.14%). It rebuilds from the videos in 4.5 h; the raw
videos rebuild from nothing. Delete in that order and never the other way.

### Why the grid and not the summary

The grid is a superset: average pooling is a mean over the last two axes, so
the 1,280-wide view is derivable from it on the CPU with no second pass over
the video. Measured max difference against `FrameEncoder(pool="avg")`:
**3.6e-07**, which is float32 rounding, so it is a true derivative rather than
an approximation. The reverse is impossible -- once averaged, the spatial
layout is gone and recovering it costs another 14-minute extraction.

Cache the thing you cannot recompute cheaply.

### 8.2 The smoke test refuted my own check

`extract.py` works first time: `(15, 1280, 7, 7) float16`, 1837.5 KB, exactly
the size 8.1 predicted. But I had written the check as "min must be exactly
0.0, because EfficientNetV2-S ends in a ReLU-family activation". It reported
**-0.279296875**.

I was wrong, not the code. `features[-1]` is `Conv2d -> BatchNorm2d -> SiLU`,
and `SiLU(x) = x * sigmoid(x)` goes negative:

```
SiLU minimum                  -0.278464556  at x = -1.2788
nearest bf16 value to it      -0.279296875
measured min                  -0.279296875   <- exact match
```

So the number is not a defect, it is a **two-for-one confirmation**: the right
layer was cached, *and* autocast really ran in bf16, because the value lands on
a bf16 grid point rather than an fp16 one.

The corrected check is a floor rather than a half-plane, which is strictly
sharper: `min >= -0.28 and pressed against it`. A hard `0.0` would now mean a
ReLU backbone (wrong model); far below -0.28 would mean a pre-activation conv
(wrong layer). Plan 8.2 and 8.5 updated.

Lesson worth keeping: I asserted an architecture detail from memory instead of
reading `features[-1]`. The smoke test cost ten seconds and caught it; had I
written the same assumption into a `>= 0` assertion in the test suite, 8.5
would have failed on a correct cache and I would have gone looking for a bug
in the extractor.

### 8.3 killed at 2,699 clips: torch.save writes a view's whole storage

Caught by arithmetic, not by an error. 1,129 files should have been 2.0 GB;
`du` said **8.9 GB**. Per file:

```
smoke test, batch of 1     1,883,177 bytes   = 1.79 MB   correct
batched run, batch of 4    7,527,977 bytes   = 7.18 MB   4.00x
```

Exactly 4x, which is `batch_size`. The cause: `feats[j]` is a **view** into
the batch tensor, and `torch.save` serialises the view's entire underlying
storage, not the view. So every file contained all four clips of its batch and
then read back only its own slice -- correct shape, correct values, four times
the bytes.

Projected cost had it finished: **69 GB against 73.4 free**. It would have
squeezed in, reported success, and then made 8.7's mirrored half impossible
while making every cached epoch read 44 GB instead of 11.

Two fixes, because the first one alone would not have caught it:

1. `feats[j].clone()` -- a fresh storage of exactly the right size.
2. `_check_size()` on the **first** file of every run: compare
   `os.path.getsize` against `numel * element_size` and raise above 5%. One
   `stat` call per run, and this class of bug can never again be silent.

Why the smoke test missed it: `--limit 1` is a batch of one, where the view
*is* the whole storage. **A batch-size-1 smoke test cannot see a
batch-size-dependent bug.** Worth remembering for 8.10, which runs four more
backbones through this same loop.

### 8.3 done: the cache exists

```
9,880 clips   0 failed   8.5 min   19.4 clips/s
files 9880    manifest ok 9880    .tmp left 0
17.3 GB written, 57 GB free      largest activation 32.8 (2000x fp16 headroom)
```

Every file 1,883,177 bytes, one value with no spread -- which is the positive
proof the view bug is gone, not just absent from a sample.

**19.4 clips/s against a predicted 12.1.** The prediction came from a bare
synchronous loop, where batch 8 was no faster than batch 4; that told me the
GPU was never the bottleneck and the .npy read plus float32 normalisation were.
`--workers 4` moves that work off the critical path and the job fell from a
projected 13.6 min to 8.5. The prediction was wrong in the useful direction,
and it was wrong *because* the diagnosis from it was right.

Plan 8.3 and the `extract.py` docstring now carry the measured number, with
the prediction kept beside it.

### 8.4 FeatureDataset: the notebook reads back

```
flat   6320 clips  (15, 62720)        float32  label 0  dim 62720
grid   6320 clips  (15, 1280, 7, 7)   float32  label 0  dim 62720

CricShotDataset 6320 clips, 146 missing
FeatureDataset  6320 clips, 146 missing   agree: True
452 tests still pass
```

The agreement is the check, not the shapes. Both readers see the same 6,320
clips, drop the same 146, and order the classes identically -- so a cached run
and the end-to-end 78.14% are measuring the same thing. That property comes
from importing `_classes_for` out of `dataset.py` rather than re-deriving the
class order here; two definitions of "which shot is label 0" would have made
every cached-versus-baseline comparison meaningless while looking fine.

**146, not 211.** 211 is the project-wide preprocessing failure count across
all three splits; `splits_author/train.csv` owns 146 of them. Caught while
writing 8.4's check, which had the wrong number in it.

`view="avg"` deliberately has no fallback to reducing the grid on the fly. It
raises and names the command that creates it:

```
no feature cache at data\features\effnetv2s\avg for view='avg'.
  Run: python -m src.extract --derive avg  (step 8.8)
```

A silent fallback is how two experiments end up reading different numbers with
nothing to tell you which.

### 8.5 The cache depended on --workers

The check I wrote as "within fp16 tolerance" failed at **55x** relative error,
so I went looking. The chain took four measurements:

```
1  two fresh encodes, same process      IDENTICAL      -> forward is deterministic
2  saved file vs fresh encode           0.39 abs gap   -> 50x bf16 spacing. not noise
3  w0 vs w0, w4 vs w4                   IDENTICAL      -> reproducible per setting
   w0 vs w4                             0.39           -> the flag is the variable
4  loader inputs, w0 vs w4              IDENTICAL      -> not the data. the compute
```

Then the cause, visible only once I printed the tensor flags:

```
workers 0:  is_shared=False  contiguous=False
workers 4:  is_shared=True   contiguous=True
```

`dataset.py` builds its tensor with `.permute(0, 3, 1, 2)`, which makes it
**channels_last**, and that format propagates through the elementwise
normalisation. The DataLoader then decides it silently: at `--workers 0` the
collated batch stays channels_last, at `--workers 4` the shared-memory collate
rewrites it to NCHW. oneDNN picks a different kernel for each and bf16 rounding
diverges, compounding over forty layers.

Neither answer is wrong -- both are valid bf16 computations. But a cache whose
contents depend on a *speed* flag cannot be verified, and it would have made
8.6's epoch-1-loss comparison worthless.

`.contiguous()` fixes it. **`.clone()` does not** -- clone preserves memory
format, which is why my first instinct was wrong. After the fix, `w0`, `w4`
and the cache already on disk are all bit-identical, so **no re-extraction**:
the real run used `--workers 4`, which was already the NCHW path.

The payoff is that check 1 is now `torch.equal` rather than a tolerance. A
tolerance would have passed on the broken version and told me nothing.

```
  1 round-trip against encode_clip   PASS   20/20 bit-exact  max abs gap 0.0000
  2 reshape order, fp32, vs pool=avg PASS   max abs 4.768e-07
  3 activation floor                 PASS   min -0.279297  (SiLU floor -0.278465)
```

474 tests, up from 452. Two of the new ones are regressions for this: that the
worker counts agree, and that `dataset.py` really is channels_last -- so if it
ever stops being, the `.contiguous()` gets reconsidered on evidence instead of
deleted on a guess.

### 8.6 A feature cache is a frozen backbone -- my target was wrong

Epoch 1 of `cached_gru_noflip` came out at train loss 2.3726 / top1 23.47% /
pin 39%, against the 1.9928 / 35.21% / 0% I had set as the target. I had
written that epoch 1's loss is "nearly a pure function of the inputs", which is
true -- and then picked a target whose inputs **change during epoch 1**.

`mimic_author_noflip` has `freeze: false`. Its backbone updates on every one of
1,580 steps, so the features its head sees at step 1,500 are better than the
ones it saw at step 1. A cache is fixed. **No cached run can ever reproduce a
trainable-backbone run.**

The right comparison was in the project already:

```
run                     backbone    ep1 train  ep1 val   pin
author_frozen (video)   frozen        23.07%    30.37%   41%
cached_gru_noflip       frozen        23.47%    26.20%   39%
mimic_author_noflip     TRAINABLE     35.21%    53.98%    0%
```

0.4 points from the frozen row, with the same 40%-ish pinning. **That is the
evidence the cache is correct**, and it was sitting in front of me labelled as
a failure for ten minutes.

`author_frozen` predates the BatchNorm fix, so that table is suggestive rather
than conclusive; the clean version is two epochs of the frozen config on video
with today's code, which is 15 minutes and worth doing.

**What it means for Phase 3, and it is not small.** The 56 cached rows of
Families A, D and V are all frozen-backbone runs, so they are comparable to
each other and **not** to 78.14%. The plan always said those families freeze --
section 6 is explicit -- but 8.6's check had the wrong reference pinned to it,
and that would have read as "the cache is broken" for every one of them.
Days 28-30 are the answer: take the best heads out of the cheap frozen sweep
and re-run only those end to end.

Letting the run finish. It is not confirming a known number -- it is producing
the **frozen baseline**, which `author_frozen` never did because it was
interrupted at epoch 12.

### Family A gains two rows, because 39% dead units is a lead, not a curiosity

Added A13 and A14, so the plan is now **46 experiments / 92 runs** (30 cached,
16 end to end). Thirty edits to keep the arithmetic consistent across the
document.

Both come out of 8.6's `pin 39%`:

```
A13   LayerNorm on the features, then GRU-128
A14   GRU-128 on the "avg" view (1,280) instead of "flat" (62,720)
```

A14 is the one I would not have thought to write a week ago. `flatten` is what
the authors did and `encoder.py` argues for it: the 7x7 grid holds bat position
relative to the body, which is much of the signal in a cricket shot. **But
flatten is also what saturates the GRU** -- 49x more inputs to sum is ~7x
larger pre-activations under random init, and a tanh gate pins. So the two
things are in direct tension and nobody has measured which wins. Two minutes
on the cache answers it.

Also corrected, and it is the kind of error that quietly breaks a schedule:
Family A was budgeted at **0.06 h a run**. The measured `flat` run is **0.15 h
at best and really 1.2 h** -- 70 minutes, 2 min an epoch over 34 epochs. On
`flat` alone Family A costs 16 GPU-hours, not 2.1. The plan's cheap-families
estimate only holds if most of those heads can run on `avg`, which is exactly
what A14 tests. So A14 is both an experiment and a dependency check on the
schedule.

The strategic point worth keeping, because the question came up directly: the
30 cached rows are **screening**, not the final answer. Their job is to decide
which 16 experiments get the 120 end-to-end GPU-hours, where 78.14% can
actually be beaten. Days 28-30 are the bridge -- top 3 from the cheap sweep,
re-run end to end with 3 seeds.

### 8.6 complete: the frozen baseline is 51.20%

```
best        epoch 68   val 51.20%   top3 80.62%
final       epoch 78   train 86.90% / val 51.07%  -> 35.8-point gap
pin         39% to 53%, mean 45%, never resolved
wall clock  93.6 min, 78 epochs, clean finish (no resume.pt, figure drawn)
```

So the three reference points now read:

```
mimic_author          trainable + flip   79.23% val   78.14% test
mimic_author_noflip   trainable          75.95% val
cached_gru_noflip     FROZEN             51.20% val     <- Family A's reference
```

**Freezing costs 24.75 points**, not the 12 I wrote from Day 7's unfinished
comparison. Every Family A, D and V row sits in this regime.

Two things in the run that are leads rather than noise:

**A 35.8-point train/val gap.** 86.90% train against 51.07% val. The head is
memorising, because frozen ImageNet features give it nothing generalisable to
latch onto. That makes augmentation worth far more here than on the baseline,
where the gap was 10.3 -- so 8.7's mirrored cache is better motivated now than
when it was written.

**pin never resolved.** 39% at epoch 1, 45% mean, 46% at epoch 78. Not a
warm-up transient; a permanent loss of about half the GRU for the whole run.
A13 goes after exactly this.

**And the lr tail is a waste to fix before Phase 3**: epochs 54-78 ran at
lr <= 1e-06 for 30 minutes and bought **+0.19 points** -- three val clips out
of 1,584. `min_lr: 0.0` lets plateau divide by ten forever, down to 1e-08 by
epoch 73, and each noise-sized "best" resets patience by 10 epochs. At 92 runs
that is roughly 30 GPU-hours spent resampling validation noise. The experiment
registry's base config should carry `min_lr: 1.0e-06`; `mimic_author.yaml`
keeps 0.0 because it reproduces the authors and every baseline used it.

### 8.7 The mirrored half

```
9,880 clips, 0 failed, 9.5 min, 17.3 clips/s
grid         9880 files, all 1,883,177 bytes, 0 .tmp
grid_mirror  9880 files, all 1,883,177 bytes, 0 .tmp
manifests    9880 ok rows each, in separate files
disk         39 GB free -- 8.1 budgeted 38.8
```

Re-measured the shortcut against the *real* mirrored features now that they
exist: **cosine 0.3753**. So the 0.41 from the earlier three-clip test holds up,
and the second 14-minute pass was not optional.

Two design points worth keeping.

**One manifest per thing being built.** `manifest_grid_mirror.jsonl`, not the
shared one. Had they shared, the mirror run would have read all 9,880 clips as
already done and written nothing while reporting success -- which is exactly
what `build_dataset --variants seg` did on Day 4.

**Flip is a choice of file, and that makes it exact.** `ClipFlip` is a binary
decision made once per clip, so two cached halves cover the entire
distribution; `FeatureDataset(mirror=0.5)` reads from one or the other. Checked
it: 12 draws of item 0 gave exactly **2** distinct tensors. Not an
approximation of the video run's augmentation -- the same distribution.

The extractor reuses `ClipFlip(1.0)` rather than its own `torch.flip`, so
"mirrored" has one definition in the codebase and it is the one training uses.

### 8.7 result: flip is worth LESS when frozen, and that is the interesting part

```
                   best val   overfit gap at the end      flip worth
frozen   noflip      51.20%   86.90 / 51.07 = 35.8
frozen   flip        53.66%   67.85 / 52.71 = 15.1        +2.46
trainable noflip     75.95%                      10.3
trainable flip       79.23%                       4.3        +3.28
```

I predicted flip would be worth **more** here, maybe 8-10 points, on the
grounds that a 35.8-point train/val gap gives augmentation three times as much
to bite on. It was worth **less**. The prediction was wrong and the reason is
the result:

```
frozen     removed 20.7 points of overfitting, converted 2.46   -> 12%
trainable  removed  6.0 points of overfitting, converted 3.28   -> 55%
```

**Flip suppressed the memorising almost completely and the accuracy barely
moved.** That is the signature of an information ceiling, not a regularisation
problem: ImageNet features do not contain the discriminative cricket
information, and stopping the head from memorising does not create it. The
frozen head was not failing because it overfits; it overfits *because* it is
failing.

Three consequences, and they are all actionable:

1. **Tier 2's regularisers will underdeliver on cached runs.** Dropout, colour
   jitter, random resized crop, speed perturbation -- all of them are attacking
   a symptom that is already 12%-convertible at best in this regime. They may
   still earn their estimates on the trainable runs, where conversion was 55%.
2. **A13 and A14 are the only credible frozen levers left.** They go after
   feature quality and saturation rather than regularisation, which is now
   measured to be the wrong target here.
3. **`pin` did not move**: 45% mean without flip, 46% with. Augmentation has no
   effect on the saturation at all, so that really is a separate problem with a
   separate fix.

Also worth noting: the flip run early-stopped at **epoch 43 in 58 min** against
78 epochs and 93 min. Less overfitting meant less of the long low-lr tail --
so the `min_lr` waste from 8.6 is partly a symptom of unregularised runs.

### A file per method, asked for 2026-10-15

`train.py` had grown two `if cfg["features"]` branches -- one choosing the
Dataset, one choosing the model -- and with 46 experiments coming that was
going to become a chain of them. Split instead:

```
src/train.py                 the loop. Zero method names in it now (grep: 0).
src/methods/__init__.py      load(name)
src/methods/video_gru.py     36 lines  encoder + GRU, end to end
src/methods/cached_gru.py    40 lines  frozen cache + GRU head
```

A method answers exactly two questions and nothing else:

```python
datasets(cfg)                -> (train_set, val_set)
model(cfg, train_set, n_cls) -> nn.Module
```

**What I did not do, and why.** Full copies of `train.py` per method would put
the resume logic, the early-stopping rule, the ledger write and the seed
handling in 46 places. The `channels_last` bug on 2026-10-15 took one fix
across everything; with 46 copies that is 46 fixes and no way to know which was
missed. It would also quietly break rule 2 -- two methods could differ in their
stopping rule with nobody intending it, and the ledger would read as though
they differed in the thing under test. So: **separated per method, shared per
concern.**

Fallout, all fixed: every config gained a `method:` key; the ledger row now
records it; the banner prints `name [method]`; and three tests in
`test_train.py` were reaching into `train.py` for `build_model`, which now
lives only in `src/models/model.py` where it belongs. 474 tests pass.

Adding A13 is now a new file, not an edit to the loop.

### 8.8 The avg view: 1,280 wide, derived on the CPU

```
grid/ -> avg/                9,880 clips, 0 failed, 5.1 min, 32 clips/s
grid_mirror/ -> avg_mirror/  9,880 clips, 0 failed, 3.8 min, 43 clips/s
each file 39,977 bytes       389 MB per half, 0 .tmp, 38 GB free
vs the float32 mean          max diff 6.3e-04 (fp16 rounding)
tests                        480 in the suite; 27 pass and 1 skips in test_extract
```

The one skip is correct, not a gap: `test_the_avg_view_names_the_command_that_
would_create_it` only means something while `avg/` is missing, so it skips
once the folder exists. I had told myself "0 skipped" -- wrong prediction,
right behaviour.

**Estimated 2.2 min, took 8.9.** Two mistakes in the estimate, both mine: I
timed the first 300 files, which tests and `--verify` had just read, so they
came from the OS cache; and I timed reading and averaging but not writing.
Measured properly afterwards, on clips 5000-5100:

```
read + average       10.6 ms
atomic_save fsync     2.3 ms
manifest-row fsync    7.1 ms
total                20.0 ms   -> ~50 clips/s, against 32-43 in the run
```

Half of every clip is the two fsyncs. That is what makes a power cut cost one
clip instead of a corrupt cache, so it stays. A job that runs once per
extractor can afford nine minutes; it cannot afford a half-written file.

The pay-off is the next experiment: A14 can now run on `view: avg`, where the
GRU's input layer is 1280 x 384 = 0.49 M parameters instead of 24.1 M and an
epoch reads 231 MB instead of 11 GB.

### 8.10 part 1: ConvNeXt-Tiny, and why it did not just plug in

The plan said `--backbone convnext_tiny` and go. It crashed:

```
FrameEncoder("convnext_tiny", pool="avg")
-> ValueError: backbone returned (1, 768, 1, 1), expected (N, D)
```

ConvNeXt's head is `LayerNorm2d -> Flatten -> Linear`, and `_strip_classifier`
replaced the whole head with `Identity`. The LayerNorm is the network's last
real layer, not part of the classifier. So the strip now drops only the Linear
(and Dropout, which would fire in a trainable run). EfficientNet's head is
`Dropout -> Linear`, so nothing is kept and it is still `Identity`; the 8.5
bit-exact checks still hold.

Two more things followed from it:

- **ConvNeXt's avg view cannot come from a grid**: the LayerNorm runs after
  the pool, and LN(mean) is not mean(LN). `--pool avg` writes `avg/` directly.
- **The default cache folder was `effnetv2s` for every backbone.** The second
  backbone would have seen a full manifest and done nothing, silently. It now
  defaults to the backbone's name, and `source.json` pins backbone + variant.

The new check for a pooled cache is that the pretrained ImageNet Linear on
the cached vector reproduces the full model's logits. Smoke run, 4 clips:

```
round trip            4/4 bit-exact
ImageNet head         logit cosine 0.9999, top-1 45/45 frames
same, no LayerNorm    logit cosine 0.87,   top-1 11-12/15 frames
                      (features ~6x smaller: mean |x| 0.09 vs 0.51)
```

Top-1 mostly survives the bug, so an accuracy check would have missed it.
The cosine catches it.

Tests: 484 pass, 2 skip (avg already derived; no ConvNeXt cache yet). The
suite crashed with MemoryError in one process: llama-server and a CricketMind
eval were using the RAM, with 4.2 of 15.6 GB free. Run file by file, every
file passes.

**The full ConvNeXt run (2026-10-16):**

```
avg/           9,880 clips, 0 failed, 9.2 min, 17.9 clips/s
avg_mirror/    9,880 clips, 0 failed, 10.9 min, 15.0 clips/s
each file      24,617 bytes, 243 MB per half, 0 .tmp, 38 GB free
verify         20/20 bit-exact; logit cosine 0.9999, top-1 43/45 frames
```

I guessed ~225 MB per half; it is 243 because every file carries ~1.6 KB of
torch.save header on top of its 23,040 bytes. The 43/45 is not a failure:
cricket frames are not ImageNet classes, so the top two logits are often
nearly tied and bf16 noise picks either. Cosine 0.9999 says it is the right
layer.

### 8.10 part 2: r3d_18, the first model that watches movement

Decision first: **all four extractors get built**, VideoMAE and CLIP
included. I had suggested holding those two back until r3d_18 showed motion
mattered; the full comparison is the point.

New file `src/models/video_encoder.py`. The clip goes in whole, channels
before time, and one 512-vector comes out. Size and colours are read from
torchvision's weights (112x112, Kinetics mean/std), not from `dataset.py`.

Found by reading, before it could bite: `FeatureDataset.feature_dim` was
the product of every axis after time. For a `(512,)` tensor that is the
empty product, **1**. It now reads 512, and each view checks its number of
axes.

```
smoke, 8 clips        8/8 bit-exact; logit cosine 1.0000; top-1 8/8
r3d_18's own guess    'playing cricket' 38/40 clips, mean prob 0.92
wrong input spec      (ImageNet, 224) top-1 still 38/40, mean prob 0.62
```

Two things I did not expect. Kinetics has a "playing cricket" class, so
this backbone already knows the sport. And the wrong input spec does not
fail: top-1 is unchanged and only the confidence drops, so nothing but
reading the spec from the weights would catch it.

Tests: 494 pass, file by file. `test_plot.py` failed 4 tests once and passed
4 runs in a row after; it imports nothing that changed, and free RAM was low
with llama-server running. Logged, not explained.

**The full r3d_18 run (2026-10-16):**

```
video/          9,880 clips, 0 failed, 2,601 bytes each, 25.7 MB
video_mirror/   9,880 clips, 0 failed, 2.7 min, 61 clips/s
verify          20/20 bit-exact; cosine 1.0000; top-1 20/20;
                'playing cricket' 19/20
```

The first half was cut at 1,664 clips (manifest gap 00:29:32 -> 00:54:39,
most likely load shedding) and the rerun did exactly the remaining 8,216.
The first real test of resume on a cache, and it held.

60 clips/s against ConvNeXt's 18 -- with an extra resize in the loader. The
difference is bytes moved per clip (112px is a quarter of 224px), so my
Day 8 "loader-bound" story was incomplete. Logged, not chased.

### 8.10 part 3: VideoMAE, a video transformer

`pip install transformers` gave 5.20.0; torch stayed 2.13.0+xpu. First
download failed: `HF_HUB_OFFLINE=1` is exported in `~/.bashrc` for another
project. Left it alone; lifted it for one command to fetch
`model.safetensors` (346 MB) into `D:/huggingface_cache`. Everything since
runs offline. A missing model now raises with the download command in the
message.

Checkpoint: `MCG-NJU/videomae-base-finetuned-kinetics`. Its vector is the
mean of all tokens after `fc_norm` -- what the classifier reads -- so only
the classifier is cut.

15 -> 16 frames by even picks (frame 7 twice). Measured against repeating
the last frame, 40 clips spread over the classes:

```
even picks       'playing cricket' 40/40, mean prob 0.625
repeat last      'playing cricket' 40/40, mean prob 0.629
```

No difference worth the name. I had argued for even picks on timing
grounds; the data says it does not matter for this check. Kept anyway.

```
smoke, 8 clips   8/8 bit-exact; logit cosine 1.0000; 'playing cricket' 8/8
speed            18.0 clips/s on 200 clips -> ~9 min per half
```

VideoMAE is surer it is cricket than not (40/40) but spreads its
probability more than r3d_18 does (0.63 against 0.92). Not a ranking --
recognising the sport is not telling a drive from a pull. The probe decides.

One test bug caught before commit: a module-level `importorskip` would
have skipped the r3d_18 tests too on a machine without transformers. Moved
into the VideoMAE tests. 501 pass, file by file.

**The full VideoMAE run (2026-10-16):**

```
video/          9,880, 0 failed -- cut at 6,860, resumed with 3,020 to go
video_mirror/   9,880, 0 failed, 6.1 min, 27.1 clips/s
each file       3,113 bytes, 30.8 MB per half; 37 GB free
verify          20/20 bit-exact; cosine 1.0000; 'playing cricket' 20/20
```

My 200-clip timing said 18 clips/s; the real run did 25-27. Start-up cost
spread over 200 clips is not small. Three of four caches done.

### 8.10 part 4: CLIP ViT-B/16

`openai/clip-vit-base-patch16` has no `model.safetensors` -- only
`pytorch_model.bin` (599 MB). Downloaded once with offline mode lifted, as
for VideoMAE; transformers loads `.bin` with `weights_only`.

Decision: **768 per frame, not the plan's 512.** The 512 is one bias-free
Linear after the 768, and the CLIP paper probes the 768. A linear probe on
768 contains every probe on 512.

In transformers 5.20, `get_image_features` returns an output object whose
`pooler_output` is the *projected* embedding, and `CLIPVisionModel` is the
tower itself (no `.vision_model` inside). Both checked before writing code.

Check 2 needed a new shape: CLIP has no classifier, so the "head" is the
projection and the class scores are similarity to eight "a photo of a
person playing <sport>" prompts. Does it discriminate?

```
right layer          cosine 1.0000
CLS before final LN  cosine 0.4764
patch-token mean     cosine 0.4140
```

Easily. Smoke run: 8/8 bit-exact, embedding cosine 0.9998, zero-shot
'cricket' on 45/45 frames. 19.8 clips/s on 400 clips.

Refactor on the way: the HF cache lookup and quiet loading moved from
`video_encoder.py` to `src/models/hf.py`, shared by both HF encoders.
509 tests pass, file by file.

### Are all the results saved? Mostly -- one gap, closed

Asked 2026-10-16. Checked rather than assumed:

```
training runs     experiments/results.jsonl (12 rows), history.json per run,
                  report_val/test.json for mimic_author and _seg     in git
checkpoints       experiments/<run>/best.pt                          disk only
feature caches    data/features/<name>/ + manifests + source.json    disk only
cache summaries   extract / derive / verify output                   TERMINAL ONLY
```

The last row was the gap. The time, speed, failures and the verify checks
existed only in the terminal, and survived because I copied them into this
journal by hand. Now every command-line `extract`, `derive` and `verify`
appends a row to **`experiments/caches.jsonl`**, which is tracked
(`.gitignore` only drops `experiments/*/*`). Library calls default to
`ledger=None`, so tests cannot write to it.

The caches themselves stay out of git on purpose (37 GB, rebuildable from
`src/extract.py` + `data/processed/`). Checkpoints stay out too; the three
runs without per-class reports (`mimic_author_noflip`, `cached_gru`,
`cached_gru_noflip`) still have `best.pt`, so those reports can be made later.

Still to clean: four `smoke_*` rows in `results.jsonl` from my own smoke
tests (Days 7-8). The Day 9 leaderboard must not rank them.

### 8.10 done: five caches

```
clip_vit_b16   avg/ 9,880 + avg_mirror/ 9,880, 0 failed, 7.5 min each,
               24,617 bytes per file, 243 MB per half; verify passed,
               embedding cosine 0.9999, zero-shot 'cricket' 45/45 frames
```

One slip on the way: `--backbone videomae_base --mirror` was run instead of
`clip_vit_b16`. It printed "9880 already done, 0 to go" and wrote nothing --
the manifest doing its job -- and with the ledger change it also writes no
row for a run with nothing to do.

All five caches, verified 2026-10-16 and logged in `experiments/caches.jsonl`:

```
effnetv2s       grid + avg     all 3 checks
convnext_tiny   avg (15,768)   bit-exact, cosine 0.9999   guess 'ballplayer'
r3d_18          video (512,)   bit-exact, cosine 1.0000   'playing cricket'
videomae_base   video (768,)   bit-exact, cosine 1.0000   'playing cricket'
clip_vit_b16    avg (15,768)   bit-exact, cosine 0.9999   'cricket'
```

1.09 GB for the four new ones with both halves, against the plan's 400 MB
(CLIP is 768 wide, not 512, and every cache has a mirror half). 36 GB free.
Tests: 512 pass, file by file. Next: Day 9, the runner and the probe.

## 2026-10-16 — Day 9 begins: the experiment harness

### 9.1 The registry: 51 experiments as data

`src/experiments/registry.py` holds every experiment as a base plus a delta of
at most 3 keys. 46 IDs from Appendix A, plus five probe IDs (P01-P05: one
Linear on each 8.10 cache). 102 run names, derived as `<id>__<split>`, never
written by hand.

The plan's own examples were wrong against Appendix A: `A03_bilstm` (A03 is
GRU-128), `A01_gru128` (A01 is mean pooling), `B00_baseline` (no B00; the
authors' backbone is B01). Fixed in the plan to `A07_bilstm64`,
`P03_r3d_18` and `B01_effnetv2s`.

What I did not expect: a key being in `DEFAULTS` does not stop a silent
no-op. `cached_gru` never reads `backbone`, so a delta setting it would train
on EfficientNet features and still write `resnet50` into the ledger row. Each
method now lists the keys it acts on (`READS`), `train.py` lists its own
(`LOOP_READS`), and a test fails on any delta key nobody reads.

`--list` names what blocks each ID. Today: 6 ready (B01, B04-B08); the rest
wait on `cached_head`, `classical`, `video3d` or a missing cache, and B02/B03
on backbones torchvision does not have (EfficientNetV2-B3, Xception).

A path slip caught by a test: `os.path.join` gives `data\splits_author` on
Windows while the yamls and ledger say `data/splits_author`. The registry now
writes `/`, which both OSes accept.

Tests: `test_registry.py` 20 pass, `test_train.py` 28 pass.

### 9.2 The runner

`python -m src.experiments.run --exp ... | --family ...` decides one of four
actions per run (train, resume, adopt, skip), then scores whatever reports
are missing or stale. Test is scored and stored but never printed.

Two things the plan did not foresee:

- `evaluate.py` could only score the video model. A harness that scores
  every run needs it to score any method, so each method now has
  `eval_set()`, and the checkpoint's own config picks the method.
- Every P, A and V entry needs `cached_head`, which was Day 10 work. Built
  now with three heads (linear, mlp, gru) so the probe runs today.

B01's author row is the old `mimic_author` run (79.23% val), adopted, not
re-trained: its stored config differs from B01's only in keys added since,
all at defaults that reproduce the old code. 200 GPU-minutes saved, and the
comparison with the grouped row stays like-for-like.

A ledger honesty fix: P03's inherited config still says `efficientnet_v2_s`
(from `cached_gru.yaml`), so the row's `backbone` now comes from the cache's
`source.json`.

Tests: 38 new/changed pass (`test_run_experiment` 11, `test_cached_head` 7,
`test_registry` 20); 133 in evaluate/train/head/model/plot/engine still pass.
17 IDs ready.

### 9.2 run: the probe, and B01 adopted

`--family P`: 10 runs, 6.5-10.5 min each, all finished; P03 (run first as
the 9.2 check) was skipped on the family run, as it should be. B01's author
row was adopted from `mimic_author` in under a second.

```
P04 VideoMAE-B   52.21% / 56.98%   (author / grouped val)
P02 ConvNeXt-T   49.49% / 51.56%
P03 r3d_18       41.41% / 43.59%
P01 EffNetV2-S   40.66% / 44.42%
P05 CLIP B/16    39.14% / 42.96%
```

VideoMAE first and ConvNeXt second on both splits. The small video model
only ties EfficientNet, so "video pretraining" alone is not the lever; a
strong video model is. CLIP last, despite naming the sport on 45/45 frames.
One seed each: r3d_18 vs EfficientNet (under a point) is a tie.

Every probe underfits rather than overfits (train top-1 50-67%), so the
absolute numbers are a floor set by the protocol; the ranking is the result.

### 9.3 The harness, proven

Six tests: idempotence (a second call trains nothing, scores nothing, and
touches no file), determinism for two heads with flip on, a different seed
changing the numbers (so determinism is not vacuous), the row and the
checkpoint both recording the resolved config, and every real ledger row
still matching what its ID means today.

What I half-expected and did not get: XPU bf16 nondeterminism. Two forced
runs gave identical histories, epoch by epoch, and identical confusion
matrices. That holds for small linear/MLP heads; the big GRU and the
end-to-end runs are not covered by this test and may not be bit-exact.

Tests: 44 pass across `test_run_experiment` (17), `test_registry` (20),
`test_cached_head` (7).

### 9.4 The leaderboard

`python -m tools.leaderboard`: one table per family, author and grouped side
by side with the gap, the latest row per run, adopted rows starred, and a
list of what is still missing. Today: 11 of 102 runs done, 45 IDs not
started, B01 waiting for its grouped run.

Every probe has a negative gap (grouped val higher by 2-5 points). Not a
finding about leakage: the two val sets are different clips. The gap that
matters is B01's, on test, after 9.6.

Tests: 9 pass (`test_leaderboard.py`).

### 9.5 Figures for every experiment

Each scored run now gets a report card (confusion matrix as % of each true
class, per-class F1), and `python -m tools.figures` rebuilds six comparison
figures plus `notes/figures/index.md` from the ledger. Validation only.

What looking at the drafts caught (the validator only checks colour):
- the colour bar sat on top of the heatmap's last column;
- the two heatmap panels had different row heights, because one split had
  a run the other did not; now both share the same rows, blanks for gaps;
- the cost plot's log axis printed "7x10^0", and its best-run label ran
  into the neighbouring panel.

First read of the mimic_author card: Late Cut is the weakest class (F1
0.56), confused with Cover Drive (15%), Square Cut (15%) and Defensive
(12%); Reverse Sweep goes to Scoop 21% of the time. In the probe heatmap,
Late Cut is the palest column for every cache.

Tests: 83 pass across plot, figures, harness, leaderboard, registry and
cached_head; train and evaluate 43 pass.

## 2026-10-17 — 9.6, the grouped-split baseline

`B01_effnetv2s__grouped`: the authors' exact configuration on the split
with no shared source videos. 56 epochs, early stop, best at epoch 46.

```
test top-1   78.14% author   75.23% grouped   -2.91
macro-F1     0.761           0.731            -0.030
val top-1    79.23%          77.69%           -1.54
```

Lower, so no sign of a leak in the grouped split, and lower by less than I
expected: about 3 points of the authors' number comes from having seen the
test footage. The rest of the gap to 89.09% is something else. Hook,
Straight Drive and Lofted Offside lose the most (about 0.1 F1 each).

Also found: the authors' val split overlaps train on 368 of 369 videos, so
it is as in-domain as their test. Model choice from now on should lean on
grouped val.

What broke: a `MemoryError` at epoch 4. RAM, not VRAM: 4.7 GB free with
another project's eval running. Re-ran the same command; resumed at epoch 4
with best 58.13%, exactly as designed. Then the row's `minutes` read 355.0
against 375.5 of epoch time, because it measured wall clock since the
resume. Fixed in `train.py` (sum of epoch secs, plus a `resumed` flag) with
a test. Also added the LaTeX build files to `.gitignore`.

### 9.7 Phase 2 closed

Tagged `phase2-complete`. Rewrote the plan's Phase 2 inventory to match what
was built: 51 IDs not 46, 1.09 GB of extra caches not 400 MB, four views not
three, plus `cached_head`, the method-generic evaluator, the probe ranking
and the grouped baseline. Flagged for Phase 3: end-to-end epochs ran at 6.7
min against 4.4 with another program open, so B and C need the machine to
themselves, and section 6's cost table needs re-checking after B02.

## 2026-10-17 — Day 10 begins: Family A's heads

### 10.1 The controls: mean and max

`PoolHead` is `GRUHead` with the GRU replaced by `pool_frames` (average or
largest value over the 15 frames); the tail (BatchNorm with the authors'
eps and momentum, Dense 1024, Dense 15) is the same. A test shuffles the
frames: the pool heads give the same answer, a GRU does not.

Not obvious until I counted: the "dumb" control has more weights than the
GRU head (about 64M against 24M), because its first Dense reads 62,720
numbers where the GRU's reads 128. If A01 ties A03, capacity is one
possible reason, and the write-up should say so.

Tests: 64 pass (cached_head, registry, head); harness 18 pass. A01 and A02
`ready`.

### 10.2-10.6 The other heads, built in one go

Asked for all of Day 10's code at once, with the walkthrough while Family A
runs. Built: bidirectional GRU, LSTM (Keras init, forget bias 1), a temporal
transformer (project to 256, learned positions), a TCN (dilations 1-2-4,
reach 15 frames), attention pooling (weights kept for the A11 figure),
NetVLAD (8 centres), and a per-frame LayerNorm wrapper for A13. All share
GRUHead's tail through one small base class.

The bug the runner test found: the transformer passed every CPU test, then
crashed on the Arc at the first validation, "expected scalar type BFloat16
but found Float". In eval mode torch's transformer uses a fused fast path
that does not notice XPU autocast. Turned off for that one call.

Checked against the committed code, not just itself: the new GRUHead draws
identical starting weights, so A03 should land on 53.66% exactly.

Also noticed: my shell's `python` is the system Python 3.10, not `.venv`.
Same torch (2.13.0+xpu), so earlier results stand; tests now run with
`.venv/Scripts/python.exe`.

Tests: 172 + 27 pass across the affected suites. All 14 Family A IDs ready.

## 2026-10-17 — Day 11: A03 did not equal cached_gru, and why

A03 (the cached_gru model, built by cached_head) scored 52.59%, not 53.66%,
and differed from epoch 1. Not a code bug: rebuilt both the way train.py
does and got identical starting weights, identical first batches and an
identical random state. The cause is the GPU: the same config three times
for 250 batches gave train loss 2.5613, 2.5574, 2.5523. The flat GRU's
62,720-wide matmuls are not bit-repeatable on the Arc; 9.3 only covered
small heads, and I had written that it might not hold for the GRU.

`torch.use_deterministic_algorithms(True)` made two runs bit-identical with
no warnings and no measurable slowdown (~13 s per 250-batch epoch both
ways). train.py now always uses it and the row records `deterministic`.
conftest resets it between tests.

The number worth keeping: identical code and seed, 53.66% vs 52.59%. A
single flat-GRU run is good to about a point, from the GPU alone. Gaps
under ~1.5 points in Family A are ties until Phase 4.

Tests: 199 pass with the project's .venv.

## 2026-10-18/19 — Family A finished, and A03 a third time

All 28 runs, about 20 hours, every row deterministic (A03 author re-run
with --force on the 19th: 51.96%).

The same model and seed, three times: 53.66% (cached_gru), 52.59% (A03,
not deterministic), 51.96% (A03, deterministic). Spread 1.70 points. That is
the noise floor for comparing single runs of the flat GRU.

What Family A says (grouped val first, author second):
- Shrink first, then read: transformer 4L/8H 70.11 / 68.81, TCN 69.66 /
  67.17, transformer 2L 68.96 / 66.22, attention pooling 68.13 / 64.90,
  NetVLAD 63.54 / 65.03. All project 62,720 -> 256 before the temporal step.
- The recurrent heads on the raw 62,720: 50-56%. The same GRU on the
  1,280-wide avg view (A14): 63.10 / 62.44, with 0% pinned and 0% dead
  against A03's 41% / 16%. The GRU's trouble is input width.
- A13 (LayerNorm, then GRU) made it worse: 52.71 / 49.43, pinned 53%, dead
  33%. LayerNorm centres the features but also scales the many near-zero
  SiLU values up to unit size, so the 62,720-wide sum into each gate gets
  larger, not smaller. My hypothesis from 8.6 was wrong; the write-up should
  say so.
- Order matters a little: attention pooling (order-blind) 68.13 against the
  transformer's 70.11 and the TCN's 69.66, within noise of each other.
- The authors' recurrent ranking (BiLSTM second best at 87.65%) does not
  carry over to frozen features: every recurrent head sits in 48-56%.
- Gap (author - grouped) across A: -5.41 to +3.32, mean -1.50. Each gap is
  two single runs, so about +-2 is noise.

Open question, not in the registry: a GRU after the same learned 256-wide
projection. It would separate "shrink" from "attention" cleanly. A Phase 4
candidate.

## 2026-10-19 — 11.3 and Day 12 built

`tools/attention.py`: A11's weights averaged per true class over the val
clips. Attention climbs across the clip: frames 1-6 about 0.02 each (even is
0.067), frames 12-15 up to 0.16. Hook and Upper Cut peak at frames 10-12,
earlier than the rest; Reverse Sweep and Scoop spread theirs. Both splits
agree. So the backlift carries little and contact carries the class, which
also suggests the 15 frames are spent unevenly: a question for Phase 4's
"frames: 15, 20, 25".

`src/methods/classical.py`: Family D, fitted once on the avg view averaged
to 1,280 numbers, standardised on train only, scored into evaluate.py's
report format. The runner now calls a method's own `fit`/`score` when it
has them.

Found: XGBoost was in the system Python, not in `.venv`. My library check
on Day 14-planning ran the wrong interpreter. Added to requirements.txt.

Tests: 107 pass, 2 skipped (XGBoost, until installed).

## 2026-10-19 — Family D finished

Classical models on the avg view averaged to 1,280 numbers per clip,
default settings, val top-1 author / grouped:

```
D01 linear (logistic)   42.49 / 45.70   0.1 min
D06 mlp                 39.39 / 44.17   0.1 min
D04 xgb                 36.17 / 41.56   5.9 min
D02 svm (rbf)           34.79 / 40.47   0.5 min
D03 random forest       27.21 / 29.32   0.2 min
D05 knn (k=10)          18.50 / 23.07   0.0 min   (chance is 6.7%)
```

- The plainest model wins. D01 agrees with P01 (40.66 / 44.42), the same
  linear probe fitted by gradient descent: the 9.1 cross-check passes.
- k-NN at 18.5% says clips of the same shot are mostly not neighbours in
  frozen ImageNet space: what sits close together is more likely the same
  ground or batter than the same stroke.
- Every D model is far below Family A's 70%. D sees one averaged vector:
  the 7x7 grid and the time order are both gone. A's heads keep both.
- Untuned on purpose; the SVM and the forest might do better with tuning.
  The claim is "how separable are these features with default models", no
  more.

A bug, caught by reading the output: for RF and k-NN the val score at fit
time and in the report differed (27.21% vs 27.27%, 18.50% vs 18.62%). Ties:
k-NN votes and forest probabilities tie often, sklearn's predict gives a tie
to the first class, and my scoring used a sort that could give it to either.
Now a stable sort; rescoring the real D03 and D05 models matches the fit to
the last digit. Their reports need redoing (scores only, no refit).

## 2026-10-19 — Day 14 built: seven new extractors

Read each model's own classifier before deciding what to cache:
- mc3_18, r2plus1d_18: like r3d_18 (fc, 112px, Kinetics colours).
- swin3d_t: `head` Linear after its pool; takes 15 frames.
- mvit_v2_s: `head` = Dropout + Linear on the class token; its weights say
  min_temporal_size 16, so it gets 16 resampled frames.
- s3d: the surprise. No Linear at all: the classifier is a 1x1x1 Conv3d,
  and forward() averages *after* it. Since a linear map commutes with an
  average, removing the classifier leaves exactly the 1,024-vector the conv
  reads; for verify the conv is rebuilt as a Linear.
- TimeSformer: classifier reads the first token after the final LayerNorm;
  8 frames, so every other one of the 15.
- DINOv2: its ImageNet head reads cat(first token, mean of patches) = 1,536,
  which is also DINOv2's own linear-evaluation feature. Cached that, from the
  checkpoint that ships the head, so verify has something to check against.

Every new encoder + its model's own last layer equals the whole model on
random weights; all seven run on the Arc in bf16. The r3d_18 and VideoMAE
caches re-verify bit-exact after the refactor.

Tests: 89 pass, 12 skip until the downloads and caches exist.

## 2026-10-19 — Day 14 run: six caches clean, DINOv2 short by 9

The overnight loop: mc3_18, r2plus1d_18, s3d, swin3d_t, mvit_v2_s and
TimeSformer each wrote 9,880 + 9,880 with 0 failures and verified bit-exact,
the pretrained head agreeing on 20/20. Every Kinetics model's favourite
guess for our clips is "playing cricket", which is a nice sanity check.

DINOv2 wrote 9,871: 9 clips came out NaN, one per batch, across classes.
I ran those 9 again on their own and in their exact original batches of 4,
three times each: finite, identical, in bf16 and fp32. So nothing about the
clips; a rare one-off on the GPU (9 in 2,470 batches).

The useful part: re-running would have done nothing. The resume added every
manifest row to "done", failed ones included, so the 9 would have stayed
missing forever (the completeness test would have caught it, with no way to
fix it short of editing the manifest by hand). Now only written clips count
as done, in extract and derive both, with a test.

Later the same morning: the mirrored half lost 10 more clips, all different
ones. So not a one-off. I recomputed all 9,880 written clips the way
extract does and compared bit for bit: no NaN this time, but 86 clips (0.9%)
didn't match their own files, by up to 0.05, about one bf16 rounding step.
Verify had passed 20/20 because 20 clips is too few to see a 1% effect.

Then one batch, 1,000 times in a row, no DataLoader: SDPA attention in bf16
gave 5 odd answers, each in one frame; eager attention gave 0, and so did
SDPA in fp32. Deterministic mode didn't reach it. CLIP and VideoMAE use the
same kernel and drift too, but much less (2/500, 1/300, gaps 0.012 and
0.004) and never went NaN in their runs.

Decision: DINOv2 switches to eager attention. It is half as fast on the GPU
(23 vs 43 clips/s), but loading is 20 clips/s, so the run takes as long as
before. Its cache is redone from scratch. CLIP and VideoMAE stay as they
are: complete, NaN-free, the drift is below anything training would notice,
and finished Phase 2 runs used them. Lesson: "bit-exact on 20 clips" says
the code is right; it doesn't say the GPU repeats itself.

DINOv2 redone with eager attention: 0 failed on both halves, verify
1,000/1,000 bit-exact. I was wrong about the speed, though: 14 clips/s,
11.6 min a half, not the ~8.5 I predicted from a 23 clips/s GPU-only number.
Under full load the GPU became the bottleneck. 6 extra minutes for a cache
that repeats itself is a fair trade.

## 2026-10-19 — Day 15: Family V

20 runs, 2.5 hours. The same MLP head on ten frozen backbones' features.

The answer to "which pretraining wins" is not the one I'd have guessed. The
top two, VideoMAE (61.7 / 67.2) and DINOv2 (61.4 / 64.2), were both
pretrained *without labels* first. DINOv2 is an image model: it sees each
frame alone and has never seen motion, and it ties the best video model on
author. The 3D CNNs, built to model motion, are at the bottom (41–50).
CLIP's captions don't help much either (46.5 / 49.7).

So for cricket shots, a strong general visual representation matters more
than motion modelling inside a frozen backbone. One caveat: DINOv2's MLP
sees all 15 frames in order; a video model has pooled time away already.

Checks: V01 beats P03 (an MLP should never lose to a Linear on the same
features), by 4.2 / 3.3. The probe's leader (VideoMAE) is the MLP's leader
too, so the cheap probe was a fair guide.

Frozen features top out around 67–69%: best V = best A within noise. B01
(fine-tuned end to end) is still ten points clear. And again every V model
does better on grouped val than author val; B01 is the only run that
doesn't. Saving that for Day 16.

MViT: tiny features (max 1.2), small gradients, slowest start. It
converged, but a shared learning rate may undersell it. Noted, not chased.

## 2026-10-19 — Day 16: the cheap half, reviewed

Best by grouped val: P04 56.98, A09 70.11, D01 45.70, V07 67.18, and B01
77.69 still well ahead of everything frozen.

The big one is section 6's question. I expected the author split to flatter
models (it shares match videos with training). Over 35 IDs it does the
opposite: grouped val is higher for 32 of them, by 2.8 points on average.

Why: the author split keeps a clip's own match out of training only for the
*same* shot, so its own match is in train with *other* shots. Same ground,
same light, same broadcast, different answer. For a model that reads the
scene, that's a trap. I tested it with k-NN on the very same author val
clips: 18.50% with all train clips, 24.75% with each clip's own match left
out. For 43% of clips the nearest train clip came from their own match, and
it was a different shot every time. Without those, author scores about
the same as grouped (23.07%).

So the gap is the data's doing, and the model decides how much it costs:
~3 points for frozen features, nothing measurable for fine-tuned B01. Which
also means section 6's "89.09% is likely a little optimistic" doesn't hold
for these models. I've added a dated note there rather than rewriting it.
The test is now tools/same_match.py.

Cost check: A, D, V took ~28 GPU-hours against ~37 estimated. The recurrent
heads on the flat view were the slow ones (up to 116 min a run); everything
else was quicker than planned.

## 2026-10-19 — The stretch families join the plan

E (pose and graph), W (bat, ball, flow, frame differences) and F
(multi-scale and hybrid) are now Days 28–37, after Family C and before
Phase 4 chooses anything, so they can compete for the top 3. 14 IDs, 28
runs, an estimated 45–80 GPU-hours. Phase 4 moves to Days 38–42.

One thing fixed while writing it up: Appendix A said F01 was "multi-scale
CNN (10/15/20 frames)" and also "your existing multiscale_model.py", but
that file is multi-scale in *space* (three EfficientNetV2-S stages joined),
not in time. F01 is now the model the file describes; the 10/15/20-frame
idea already lives in Tier 2 #5 and Phase 4's frames axis.

## 2026-10-19 — Day 17 built: timm backbones and a memory probe

B02 and B03 aren't in torchvision, so the frame encoder now asks timm for
any name torchvision doesn't know. torchvision still wins a shared name
(resnet50), so nothing earlier changes. B02 = tf_efficientnetv2_b3.in1k,
B03 = legacy_xception.tf_in1k, the original Keras Xception.

Caught before it cost a run: Xception was trained on inputs scaled to
mean/std 0.5, not ImageNet's. Fed ImageNet-normalised frames, fine-tuning
would have spent its first epochs undoing the mismatch, and B03 would have
lost to the others for a reason that has nothing to do with Xception. The
encoder now converts, with one multiply-add derived from the name.

Swin turned out not to need anything: torchvision already permutes it to
channels first before the pool.

RegNetY-16GF's grid flattens to 148,176 numbers, so its GRU kernel alone is
57M parameters. If anything doesn't fit, it's that one; the memory probe
will say.

## 2026-10-19 — 17.2 measured: everything fits except, really, B07

The probe said all eight fit. Seven do, with room to spare (2–6 GB of
11.6). B07, RegNetY-16GF, was reported as fitting at 9.4 GB but ran at
8–28 seconds a batch, wildly variable, where the others take a third of a
second. Halving the batch gave 0.17 s a clip against 7.0: forty times
faster per clip. That's not a slow kernel, that's the card running out of
memory and Windows spilling it to system RAM. So B07 trains at batch 2 ×
accum 2, the plan's own fallback. The head's BatchNorm now sees 2 clips at
a time instead of 4, which is the one real difference, and it's recorded.

The probe called it a fit because I checked peak allocated memory under
90%. Reserved memory is what the card actually holds, and 87% plus the
desktop was too much. Now: reserved under 80%.

Also found: deterministic mode (11.1) costs about 1.6x on B01 in the probe.
B01 ran before that mode existed; B02 onwards run with it. It doesn't
change what's learned, only which kernels run, so I'm keeping it, and B02's
real run will show what it costs over a whole epoch.

## 2026-10-20 — 17.3: B02 alone, 77.02%, and what Family B will cost

B02 (EfficientNetV2-B3) author: 77.02% val, F1 0.747, against B01's 79.23.
A power cut hit at epoch 23 and the resume picked it up cleanly. Epoch 23
looked bad (val 68.3, train loss up) but that trend had started two epochs
before the cut; the drop to lr 1e-5 after epoch 24 fixed it, and the best
came at epoch 38.

Cost: 9.1 min an epoch, 7.3 hours in all, where B01 took 4.4 and 3.3. The
probe predicted B02's speed almost exactly, so it's B01's old number that's
the odd one: it ran before deterministic mode existed. Re-costing the rest
of Family B from the probe: ~95 GPU-hours for the 13 runs left, B07 alone
~29 of them. The plan said 50–85.

Two things noticed while checking the resume: the data loader's shuffle
state isn't saved, so after a cut the shuffle order restarts from the seed.
Harmless for learning, but a resumed run isn't bit-identical to an
unbroken one. And the open question: keep deterministic mode for end-to-end
runs at roughly 1.6x the time, or not.

Deterministic kernels: measured properly this time, 60 batches each way,
B02 0.21 vs 0.31 s a batch, B01 0.23 vs 0.40. Off would save a third to
two-fifths of every end-to-end run, ~35 GPU-hours on Family B alone. I
recommended switching them off; the user's answer was that time isn't the
problem, and with time off the table they're worth keeping: every run
repeats exactly, and with the resume fix a power cut leaves no trace in the
result. Neither choice changes accuracy. So they stay on everywhere, and
Family B is ~95 GPU-hours. `deterministic` became a config key anyway, so
the choice is visible in every config, and it doesn't count as a change in
what a run means: it picks kernels, not the experiment.

And the resume gap is closed: the shuffle state is saved now. The test cuts
a run in epoch 2, resumes it, and checks its losses equal an unbroken run's
exactly, then deletes the saved state and checks they no longer do. That
second half matters: it proves the test can see the bug at all.

While in there: set_seed(deterministic=False) used to leave the mode as
the last run left it, so a run asking for "off" after one that turned it on
would quietly have run deterministic anyway. Fixed, tested, for the day
anyone does switch it off.

## 2026-10-21 — The authors' notebooks were here all along

Asked to find their GitHub: it's github.com/Dihan69/CricShot10k, and
CricShot10kAuthorsCode/ is a clone of it, up to date. In Phase 1 I wrote
"neither archive contains the notebook" -- true of the zips, but the clone
has Codes/ with the cropping, segmentation and training notebooks. I never
looked there. That's the biggest miss of the project so far.

What their code actually does, against what the paper says and what we built:

- The crop box is recomputed for every frame (striker ∪ bat), stretched to
  224x224 without keeping the aspect ratio, padded 10% sideways only, and
  frames without a striker are dropped. Ours: one smoothed square box per
  clip, 25% padding all round, gaps interpolated.
- The overlay is ultralytics' results.plot(): striker blue (4, 42, 255),
  bat CYAN (11, 219, 235). The paper says green; we painted (0, 255, 0).
- Early stopping and the lr schedule watch val loss, not accuracy.
- Player_Type_Detection_Model.pt is a byte copy of the segmentation model,
  so the striker detector their cropping used was never released.

None of the crop differences were in Day 7's sweep, so "every cause
eliminated" was true only of the causes we thought of. The ruler (their
model on our frames, 43%) can now test the real recipe on a few hundred
clips. The training notebook itself is plain and inflates nothing.

Built the replica (7.6): tools/replica.py makes 300 author-val clips the
way their notebooks do, in two versions, crop only and crop plus their
ultralytics overlay. Looking at one Hook side by side with ours made the
difference obvious: their batter fills the whole frame, slightly widened by
the stretch; ours stands small in the middle of the pitch. And their bat is
cyan. The only part I can't copy is their striker detector, so the striker
box comes from our own tracker.

A small trap on the way: on the CPU, the 896 px detector segfaulted on
batches of four frames while a training run held most of the RAM (4.3 GB
free). One frame at a time works, 40 s a clip. On the GPU it will be fast,
so the plan is to pause Family B for half an hour, build and score on the
GPU, and resume; the resume is exact now, so the pause costs nothing but
time. The user also wants our own best models scored on the replica, so
`--score` runs B01 and B03 on box, replica_crop and replica.

## 2026-10-21 — B04 grouped killed by a GPU timeout, not a bug

B03 grouped 80.82, B04 (ConvNeXt-T) author 80.24: the two best rows of
Family B so far. Then B04 grouped died in epoch 1 with "Segmentation fault".
Windows' application log has the cause: LiveKernelEvent 141, a GPU engine
timeout that the driver recovers from by resetting the engine. The log
shows dozens of them since 14 October, several a day, mostly while runs
carried on fine; this one took the training process with it. Driver
32.0.101.9034 (30 Sep). Cost: six minutes, since epoch 1 had not saved.
If it keeps happening: a newer Arc driver, or a longer Windows GPU timeout
(TdrDelay), the usual fix for long compute jobs on a card that also drives
the screen. It may also be part of the story behind Day 14's DINOv2 NaNs,
though the SDPA test reproduced those with no timeout in sight.

## 2026-10-21 — Porting their notebook, properly this time (7.7)

The user wants their method converted to PyTorch from their actual code and
trained on frames made their way, as Days 1–7 did from the .keras file. Most
of mimic_author already matched (GRU init, BatchNorm settings), but reading
the notebook turned up four things it got wrong or couldn't have known:

- Their backbone is Keras' EfficientNetV2S: Google's own TF checkpoint,
  expecting colours in -1..1. timm ships exactly that as
  tf_efficientnetv2_s.in1k; torchvision's efficientnet_v2_s is a different
  set of weights. The Day 17 encoder already converts colours by itself.
- Early stopping and lr drops watch val loss, keeping the lowest-loss
  weights. New config key `monitor`.
- An off-by-one I made in Phase 1: Keras' ReduceLROnPlateau(patience=4)
  drops the rate after 4 bad epochs, PyTorch's after 5. mimic_author has
  been dropping it one epoch late all along. The port uses 3.
- Adam's epsilon (1e-7) and Glorot init on the dense layers: small, but the
  port is meant to be exact. Both opt-in keys, so no old config changes.

Family R is a 2x2 with B01 (protocol x frames), so whatever happens can be
pinned on a cause. R needs the replica for all 9,880 clips: 20.8 GB, which
fits in the 34 GB free.

Two bugs caught while building it, both mine, both before anything ran:
a variable called `watch` that the epoch printout also uses, which would
have quietly turned a loss-watching run back into an accuracy-watching one
after epoch 1; and a readiness check that only asked whether the replica
folder existed. The 300-clip sample already in it made R03 look ready, and
the dataset skips missing clips without complaint, so R03 would have trained
on 300 clips. The check now counts clips against box's 6,320. Its first
version counted against the split's 6,466 and declared all of Family B not
ready; box is the yardstick, because 146 clips never survived Day 4.

## 2026-10-21 — The replica test: closer, not there

Their own model, same 300 val clips: our box 36.3, our seg 44.8, their crop
50.0, their crop and colours 51.0. So the per-frame stretched crop is real
and worth ~6 points to their model over our best guess, and the cyan bat is
worth nothing. But 51% is not 89%, on clips it probably partly trained on.
Whatever their frames look like, the replica isn't it yet.

The biggest piece I couldn't copy is their striker detector, and in their
code the striker box is the crop. Also missing: the 15-frame sampling script
(one README line) and the meaning of their folder name, 15fs3it. The honest
next step for those is an email to the authors, not more guessing.

Our models: B01 75.9 / 61.7 / 42.7 and B03 76.6 / 63.0 / 51.0 on box /
replica crop / replica. Expected drops on unfamiliar frames; Xception the
steadier. Training on the replica (R03) is the real test of the recipe.

## 2026-10-22 — B07 at batch 2 was broken, and it was my call

B04 grouped 81.52 (best in Family B on both splits now), B05 author 75.00.
Then B07, RegNetY-16GF, at batch 2 x accum 2: 29% val after epoch 1,
32% after 2, then 21, 16, 22, 22, with train accuracy falling too and
gradient norms of 40 where the others sit at 10-20.

The cause is simple once seen: the head ends GRU -> BatchNorm -> Dense,
and BatchNorm in training normalises each feature over the batch. Over two
clips, each of the two values sits exactly one standard deviation from
their mean, so every feature comes out +1 or -1. The head trains on sign
bits relative to whichever clip it happened to be paired with. Accumulation
doesn't help; BatchNorm still sees two. In 17.2 I wrote that two-at-a-time
BatchNorm was "the one real difference" and moved on. It was fatal.

Fix: batch 4 like every other B run, with the memory found elsewhere.
grad_checkpoint keeps only the boundaries between RegNet's four stages for
backward and recomputes inside them. The recompute would update BatchNorm's
running stats twice, so those layers get momentum 1 - sqrt(0.9), which makes
two updates exactly one. Tested equal to the plain backbone: outputs,
gradients, running stats, weight names (the last one caught a bug: wrapping
the stages renamed block1..block4 to 0..3, and no saved weights would have
loaded).

Also: the half-finished batch-2 run left a resume file, and the runner would
have continued it at batch 4 as if nothing had changed. Training now refuses
a resume file whose config differs and says to use --force.

## 2026-10-24 — Family B, seven of eight: the backbone matters less than hoped

B07 at batch 4 with checkpointing came back healthy: 58, 66, 71% val over
its first three epochs, where batch 2 had gone 29, 32, 21. It finished
77.90 author and 79.67 grouped, after 23 GPU-hours (the grouped half
resumed across a power cut).

With B05, B06 and B07 done, the picture is clear enough:

- ConvNeXt-T (B04) is best on both splits, 80.24 / 81.52, with Xception
  (B03) level with it. The frozen probe gave ConvNeXt an 8-9 point lead
  over EfficientNet; fine-tuning shrinks that to 1-4.
- Size buys almost nothing. RegNetY-16GF is 83M parameters and lands
  between B01 and B06. MobileNetV3 at 5.4M is last, but only 2.4 points
  behind it on average.
- Everything but B01 sits at 97-99% train accuracy at its best val epoch.
  That's a 17-24 point gap with no dropout anywhere. The next points are
  in regularisation (Phase 4), not in a bigger backbone.
- Grouped beats author for six of seven, which fits Day 16: the author
  val split is the harder one.

B08 (Swin-T) is the last of the family.

## 2026-10-24 — Family B done; a hidden regulariser, and a hole in the port

B08 (Swin-T) finished at 78.41 author and 79.54 grouped: third, behind
ConvNeXt-T and Xception. A vision transformer per frame is no better than
a good CNN of the same size.

Swin did one odd thing: 87-90% train accuracy at its best epoch, where
B03-B07 all sit at 97-99. Asking why turned up something I hadn't
looked at. torchvision builds stochastic depth into some backbones:
during training, residual blocks are skipped at random, at a rate that
rises through the network. EfficientNetV2-S (B01) and Swin-T (B08) get up
to 0.2, ConvNeXt-T 0.1, the rest none. So "only the backbone changes" in
Family B wasn't quite true: B01 and B08 carried an extra regulariser. It
didn't win them val accuracy (both mid-table), and B02 overfits less
without it, so it's a partial explanation at best. Phase 4 can test it
directly.

The bigger consequence was for the port. The authors' Keras EfficientNetV2S
has the same thing (drop_connect_rate 0.2); their .keras file holds 34
Dropout layers from 0.005 to 0.195. The timm model I ported to has none
by default. So the "exact" port had less regularisation than their
original, and less than B01. New config key drop_path, timm only, and the
port sets 0.2. The 34 rates now match their file exactly, checked block by
block and pinned in a test. Caught before any R run.

Also fixed: the 7.7 heading in the plan had been glued onto the end of
7.6's last line by an earlier edit of mine.

## 2026-10-25 — The replica is built, and it has more clips than box

tools.replica --build --all finished: 10,089 of 10,091 clips, about six
hours. The two failures had no striker in any frame, and box never made
them either.

Checking coverage before launching R turned up the opposite problem: the
replica has clips box doesn't. 144 train, 30 val and 35 test on the author
split. The authors' detector (their segmentation model doing a detector's
job) finds a batter in about 2% of the clips our Day 4 pipeline gave up on.

That would have quietly broken family R's whole point. R01 and R03 would
have been scored on a different val and test set from B01, so any
difference would have mixed "their frames" with "a few more, probably
harder, clips". New dataset option same_clips_as: keep only the clips
another variant also has. R01 and R03 set it to box, and all three R runs
now see exactly B01's clips. Tested, and counted on the real data.

Later the same day: family R switched to non-deterministic kernels, for
speed (~1.6x). Fair for the comparison that matters, since B01 itself ran
before deterministic mode existed. Everything else in Phase 3 stays
deterministic.

## 2026-10-27 — Family R done: their method is ours, their frames don't help

All four cells of the 2x2 are in (val, author / grouped):

    B01  our protocol,   our crops     79.23 / 77.69
    R02  their protocol, our crops     79.17 / 78.65
    R03  our protocol,   their frames  75.44 / 78.14
    R01  their protocol, their frames  76.77 / 77.44

Their training recipe, ported cell by cell from the notebook, ties ours.
Their preprocessing, copied as closely as their code allows, is a little
worse: clearly on the author split, a wash on grouped. The full
reproduction lands at 77%, nowhere near 89.

So from everything the authors released, the 89% does not come back. What
they didn't release is the striker detector (and in their code the striker
box is the crop), the frame-sampling script, and their clips and split.
That is where the gap has to be, and it's what the email asks for.

R03 ran 79 and 78 epochs, twice the others, because B01's early stop
counts a one-clip improvement as a gain. With the lr at 1e-7, BatchNorm's
running stats still nudge val accuracy around, and every lucky nudge reset
the countdown. Left alone here so R03 stays B01's protocol; it goes on
Phase 4's list.

## 2026-10-27 — Day 23: the video3d method

Family C needed one new piece: a way to train a whole video model on our
clips. VideoClassifier wraps any of the eight. It takes the model as
published, swaps only the last Linear (400 Kinetics classes to our 15),
and converts our clips to what the weights were trained on: frame count,
size, colours.

The conversion is the part I tested hardest. Family V's frozen features
were made by extract.py on the CPU; C does the same on the GPU, in the
other order (recolour, then resize). The two are equal in maths because
both steps are linear, and the test checks they are equal in numbers too,
to 1e-4. Without that, C02 against V01 would not be "fine-tuned against
frozen" alone.

Smoke test on the Arc with the real weights: one training step each for
the five models that can run today, all finite. MViT turned out to have
one operation (max_pool3d's backward) with no deterministic version, so
its runs won't repeat exactly. Noted, not fixable from here.

pytorchvideo (I3D, SlowFast, X3D) isn't installed; the code is written and
waits for the install and its tests. And the batch-2 fallback that broke
B07 is safe for the heavy C models: they use LayerNorm, so two clips twice
is exactly four clips once.

Later: deterministic kernels are now off by default, for Family C and
everything after it. C goes from ~5-7 days to ~3-4. Finished runs are
untouched: their rows say how they ran, and the runner still treats them
as done.

## 2026-10-27 — Family C probed; VideoMAE trains two at a time

pytorchvideo installed after all (0.1.5, torch untouched), so all eight C
models are in. The probe: seven fit with room to spare, the heaviest
(Swin3D) at 55% of the card. VideoMAE at batch 4 took the GPU down. The
Arc said OUT_OF_RESOURCES rather than "out of memory", so the probe
didn't recognise it as the out-of-memory case it was built to catch;
it does now, and stops cleanly when the device is lost.

Batch 2 fits at 6.7 GB. C08 trains at batch 2 x accumulation 2, which is
exactly batch 4 for VideoMAE: it has no BatchNorm, only LayerNorm, so
nothing compares one clip with another inside a batch. That is the
difference from B07, where BatchNorm over two clips broke the run.

Also found: my shell had been running tests with the system Python, not
the project's .venv. Same torch in both, so no result changes, and the
full suite passes in .venv. Tests run there from now on.

Family C re-costed from the probe: ~90 GPU-hours, about four days.

## 2026-10-29 — Day 44 added: a shot-classification agent

Added to the plan after the supervisor package: a small LangGraph agent
that crops the batter, refuses unreadable clips, answers easy clips with
one fast model, and sends hard ones to several different experts whose
weighted vote it explains. It waits for Phase 4, since the models, vote
weights and confidence threshold all come from val. Same rule as always:
if it goes into CricketMind, the accuracy numbers stay in the CricShot
paper.

Later: decided the whole Day 44 agent becomes CricketMind's classifier
node, rather than one exported model. Its crop and quality check travel
inside it, so CricketMind can't rebuild them slightly wrong, and its
expert route replaces Phase 8's blind "retry up to 2 times". The note for
the CricketMind chat is notes/to_cricketmind_phase8.md. It carries no
accuracy numbers, by the usual rule.

## 2026-10-29 — Day 28, pose: everything but the detector

While Family C finishes (MViT is now best at 83.40 / 84.51), the pose side
of Family E is built and tested on made-up skeletons: normalising, exact
mirroring, the resumable cache, a pose view for the feature dataset, the
ST-GCN head, and E01/E02 in the registry.

Two choices worth remembering. Poses are centred per clip, not per frame,
because per frame would erase a step forward or back inside the crop, and
footwork is half of a stroke. And a frame where no pose was found stays all
zeros with visibility zero, rather than being copied from its neighbours,
so the models can tell "not seen" from "standing still".

The detector waits for mediapipe, which gets a dry-run install first: it
has pinned old numpy and protobuf in the past, and this venv runs numpy 2.
