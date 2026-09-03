# CricShot10k — Methodology Breakdown

> **Paper:** CricShot10k: A Large-Scale Video Dataset for Cricket Shot Classification  
> **Authors:** Mubtasim Kamal Dihan, Abdullah, Amina, Sabbir Ahmed  
> **Institution:** Islamic University of Technology (IUT), Gazipur, Bangladesh  
> **Published:** IEEE Access, February 2026  
> **DOI:** 10.1109/ACCESS.2026.3663220

---

## Overview

The methodology consists of two major parts:
1. **Dataset Construction** — How CricShot10k was built
2. **CricShotNet** — The proposed classification model

---

## Part 1: Dataset Construction

### Step 1 — Match Video Collection

- Collected **536 full-length cricket highlight videos** from online sources
- Video durations: **5 to 36 minutes** (average ~12 minutes)
- Total raw footage: **~107 hours**
- Coverage:
  - Match types: International Friendly, International Tournament, Franchise League
  - Formats: Test, 50-Over (ODI), T20
  - Players: Senior Men (73%), Senior Women (20%), Under-19 Men (7%)
  - Era: 1996 (Cricket World Cup) to present

---

### Step 2 — Automated Shot Splitting

This is the **core novelty** of the dataset construction pipeline.

#### Problem with Previous Approaches
- Manual trimming of match videos into individual shot clips is extremely time-consuming
- Led to small datasets (KUCricShot: 1,278 videos; CricShotClassify: 1,867 videos)

#### Proposed Solution — YOLOv11-Based Detection Pipeline

Three separate **YOLOv11m** object detection models were trained:

| Model | Purpose | Training Images |
|---|---|---|
| Player Detection | Detect striker, bowler, batter, fielder, umpire, keeper | 602 images |
| Ball Detection | Detect the cricket ball | 1,400 images |
| Bat Detection | Detect the bat (used later in CricShotNet) | 750 images |

Annotation was done using the **CVAT online platform**. Each model used **80% training / 20% testing** split.

#### Sliding Window Logic

A **10-frame sliding window** runs over each video frame-by-frame. A frame is considered the **starting point of a shot** only if ALL three conditions are satisfied simultaneously within the window:

```
Condition 1: Bowler appears in at least 1 of the FIRST 5 frames
Condition 2: Batter appears in at least 5 of the 10 frames
Condition 3: Ball appears in at least 5 of the 10 frames
```

**Why these specific thresholds?**
- After ball release, camera zooms in on the striker → bowler exits frame → only check bowler in first 5 frames
- Ball is small and can briefly disappear → 5/10 threshold balances false positives and missed shots
- Batter must be consistently present → 5/10 threshold

**Once a valid starting frame is found:**
- Extract exactly **1 second** of video from that frame
- Reset the sliding window
- Search for the next shot starting from the next frame

This produced **14,369 raw one-second clips** from 107 hours of footage.

#### Why 1-Second Duration?
- Chosen after extensive testing across diverse matches
- Long enough to capture full shot execution
- Short enough to avoid irrelevant segments
- Makes annotation similar to text/image labeling tasks

---

### Step 3 — Shot Data Annotation

#### Annotator Setup
- **3 expert annotators** with extensive cricket knowledge
- Inspired by collaborative crowdsourcing approach (Chang et al.)
- Each annotator labeled videos independently

#### Label Categories
Each video was assigned one of **18 labels**:

| Category | Labels |
|---|---|
| Shot Classes (15) | Cover Drive, Defensive, Down The Wicket, Flick, Hook, Late Cut, Lofted Legside, Lofted Offside, Pull, Reverse Sweep, Scoop, Square Cut, Straight Drive, Sweep, Upper Cut |
| Special Labels (3) | Uncertain, Uncategorized, Not a Shot |

#### Consensus Rules
```
≥ 2/3 annotators agree     → Assigned that label
All 3 annotators conflict   → Group discussion
Still unresolved after discussion → Marked "Uncategorized"
```

#### Annotation Quality — Fleiss' Kappa
- **Kappa Score: 0.9376** → Falls in "Perfect Agreement" range (0.80–1.00)
- All 3 annotators agreed on **9,218 out of 10,086** videos
- Only **868 videos** showed any disagreement

#### Annotation Pipeline Summary

| Phase | Count |
|---|---|
| Total clips after automated splitting | 14,369 |
| Videos labeled same by ≥ 2 annotators | 9,420 |
| Videos requiring discussion | 1,420 |
| Videos labeled after discussion | 666 |
| Videos remained uncategorized | 754 |
| Total uncategorized after all processes | 1,840 |
| Videos labeled as "Not a Shot" | 2,443 |
| **Total shots used in dataset** | **10,086** |

---

### Step 4 — Dataset Analysis

#### Class Distribution

| No | Class Label | Count | Percentage |
|---|---|---|---|
| 1 | Cover Drive | 793 | 7.85% |
| 2 | Defensive | 587 | 5.81% |
| 3 | Down The Wicket | 844 | 8.36% |
| 4 | Flick | 947 | 9.38% |
| 5 | Hook | 532 | 5.27% |
| 6 | Late Cut | 325 | 3.22% |
| 7 | Lofted Legside | 1094 | 10.83% |
| 8 | Lofted Offside | 1054 | 10.44% |
| 9 | Pull | 769 | 7.62% |
| 10 | Reverse Sweep | 257 | 2.55% |
| 11 | Scoop | 279 | 2.76% |
| 12 | Square Cut | 1000 | 9.89% |
| 13 | Straight Drive | 419 | 4.15% |
| 14 | Sweep | 905 | 8.97% |
| 15 | Upper Cut | 281 | 2.78% |
| | **Total** | **10,086** | **100%** |

- **Class imbalance ratio:** 4.26 (largest/smallest class) → considered minimal
- Imbalance reflects **natural cricket gameplay**, not sampling bias

#### Key Dataset Challenges
1. **Inter-class Similarity** — Cover Drive vs Straight Drive look almost identical in early/late frames; Pull vs Hook are visually very similar
2. **Intra-class Variation** — Same shot type can look very different depending on player stance, camera angle, and execution style
3. **Environmental Factors** — Camera quality, time of day, ground conditions, jersey colors vary significantly

---

### Step 5 — Data Augmentation

- **Technique used:** Horizontal Flipping only
- **Reason:** Simulates left-handed vs right-handed batters realistically
- **When applied:** AFTER train/val/test split (to avoid data leakage)
- Flipped samples are **NOT counted** in the 10,086 total
- Other augmentations (rotation, color jitter) were avoided as they don't produce realistic cricket variations

---

## Part 2: CricShotNet — Classification Model

### Architecture Overview

```
Input Video (720×1280, variable frames)
        ↓
[Layer 2]  Cropping Layer
        ↓
[Layer 3]  Reframing & Resizing (15 frames × 224×224)
        ↓
[Layer 4]  Segmentation Layer
        ↓
[Layer 5]  EfficientNetV2-S (CNN Feature Extractor)
        ↓
[Layer 6]  Flattening Layer
        ↓
[Layer 7]  GRU (128 units) — Temporal Feature Extractor
        ↓
[Layer 8]  Batch Normalization
        ↓
[Layer 9]  Dense Layer (1024 units, ReLU)
        ↓
[Layer 10] Softmax (15 classes)
        ↓
Predicted Shot Label
```

---

### Layer 2 — Cropping Layer

#### Purpose
Remove irrelevant background elements (crowd, pitch markings, fielders) from each frame.

#### How It Works
- Run **YOLOv11 striker detection** on each frame
- Run **YOLOv11 bat detection** on each frame (trained on 750 images)
- Combine both bounding boxes into a single crop region
- If **striker not detected** → discard that frame entirely
- If **bat not detected** → use only striker bounding box

#### Effect
```
Before cropping: Average frames per video = 26.12
After cropping:  Average frames per video = 24.77
Minimum frames:  Dropped from 22 → 17 (still above 15 threshold)
```

All cropped frames resized to **224 × 224 pixels**.

---

### Layer 3 — Reframing & Resizing

- **Uniformly sample 15 frames** from the cropped video
- Chosen after extensive testing with varying frame counts
- Ensures GRU compatibility across all videos
- No video in the dataset fell below 15 frames after cropping

---

### Layer 4 — Segmentation Layer

#### Purpose
Reduce noise from jersey colors and lighting variations across different match formats.

#### How It Works
- Fine-tuned **YOLOv11x-seg** on **674 manually annotated images**
- Two classes annotated: **bat** and **striker**
- After segmentation, predicted masks overlaid with semi-transparent colors:
  - 🔵 **Blue (alpha=0.5)** → Striker
  - 🟢 **Green (alpha=0.5)** → Bat
- Striker colored first → bat colored on top (so bat remains visible even when overlapping striker)

#### Why Semi-Transparent (not solid)?
- Solid color would hide body posture information
- Semi-transparent preserves posture + contextual cues while suppressing jersey color noise

#### Why Not Use Mask-Only or Attention?
- Mask-only removes contextual cues like player posture and surrounding motion
- Attention mechanisms may overlook the bat when it blends with the background
- Segmentation + coloring explicitly highlights key regions while preserving context

---

### Layer 5 — EfficientNetV2-S (Deep Feature Extractor)

#### Selection Process
- Compared **15 different CNN architectures** on CricShot10k
- All other layers kept fixed during comparison

#### Performance Comparison (Top Results)

| Model | Params (M) | Top-1 (%) | Top-2 (%) | Top-3 (%) | F1-Score |
|---|---|---|---|---|---|
| **EfficientNetV2-S** | 20.33 | **89.09** | 97.22 | **99.25** | **0.8847** |
| Xception | 20.86 | 88.64 | 97.17 | 98.51 | 0.8655 |
| EfficientNetV2-B3 | 12.93 | 88.39 | **97.32** | 98.81 | 0.8727 |
| ConvNeXt-Tiny | 27.82 | 87.20 | 97.22 | 99.15 | 0.8596 |
| MobileNetV2 | 2.26 | 85.02 | 95.48 | 97.66 | 0.8432 |
| InceptionV3 | 21.80 | 68.84 | 83.13 | 91.26 | 0.6183 |

#### Configuration
- Pre-trained on **ImageNet**
- Classifier (top) block **removed**
- Wrapped in **TimeDistributed layer** → processes each of the 15 frames independently
- Input size: **224 × 224 × 3**
- Output: Feature vector per frame → passed to GRU

---

### Layer 7 — GRU (Temporal Feature Extractor)

#### Purpose
Capture temporal relationships between the 15 frame feature vectors (i.e., how the shot evolves over time).

#### Selection Process
Tested **9 RNN variants** with different hidden unit sizes:

| Model | Units | Params (M) | Top-1 Acc (%) |
|---|---|---|---|
| LSTM | 64 | 16.07 | 85.42 |
| LSTM | 128 | 32.17 | 84.97 |
| LSTM | 256 | 64.48 | 83.76 |
| GRU | 64 | 12.05 | 85.52 |
| **GRU** | **128** | **24.13** | **89.09** |
| GRU | 256 | 48.36 | 86.06 |
| BiLSTM | 32+32 | 16.06 | 83.38 |
| BiLSTM | 64+64 | 32.15 | 87.65 |
| BiLSTM | 128+128 | 64.36 | 85.32 |

#### Why GRU over LSTM/BiLSTM?
- GRU has **fewer gates** → more computationally efficient, less prone to overfitting
- **Forward bat movement** is most critical for shot classification → BiLSTM's backward pass adds little value
- 256 units **reduced accuracy** → overfitting on this dataset size

---

### Layers 8-10 — Classifier Network

```
GRU Output (128-dim)
    → Batch Normalization   (stabilizes training, reduces internal covariate shift)
    → Dense Layer (1024 units, ReLU activation)
    → Softmax Layer (15 units)
    → argmax → Predicted Class Label
```

---

### Ablation Study Results

Systematic evaluation of each novel component:

| Cropping | Segmentation | Augmentation | Accuracy (%) |
|---|---|---|---|
| ✗ | ✗ | ✗ | 72.03 |
| ✗ | ✗ | ✓ | 75.50 |
| ✗ | ✓ | ✗ | 79.45 |
| ✗ | ✓ | ✓ | 80.20 |
| ✓ | ✗ | ✗ | 83.39 |
| ✓ | ✗ | ✓ | 85.76 |
| ✓ | ✓ | ✗ | 87.60 |
| ✓ | ✓ | ✓ | **89.09** |

**Key Takeaways:**
- Every component contributes positively
- Cropping alone gives the biggest single boost
- Segmentation adds ~4% on top of cropping
- Augmentation adds ~1.5% consistently

---

## Experimental Setup

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB GDDR6X) |
| Video Reader | Decord (hardware-accelerated) |
| Batch Size | 8 |
| Max Epochs | 50 |
| Early Stopping Patience | 10 |
| Convergence | ~30 epochs for all runs |
| Optimizer | Adam |
| Initial Learning Rate | 10⁻⁴ |
| LR Reduction Factor | 0.1 (after 4 patient epochs) |
| Train / Val / Test Split | 64% / 16% / 20% |
| Splitting Strategy | Stratified (same batter shots don't overlap across sets) |
| CNN Weight Init | ImageNet pretrained |
| GRU Weight Init | Glorot uniform (input→hidden), Orthogonal (recurrent) |

---

## Final Results

### CricShot10k Performance

| Metric | Score |
|---|---|
| Top-1 Accuracy | **89.09%** |
| Top-2 Accuracy | **97.22%** |
| Top-3 Accuracy | **99.25%** |
| Precision | 0.8846 |
| Recall | 0.8881 |
| F1-Score | 0.8847 |
| AUC-ROC | 0.9956 |

### Cross-Dataset Performance

| Dataset | Without Fine-Tuning | With Fine-Tuning |
|---|---|---|
| KUCricShot | 97% | ~99% |
| CricShotClassify | 86% | 97% |

### Comparison With Prior Work

| Dataset | Classes | Size | Authors' Accuracy | CricShotNet Accuracy |
|---|---|---|---|---|
| KUCricShot | 4 | 1,278 | 73% (LRCN) | **98.86%** |
| CricShotClassify | 10 | 1,867 | 93% (VGG16-GRU) | **97.10%** |
| **CricShot10k (Ours)** | **15** | **10,086** | — | **89.09%** |

---

## Research Achievements Summary

| Metric | Value |
|---|---|
| Raw footage processed | 107 hours |
| Final curated dataset | 2.8 hours (10,086 clips) |
| Manual effort saved | ~500 hours |
| Annotation time per clip | Similar to text/image annotation |
| Largest cricket shot dataset | ✅ Yes (as of Feb 2026) |
| First to include women's & U19 | ✅ Yes |
| Oldest matches included | 1996 Cricket World Cup |

---

## Future Work Suggested by Authors

- Add new shot classes: **leave, leg glance, backfoot punch**
- Subdivide existing classes (e.g., defensive → front-foot vs back-foot)
- Explore **Video Swin Transformer** and **TimeSformer**
- Advanced augmentation: **temporal jittering, color perturbation**
- Domain-specific architectures: **bat motion tracking**
- Extend to detect other cricket events for **automated commentary generation**

---

*Generated from: CricShot10k paper (IEEE Access, Vol. 14, 2026)*
