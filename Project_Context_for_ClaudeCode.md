# Project Context — CricShot10k Research
> **Purpose:** This file summarizes a full research conversation to provide context for Claude Code.  
> **Student:** MIST (Military Institute of Science and Technology), CSE Department, Bangladesh  
> **Goal:** Build a novel research paper on top of the CricShot10k dataset for conference submission

---

## 1. Base Paper Summary

### Paper Title
**CricShot10k: A Large-Scale Video Dataset for Cricket Shot Classification**

### Authors & Affiliation
- Mubtasim Kamal Dihan, Abdullah, Amina, Sabbir Ahmed
- Islamic University of Technology (IUT), Gazipur, Bangladesh
- Contact: mubtasimkamal@iut-dhaka.edu

### Published
- IEEE Access, Volume 14, February 2026
- DOI: 10.1109/ACCESS.2026.3663220
- Citations: Only 1 (very new — high opportunity for follow-up work)

### GitHub Repository
- **URL:** https://github.com/Dihan69/CricShot10k
- **Status:** ✅ Fully PUBLIC
- **Language:** Jupyter Notebook (100%)

### What's Available in the Repo
| Resource | Location | Status |
|---|---|---|
| Source code | GitHub `/Codes` folder | ✅ |
| Full dataset (10,086 videos) | Google Drive | ✅ |
| Demo videos (5 per class) | Google Drive | ✅ |
| All trained YOLO models | Google Drive | ✅ |
| Player detection dataset (~600 images) | Google Drive | ✅ |
| Bat detection dataset (~670 images) | Google Drive | ✅ |
| Ball detection dataset (~1,900 images) | Google Drive | ✅ |
| Segmentation dataset (~650 images) | Google Drive | ✅ |
| Match video list (536 videos) | Google Spreadsheet | ✅ |

### BibTeX Citation
```bibtex
@ARTICLE{11389735,
  author={Dihan, Mubtasim Kamal and Abdullah and Amina and Ahmed, Sabbir},
  journal={IEEE Access},
  title={CricShot10k: A Large-Scale Video Dataset for Cricket Shot Classification},
  year={2026},
  volume={14},
  pages={23428-23447},
  doi={10.1109/ACCESS.2026.3663220}
}
```

---

## 2. Dataset Details

### CricShot10k Overview
- **Total videos:** 10,086 one-second clips
- **Classes:** 15 shot types
- **Source:** 536 full-length cricket highlight videos (~107 hours raw footage)
- **First dataset** to include women's and under-19 matches
- **Oldest matches:** dating back to 1996 Cricket World Cup
- **Formats covered:** Test, ODI, T20
- **Nearly 8x larger** than previous datasets

### Class Distribution
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

- **Class imbalance ratio:** 4.26 (considered minimal)
- **Minority classes:** Reverse Sweep (257), Scoop (279), Upper Cut (281), Late Cut (325)

---

## 3. Their Methodology (CricShotNet)

### Step 1 — Data Collection
- 536 highlight videos collected from YouTube
- Duration: 5–36 minutes each (~12 min average)
- Total: ~107 hours of raw footage

### Step 2 — Automated Shot Splitting
Using fine-tuned **YOLOv11** models:

**3 detection models trained:**
- Player Detection (7 classes: striker, bowler, batter, fielder, umpire, keeper, others) — 602 images
- Ball Detection — 1,400 images
- Bat Detection — 750 images

**10-frame sliding window logic — shot starts only when ALL 3 conditions met:**
```
1. Bowler in at least 1 of first 5 frames
2. Batter in at least 5 of 10 frames
3. Ball in at least 5 of 10 frames
```
→ Extracts exactly **1 second** of video from valid starting frame
→ Produced **14,369 raw clips** initially

### Step 3 — Annotation
- 3 expert annotators with cricket knowledge
- Labels: 15 shot classes + uncertain + uncategorized + not a shot
- Consensus: ≥ 2/3 annotators must agree
- **Fleiss' Kappa: 0.9376** (perfect agreement range: 0.80–1.00)
- 9,218 of 10,086 videos had unanimous agreement

### Step 4 — Augmentation
- **Horizontal flipping only** (simulates left/right handed batters)
- Applied AFTER train/val/test split (avoids data leakage)
- Flipped samples NOT counted in the 10,086 total

---

## 4. CricShotNet Architecture

### Full Pipeline
```
Input Video (720×1280, variable frames)
        ↓
[Layer 2]  Cropping Layer (YOLOv11 striker + bat detection)
        ↓
[Layer 3]  Reframing & Resizing → 15 frames × 224×224
        ↓
[Layer 4]  Segmentation Layer (YOLOv11x-seg)
           Striker → Blue (alpha=0.5)
           Bat → Green (alpha=0.5)
        ↓
[Layer 5]  EfficientNetV2-S (ImageNet pretrained, top removed)
           Wrapped in TimeDistributed layer
        ↓
[Layer 6]  Flattening Layer
        ↓
[Layer 7]  GRU (128 hidden units)
        ↓
[Layer 8]  Batch Normalization
        ↓
[Layer 9]  Dense (1024 units, ReLU)
        ↓
[Layer 10] Softmax (15 classes)
        ↓
Predicted Shot Label
```

### Key Design Choices
- **15 frames** uniformly sampled per video (chosen after extensive testing)
- **EfficientNetV2-S** — best among 15 CNN architectures tested (20.33M params)
- **GRU with 128 units** — best among 9 RNN variants tested
- **Cropping** — removes irrelevant background, focuses on striker+bat region
- **Segmentation** — reduces jersey color noise across formats

### Train/Val/Test Split
- Test: 20% (manually constructed, same batter shots don't overlap)
- Val: 20% of remaining (randomly selected)
- Train: remaining ~64%
- Stratified splitting applied

### Training Setup
| Setting | Value |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB) |
| Batch Size | 8 |
| Max Epochs | 50 |
| Early Stopping Patience | 10 |
| Convergence | ~30 epochs |
| Optimizer | Adam |
| Learning Rate | 10⁻⁴ |
| LR Reduction | ×0.1 after 4 patient epochs |
| Video Reader | Decord (hardware-accelerated) |

---

## 5. Their Results (Baseline to Beat)

### CricShot10k Performance
| Metric | Score |
|---|---|
| **Top-1 Accuracy** | **89.09%** |
| Top-2 Accuracy | 97.22% |
| Top-3 Accuracy | 99.25% |
| Precision | 0.8846 |
| Recall | 0.8881 |
| F1-Score | 0.8847 |
| AUC-ROC | 0.9956 |

### CNN Architecture Comparison (Top 5)
| Model | Params (M) | Top-1 (%) | F1-Score |
|---|---|---|---|
| **EfficientNetV2-S** | 20.33 | **89.09** | **0.8847** |
| Xception | 20.86 | 88.64 | 0.8655 |
| EfficientNetV2-B3 | 12.93 | 88.39 | 0.8727 |
| ConvNeXt-Tiny | 27.82 | 87.20 | 0.8596 |
| MobileNetV2 | 2.26 | 85.02 | 0.8432 |

### RNN Comparison (Best Configs)
| Model | Units | Top-1 (%) |
|---|---|---|
| **GRU** | **128** | **89.09** |
| BiLSTM | 64+64 | 87.65 |
| GRU | 64 | 85.52 |
| LSTM | 64 | 85.42 |

### Ablation Study
| Cropping | Segmentation | Augmentation | Accuracy (%) |
|---|---|---|---|
| ✗ | ✗ | ✗ | 72.03 |
| ✗ | ✗ | ✓ | 75.50 |
| ✗ | ✓ | ✓ | 80.20 |
| ✓ | ✗ | ✓ | 85.76 |
| ✓ | ✓ | ✗ | 87.60 |
| ✓ | ✓ | ✓ | **89.09** |

### Cross-Dataset Performance
| Dataset | Without Fine-Tuning | With Fine-Tuning |
|---|---|---|
| KUCricShot (4 classes, 1278 videos) | 97% | ~99% |
| CricShotClassify (10 classes, 1867 videos) | 86% | 97% |

### Worst Performing Class
- **Late Cut** — F1: 0.7451, Recall: 0.6786
- Reason: Visually similar to Upper Cut and Square Cut

### Best Performing Classes
- **Scoop** — Recall: 1.0000, AUC: 1.0000 (most visually distinct)
- **Reverse Sweep** — AUC: 0.9992

---

## 6. Their Future Work (Explicitly Mentioned)

The authors themselves suggested these directions:
1. Add new shot classes: **leave, leg glance, backfoot punch**
2. Subdivide defensive shots into front-foot and back-foot
3. Explore **Video Swin Transformer** and **TimeSformer**
4. Advanced augmentation: **temporal jittering, color perturbation**
5. Domain-specific architectures: **bat motion tracking**
6. Extend to detect other cricket events for **automated commentary**

---

## 7. Our Proposed Improvement Ideas (10 Total)

### Quick Reference
| # | Idea | Difficulty | Time | Expected Gain |
|---|---|---|---|---|
| 1 | Transformer-Based Video Classification | 🔴 High | 3-4 months | +3-6% |
| 2 | Multi-Task Learning | 🔴 High | 4-5 months | +2-4% |
| 3 | Few-Shot Learning for Rare Shots | 🔴 High | 4-6 months | High novelty |
| 4 | Pose-Guided Shot Classification | 🟡 Medium | 3-4 months | +3-5% |
| 5 | Temporal Attention Mechanism | 🟡 Medium | 2-3 months | +2-4% |
| 6 | Cross-Format Generalization Study | 🟡 Medium | 2-3 months | Analysis paper |
| 7 | Knowledge Distillation / Edge Deployment | 🟡 Medium | 3-4 months | Practical |
| 8 | Explainability Study (XAI / GradCAM) | 🟢 Low | 1-2 months | Add-on |
| 9 | Data Augmentation Study | 🟢 Low | 1-2 months | +1-3% |
| 10 | Ensemble Methods | 🟢 Low | 2-3 months | +1-3% |

---

### Idea 1 — Transformer-Based Video Classification
**Replace:** EfficientNetV2-S + GRU pipeline  
**With:** Modern video transformers  
**Models:** Video Swin Transformer, TimeSformer, VideoMAE, MViT  
**Why novel:** Paper explicitly suggests this as future work; no prior paper tried it on CricShot10k  
**Target:** IEEE Access or ICCIT 2026

---

### Idea 2 — Multi-Task Learning
**Add extra prediction heads:**
- Head 1: Shot type (15 classes) — primary task
- Head 2: Batter handedness (Left/Right)
- Head 3: Match format (Test/ODI/T20)

**Loss:** `Total = α×Loss_shot + β×Loss_handedness + γ×Loss_format`  
**Why novel:** No cricket shot paper uses multi-task learning  
**Target:** IEEE Access or Sensors journal

---

### Idea 3 — Few-Shot Learning for Rare Shots
**Problem:** Reverse Sweep (257), Scoop (279), Upper Cut (281) are underrepresented  
**Approaches:** Prototypical Networks, MAML, VideoGAN synthetic augmentation  
**Why novel:** Few-shot learning for video sports action is an open problem  
**Target:** Pattern Recognition journal or IEEE Access

---

### Idea 4 — Pose-Guided Two-Stream Fusion
**Stream 1:** RGB visual (their CricShotNet pipeline)  
**Stream 2:** MediaPipe body pose keypoints (33 joints)  
**Fusion:** Late fusion / Cross-attention  
**Why novel:** First paper to fuse visual + pose on CricShot10k  
**Why better than pose-only:** Rao et al. got only 80% with pose-only  
**Target:** Sensors journal or ICECE 2026

---

### Idea 5 — Temporal Attention Mechanism ⭐ (Recommended for MIST)
**Add:** Self-attention layer between EfficientNetV2-S and GRU  
**Effect:** Model learns which of the 15 frames matter most  
**Bonus:** GradCAM visualization of attention weights per shot class  

```python
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
    
    def forward(self, x):
        # x: [batch, 15, feature_dim]
        x = x.permute(1, 0, 2)  # [15, batch, feature_dim]
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output.permute(1, 0, 2), attn_weights
```

**Why novel:** Simple, interpretable, directly on top of their pipeline  
**Target:** ICCIT 2026 or ICECE 2026

---

### Idea 6 — Cross-Format Generalization Study
**Experiments (no new model needed):**
```
Train on T20 only      → Test on ODI, Test cricket
Train on men's only    → Test on women's
Train on pre-2017      → Test on 2022-present
Train on international → Test on franchise leagues
```
**Why novel:** CricShot10k is the ONLY dataset that makes this possible  
**Why easy:** Pure analysis — no new model required  
**Target:** ICCIT 2026

---

### Idea 7 — Knowledge Distillation / Edge Deployment ⭐ (Best fit for MIST)
**Teacher:** CricShotNet (EfficientNetV2-S + GRU, ~25M params, 89.09%)  
**Student:** MobileNetV2 + GRU-64 (<5M params, target >85%)  

**Target hardware:**
- Raspberry Pi 4
- NVIDIA Jetson Nano
- Android Phone

**Why fits MIST:** MIST's embedded/hardware engineering background  
**Why novel:** No cricket shot paper has done edge deployment  
**Target:** IEEE Access or ICECE 2026

---

### Idea 8 — Explainability Study (XAI)
**Techniques:** GradCAM, SHAP  
**Questions to answer:**
- Does the model look at the bat or the body?
- Does cropping actually help focus?
- Which of the 15 frames is most decisive?
- What causes Cover Drive / Straight Drive confusion?

**Why novel:** No cricket shot paper has done proper XAI analysis  
**Can be added as a section** to any other idea above

---

### Idea 9 — Data Augmentation Study
**They used:** Only horizontal flipping  
**Test additionally:**
- Temporal jittering (random frame sampling)
- Color jitter (day/night match simulation)
- Speed perturbation (fast/slow bowlers)
- MixUp, CutMix for videos
- RandAugment policy search

**Why novel:** First comprehensive augmentation study on CricShot10k

---

### Idea 10 — Ensemble Methods
**Approaches:**
- Top-3 CNN model ensemble (EfficientNetV2-S + Xception + EfficientNetV2-B3)
- Multi-scale ensemble (10, 15, 20 frames)
- Test-Time Augmentation (TTA)
- Two-stream: RGB + Optical Flow

**Expected gain:** +1 to +5% over single model

---

## 8. Recommended Approach for MIST CSE Student

### Top Recommendation
> **Idea 7 (Edge Deployment) — Best fit for MIST**

MIST has strong embedded systems culture. Deploy CricShotNet on Raspberry Pi or Jetson Nano using knowledge distillation. No other cricket shot paper has done this.

**Paper title idea:**
> "Efficient Cricket Shot Recognition for Real-Time Broadcast Analytics: A Lightweight Approach on CricShot10k"

### Alternative (Faster, 2-3 months)
> **Idea 5 (Temporal Attention) + Idea 8 (XAI)**

Add attention on top of their pipeline + GradCAM visualization. Strong enough for ICCIT 2026.

---

## 9. Paper Writing Checklist

For any idea chosen, make sure to include:

```
□ Reproduce baseline first (89.09%) — before adding anything new
□ Ablation study — show each component's contribution separately
□ Confusion matrix — full 15×15 matrix
□ GradCAM / visualization — at least one visual analysis
□ Cross-dataset test — also test on KUCricShot and CricShotClassify
□ Statistical significance — run 3-5 times, report mean ± std
□ Failure case analysis — show what your model still gets wrong
□ Compare with their paper's results fairly
□ Cite CricShot10k paper properly (BibTeX above)
```

---

## 10. Related Papers to Know

| Paper | Dataset | Classes | Size | Accuracy | Model |
|---|---|---|---|---|---|
| Hoque et al. 2023 (KUCricShot) | KUCricShot | 4 | 1,278 | 73% | LRCN |
| Sen et al. 2021 (CricShotClassify) | CricShotClassify | 10 | 1,867 | 93% | VGG16-GRU |
| Rao et al. 2025 | Custom (1922 clips) | 6 | 1,922 | 80% | MediaPipe-GRU |
| **Dihan et al. 2026 (CricShot10k)** | **CricShot10k** | **15** | **10,086** | **89%** | **EfficientNetV2-S+GRU** |

### EfficientNetV2 Paper (used as backbone)
- **Title:** EfficientNetV2: Smaller Models and Faster Training
- **Authors:** Mingxing Tan, Quoc V. Le (Google Research)
- **Published:** ICML 2021
- **Key claim:** 5-11x faster training than prior models, up to 6.8x smaller
- **GitHub:** https://github.com/google/automl/tree/master/efficientnetv2 (Public ✅)
- **PyTorch (via timm):** `timm.create_model('tf_efficientnetv2_s', pretrained=True)`

---

## 11. Key Technical Notes for Implementation

### Reproducing Their Baseline
```python
# Install dependencies
pip install timm decord torch torchvision ultralytics

# Load EfficientNetV2-S (feature extractor only)
import timm
cnn = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)

# Frame sampling
import decord
vr = decord.VideoReader(video_path)
frame_indices = np.linspace(0, len(vr)-1, 15, dtype=int)
frames = vr.get_batch(frame_indices).asnumpy()

# GRU temporal module
import torch.nn as nn
gru = nn.GRU(input_size=feature_dim, hidden_size=128, batch_first=True)

# Classifier
classifier = nn.Sequential(
    nn.BatchNorm1d(128),
    nn.Linear(128, 1024),
    nn.ReLU(),
    nn.Linear(1024, 15),
    nn.Softmax(dim=1)
)
```

### Important Implementation Details
- Frame size: **224 × 224** (after cropping)
- Frames per video: **15** (uniformly sampled)
- GRU hidden units: **128**
- Batch size: **8** (hardware limited)
- Optimizer: **Adam** with lr=**1e-4**
- Early stopping patience: **10** epochs
- Train videos shuffled each epoch
- Stratified splitting (class-balanced across splits)
- Horizontal flip augmentation on **training set only**

---

## 12. Target Conferences / Journals

| Venue | Type | Relevance |
|---|---|---|
| **ICCIT** (Int. Conf. Computer & Info. Technology) | Conference | ⭐⭐⭐ Bangladesh, very relevant |
| **ICECE** (Int. Conf. Electrical & Computer Eng.) | Conference | ⭐⭐⭐ BUET, strong for embedded work |
| **IEEE Access** | Open-access Journal | ⭐⭐⭐ Same venue as base paper |
| **Sensors** (MDPI) | Journal | ⭐⭐ Same venue as CricShotClassify |
| **Multimedia Tools & Applications** | Journal | ⭐⭐ Good for analysis papers |

---

*This context file was generated from a research planning conversation about building on the CricShot10k paper (IEEE Access, Feb 2026) for a MIST CSE student's conference paper.*
