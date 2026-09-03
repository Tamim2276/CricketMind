# CricShot10k — 10 Novel Improvement Ideas
> **For:** MIST CSE Department Research  
> **Base Paper:** CricShot10k (IEEE Access, Feb 2026)  
> **Baseline to Beat:** 89.09% Top-1 Accuracy  
> **GitHub:** https://github.com/Dihan69/CricShot10k

---

## Quick Reference Table

| # | Idea | Difficulty | Time | Expected Gain | Target Venue |
|---|---|---|---|---|---|
| 1 | Transformer-Based Video Classification | 🔴 High | 3-4 months | +3-6% | IEEE Access / ICCIT |
| 2 | Multi-Task Learning | 🔴 High | 4-5 months | +2-4% | IEEE Access |
| 3 | Few-Shot Learning for Rare Shots | 🔴 High | 4-6 months | High novelty | Pattern Recognition |
| 4 | Pose-Guided Shot Classification | 🟡 Medium | 3-4 months | +3-5% | Sensors / ICECE |
| 5 | Temporal Attention Mechanism | 🟡 Medium | 2-3 months | +2-4% | ICCIT / ICECE |
| 6 | Cross-Format Generalization Study | 🟡 Medium | 2-3 months | Analysis paper | ICCIT |
| 7 | Knowledge Distillation / Edge Deployment | 🟡 Medium | 3-4 months | Practical | IEEE Access |
| 8 | Explainability Study (XAI) | 🟢 Low | 1-2 months | Add-on | Any conference |
| 9 | Data Augmentation Study | 🟢 Low | 1-2 months | +1-3% | Workshop / ICCIT |
| 10 | Ensemble Methods | 🟢 Low | 2-3 months | +1-3% | ICCIT / ICECE |

---

---

# 🔴 HIGH IMPACT IDEAS

---

## Idea 1 — Transformer-Based Video Classification

### Overview
Replace the EfficientNetV2-S + GRU pipeline with modern video transformer architectures that capture long-range temporal dependencies more effectively.

### Problem With Current Approach
- GRU processes frames **sequentially** — limited in capturing global temporal context
- GRU can suffer from **vanishing gradients** over long sequences
- Transformer **self-attention** can relate any two frames directly regardless of distance

### Proposed Architecture

```
Input: 15 frames × 224×224
        ↓
Cropping + Segmentation (keep their preprocessing)
        ↓
[OPTION A] Video Swin Transformer
        ↓
[OPTION B] TimeSformer (Divided Space-Time Attention)
        ↓
[OPTION C] VideoMAE (Masked Autoencoder Pre-training)
        ↓
[OPTION D] MViT (Multiscale Vision Transformer)
        ↓
Classification Head (15 classes)
```

### Models to Try

| Model | Key Idea | Why Suitable |
|---|---|---|
| **Video Swin Transformer** | Shifted window attention in 3D | Efficient for short clips like 1-second shots |
| **TimeSformer** | Divided space-time attention | Separates spatial and temporal attention |
| **VideoMAE** | Self-supervised pre-training on videos | Less labeled data needed, strong features |
| **MViT** | Multiscale feature hierarchy | Captures both fine-grained and global motion |

### Experiments to Run

```
Experiment 1: Video Swin-T vs CricShotNet baseline
Experiment 2: TimeSformer vs CricShotNet baseline
Experiment 3: VideoMAE (pretrained) vs CricShotNet baseline
Experiment 4: Transformer + their cropping/segmentation layers
Experiment 5: Ablation — with vs without cropping/segmentation
```

### Why Novel
- Paper explicitly mentions Video Swin and TimeSformer as **future work**
- No cricket shot paper has used video transformers on CricShot10k
- Directly comparable baseline exists (89.09%)

### Expected Results
- Likely **+3 to +6%** improvement over baseline
- Strong cross-dataset generalization

### Implementation Resources
- PyTorch + `timm` library for Video Swin
- HuggingFace `transformers` for VideoMAE
- Pre-trained weights available on ModelZoo

### Target Venue
- **IEEE Access** (if >92% accuracy achieved)
- **ICCIT 2026** (if results are strong)

---

## Idea 2 — Multi-Task Learning

### Overview
Train a single unified model that simultaneously learns to predict multiple cricket-related attributes from the same video, forcing it to learn richer and more generalizable representations.

### Problem With Current Approach
- CricShotNet is trained for **only one task** — shot type classification
- A model that understands multiple aspects of the game should generalize better

### Proposed Architecture

```
Input Video
        ↓
Shared Backbone (EfficientNetV2-S or Transformer)
        ↓
Shared GRU / Temporal Module
        ↓
        ├──→ Head 1: Shot Type (15 classes) — Primary Task
        ├──→ Head 2: Batter Handedness (2 classes: Left / Right)
        └──→ Head 3: Match Format (3 classes: Test / ODI / T20)
```

### Loss Function
```
Total Loss = α × Loss_shot + β × Loss_handedness + γ × Loss_format

Where: α = 1.0 (primary), β = 0.3, γ = 0.3 (auxiliary)
```

### Labels Available in CricShot10k
- Shot type → already labeled (15 classes)
- Batter handedness → derivable from video metadata / manual labeling
- Match format → available in dataset metadata (Test/ODI/T20)

### Experiments to Run

```
Experiment 1: Single task baseline (shot only) — 89.09%
Experiment 2: Shot + Handedness
Experiment 3: Shot + Format
Experiment 4: Shot + Handedness + Format (full multi-task)
Experiment 5: Vary α, β, γ weights — find best combination
```

### Why Novel
- No existing cricket shot classification paper uses multi-task learning
- Forces shared features to encode **both temporal and contextual information**
- Practically useful — a real system needs to know format AND shot type simultaneously

### Expected Results
- Auxiliary tasks act as **regularizers** → reduce overfitting
- Expected **+2 to +4%** improvement on shot classification
- Bonus: free handedness and format classifiers

### Target Venue
- **IEEE Access** or **Sensors** journal
- **ICECE 2026**

---

## Idea 3 — Few-Shot Learning for Rare Shots

### Overview
Apply few-shot learning techniques specifically targeting the underrepresented shot classes in CricShot10k (Reverse Sweep: 257, Scoop: 279, Upper Cut: 281).

### Problem With Current Approach
- Model performance is heavily influenced by visual similarity, not just sample count
- However, rare shots are harder to collect in real-world scenarios
- A model that can learn from **few examples** is more practically deployable

### Class Imbalance in CricShot10k

| Class | Samples | Status |
|---|---|---|
| Lofted Legside | 1,094 | Majority |
| Lofted Offside | 1,054 | Majority |
| Reverse Sweep | 257 | **Minority** |
| Scoop | 279 | **Minority** |
| Upper Cut | 281 | **Minority** |
| Late Cut | 325 | **Minority** |

### Proposed Approaches

#### Approach A — Prototypical Networks
```
1. Compute a "prototype" (mean feature vector) for each shot class
2. Classify new shots by distance to nearest prototype
3. Works well with few examples per class
```

#### Approach B — MAML (Model-Agnostic Meta-Learning)
```
1. Train model to quickly adapt to new tasks with few gradient steps
2. Apply to cricket: train on common shots, adapt to rare shots
3. Few-shot evaluation: 5-shot, 10-shot classification
```

#### Approach C — Class-Conditional Data Synthesis
```
1. Use conditional video generation (VideoGAN / Diffusion)
2. Generate synthetic rare shot videos
3. Augment training set with synthetic samples
4. Evaluate improvement on rare class metrics
```

### Experiments to Run

```
Experiment 1: Baseline (standard training) — precision/recall on rare classes
Experiment 2: Prototypical Networks — 5-shot, 10-shot evaluation
Experiment 3: MAML — few-shot adaptation
Experiment 4: Synthetic augmentation with VideoGAN
Experiment 5: Compare all approaches on rare class F1-score
```

### Why Novel
- Most challenging and most publishable idea
- Directly addresses a real limitation of CricShot10k
- Few-shot learning for **video sports action recognition** is an open problem
- Scoop achieved perfect recall (1.0000) with only 279 samples — interesting phenomenon to study

### Expected Results
- Improved F1-score on rare classes (currently Late Cut F1 = 0.7451)
- Better generalization to unseen shot variations

### Target Venue
- **Pattern Recognition** journal
- **CVPR Workshop** on Few-Shot Learning
- **IEEE Access**

---
---

# 🟡 MEDIUM IMPACT IDEAS

---

## Idea 4 — Pose-Guided Shot Classification (Two-Stream Fusion)

### Overview
Combine RGB visual features WITH body skeleton/pose keypoints in a two-stream architecture to better distinguish visually similar shots.

### Problem With Current Approach
- CricShotNet relies purely on **visual appearance**
- Visually similar shots (Cover Drive vs Straight Drive, Pull vs Hook) are hard to distinguish
- Body pose encodes **biomechanical information** that appearance alone misses

### Why Previous Pose-Only Approach Failed
- Rao et al. [29] used MediaPipe pose landmarks only → 80% accuracy
- Pose-only removes texture, appearance, and context
- **Fusion** of both should outperform either alone

### Proposed Two-Stream Architecture

```
Input Video (15 frames)
        │
        ├──────────────────────────────┐
        ↓                              ↓
[Stream 1: Visual]              [Stream 2: Pose]
Cropping + Segmentation         MediaPipe Pose Estimation
        ↓                              ↓
EfficientNetV2-S                Pose Keypoint Sequence
        ↓                              ↓
GRU (128 units)                 GRU / Transformer
        ↓                              ↓
Visual Feature Vector           Pose Feature Vector
        └──────────────┬───────────────┘
                       ↓
               Fusion Layer
               (Concatenation / Cross-Attention)
                       ↓
               Classifier (15 classes)
```

### Pose Features to Extract (MediaPipe)
- 33 body keypoints per frame
- Key joints: wrists, elbows, shoulders, hips, knees, ankles
- Derived features: joint angles, limb velocities, bat swing trajectory

### Fusion Strategies to Compare

| Strategy | Description |
|---|---|
| Early Fusion | Concatenate pose + visual features before GRU |
| Late Fusion | Separate GRUs, concatenate outputs before classifier |
| Cross-Attention Fusion | Pose queries attend to visual keys/values |
| Weighted Fusion | Learn adaptive weights for each stream |

### Experiments to Run

```
Experiment 1: Visual only (baseline) — 89.09%
Experiment 2: Pose only — reproduce ~80% from literature
Experiment 3: Early fusion
Experiment 4: Late fusion
Experiment 5: Cross-attention fusion
Experiment 6: Focus analysis — does pose help on Pull vs Hook specifically?
```

### Why Novel
- First paper to do **visual + pose fusion** on CricShot10k
- Directly addresses the inter-class similarity problem they identified
- Cross-attention fusion is a modern, publishable approach

### Expected Results
- **+3 to +5%** improvement especially on visually similar shot pairs
- Better interpretability — can analyze which joints matter most

### Target Venue
- **Sensors** journal (same as CricShotClassify paper)
- **ICECE 2026**
- **ICCIT 2026**

---

## Idea 5 — Temporal Attention Mechanism

### Overview
Add a self-attention layer between the CNN feature extractor and GRU to let the model learn **which frames are most important** for shot classification.

### Problem With Current Approach
- All 15 frames are treated **equally** by the GRU
- In reality, the moment of bat-ball contact is most critical
- Early frames (ball approaching) and late frames (ball trajectory) are less informative

### Proposed Architecture

```
15 frames × 224×224
        ↓
Cropping + Segmentation (unchanged)
        ↓
EfficientNetV2-S → 15 feature vectors [f₁, f₂, ..., f₁₅]
        ↓
┌─────────────────────────────────┐
│     TEMPORAL ATTENTION MODULE   │
│                                 │
│  Query = Key = Value = [f₁...f₁₅]│
│  Attention Score = softmax(QKᵀ/√d)│
│  Output = weighted sum of values │
└─────────────────────────────────┘
        ↓
Attended feature sequence
        ↓
GRU (128 units)
        ↓
Batch Norm → Dense → Softmax
```

### Types of Attention to Compare

| Type | Description |
|---|---|
| Self-Attention | Frames attend to each other |
| Scaled Dot-Product | Standard transformer attention |
| Multi-Head Attention | Multiple attention heads in parallel |
| Temporal Pooling Attention | Learns a single importance weight per frame |

### Visualization (Key Novelty)
```python
# Visualize attention weights across 15 frames
attention_weights = [0.02, 0.03, 0.05, 0.12, 0.25, 0.28, 0.10, 0.05, ...]
# Shows model focuses on frames 5-7 (moment of shot execution)
```
- Plot attention heatmaps over frames for different shot types
- Show that model learns to focus on the **critical moment** of shot execution
- This is a strong visualization for conference papers

### Experiments to Run

```
Experiment 1: Baseline (no attention) — 89.09%
Experiment 2: + Single-head temporal attention
Experiment 3: + Multi-head temporal attention (2, 4, 8 heads)
Experiment 4: Attention before GRU vs after GRU
Experiment 5: Visualize attention weights for each shot class
```

### Why Novel
- Simple addition on top of their **exact pipeline**
- Highly interpretable — you can visualize what the model looks at
- Low implementation complexity, high novelty value
- The paper suggested this direction implicitly

### Expected Results
- **+2 to +4%** improvement over baseline
- Strong qualitative results through attention visualization

### Implementation
```python
import torch
import torch.nn as nn

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

### Target Venue
- **ICCIT 2026**
- **ICECE 2026**
- **IEEE Access** (if combined with other contributions)

---

## Idea 6 — Cross-Format Generalization Study

### Overview
A systematic experimental study analyzing how cricket shot classifiers trained on one format/demographic generalize to others — using CricShot10k's unique diversity.

### Why This Is Uniquely Possible
CricShot10k is the **FIRST and ONLY** dataset that includes:
- ✅ All formats (Test, ODI, T20)
- ✅ Men's and Women's cricket
- ✅ Under-19 matches
- ✅ Matches from 1996 to 2024

No previous paper could run these experiments.

### Proposed Experiments

#### Set A — Format Generalization
```
Train on T20 only      → Test on ODI, Test cricket
Train on ODI only      → Test on T20, Test cricket
Train on Test only     → Test on T20, ODI
Train on all formats   → Test on each format separately
```

#### Set B — Gender Generalization
```
Train on men's only    → Test on women's
Train on women's only  → Test on men's
Train on combined      → Test on each gender separately
```

#### Set C — Era Generalization
```
Train on pre-2017      → Test on 2022-present
Train on 2022-present  → Test on pre-2017
Train on all eras      → Test on each era
```

#### Set D — Tournament Type Generalization
```
Train on international → Test on franchise leagues
Train on franchise     → Test on international
```

### Analysis Questions to Answer
1. Which shot types generalize best across formats?
2. Does jersey color (Test whites vs colored kits) cause significant performance drops?
3. Are women's batting shots classified differently from men's?
4. Has batting style changed significantly from 1996 to 2024?
5. Do franchise league shots (IPL, PSL) look different from international cricket?

### Metrics to Report

| Metric | Purpose |
|---|---|
| Per-format Top-1 Accuracy | Direct generalization measure |
| Per-class F1 (cross-format) | Which shots transfer well |
| Confusion matrices per format | Visual analysis |
| Domain gap analysis | Quantify distribution shift |

### Why Novel
- **Pure analysis paper** — no new model required
- CricShot10k is the only dataset making this possible
- Results will be insightful and immediately useful to the community
- Directly contributes to understanding **domain shift** in sports analytics

### Expected Contribution
- First systematic cross-format generalization study for cricket
- Identification of format-specific challenging shots
- Guidelines for building format-agnostic cricket AI systems

### Target Venue
- **ICCIT 2026** (strong fit — analysis-style paper)
- **Multimedia Tools and Applications** journal
- **IEEE Transactions on Sports Engineering**

---

## Idea 7 — Knowledge Distillation for Edge Deployment

### Overview
Compress the heavy CricShotNet model into a lightweight version suitable for **real-time deployment on mobile or embedded devices** using knowledge distillation.

### Problem With Current Approach
- EfficientNetV2-S has **20.33M parameters** — too heavy for edge devices
- GRU adds additional parameters
- Real-time cricket analytics needs **fast inference** on broadcast hardware

### Proposed Pipeline

```
TEACHER MODEL (CricShotNet)
EfficientNetV2-S (20.33M) + GRU (128)
Accuracy: 89.09%
        ↓ Knowledge Transfer
STUDENT MODEL (Lightweight)
MobileNetV2 (2.26M) + GRU (64)
Target: >85% accuracy with 5× fewer parameters
```

### Knowledge Distillation Types to Try

| Type | Description |
|---|---|
| **Response-based** | Student mimics teacher's softmax output (soft labels) |
| **Feature-based** | Student mimics teacher's intermediate feature maps |
| **Relation-based** | Student mimics relationships between sample pairs |

### Loss Function
```
Total Loss = α × CrossEntropy(student, true_labels)
           + β × KL_Divergence(student_logits, teacher_logits / T)

Where T = temperature (controls softness of teacher predictions)
      α = 0.5, β = 0.5, T = 4 (tune experimentally)
```

### Target Hardware (Fits MIST Engineering Background)

| Device | Use Case |
|---|---|
| Raspberry Pi 4 | Low-cost broadcast assistant |
| NVIDIA Jetson Nano | Real-time stadium analytics |
| Android Phone | Player self-analysis app |
| Coral Edge TPU | Ultra-low power IoT deployment |

### Experiments to Run

```
Experiment 1: Teacher baseline — 89.09%, inference time
Experiment 2: Student without distillation — accuracy, speed
Experiment 3: Student with response-based distillation
Experiment 4: Student with feature-based distillation
Experiment 5: Student with combined distillation
Experiment 6: Benchmark on Raspberry Pi / Jetson Nano
```

### Metrics to Report

| Metric | Teacher | Student (Target) |
|---|---|---|
| Top-1 Accuracy | 89.09% | >85% |
| Parameters | ~25M | <5M |
| Inference Time (GPU) | X ms | <X/3 ms |
| Inference Time (CPU/Edge) | Measure | Real-time (>30fps) |
| Model Size (MB) | Large | <50 MB |

### Why Novel
- **No cricket shot paper** has done edge deployment
- Plays directly to **MIST's embedded systems strength**
- Practically very impactful — real broadcast applications
- Strong engineering contribution

### Target Venue
- **IEEE Access** (strong fit — practical AI systems)
- **ICECE 2026** (embedded systems track)
- **IoT journals** if Raspberry Pi results are strong

---
---

# 🟢 LOWER EFFORT IDEAS

---

## Idea 8 — Explainability Study (XAI)

### Overview
Apply explainable AI techniques to understand **what CricShotNet actually looks at** when classifying cricket shots.

### Techniques to Apply

#### GradCAM (Gradient-weighted Class Activation Mapping)
```
- Produces heatmap overlay on input frames
- Shows which spatial regions activated the classification decision
- Apply per frame across the 15-frame sequence
```

#### SHAP (SHapley Additive exPlanations)
```
- Quantifies contribution of each feature/frame to final prediction
- Shows which frames were most decisive for each shot type
```

#### Attention Visualization (if Idea 5 is combined)
```
- Plot attention weights across 15 frames
- Show temporal focus patterns per shot class
```

### Analysis Questions

```
1. Does the model look at the bat or the body?
   → Expected: Both, but bat movement is most critical

2. Does cropping actually help focus?
   → Compare GradCAM with vs without cropping layer

3. Does segmentation help?
   → Compare GradCAM with vs without segmentation layer

4. Which frames are most important?
   → Frame 1 (setup), Frame 8 (contact), Frame 15 (follow-through)?

5. What causes Cover Drive / Straight Drive confusion?
   → GradCAM on misclassified samples
```

### Visualizations to Include in Paper

| Figure | Description |
|---|---|
| GradCAM grid | 15 frames × top-5 shot classes with heatmaps |
| Frame importance plot | Bar chart of which frame index matters most |
| Misclassification analysis | GradCAM on confused shot pairs |
| Cropping effect | GradCAM before vs after cropping |

### Why Novel
- **No cricket shot paper** has done proper XAI analysis
- Builds trust and interpretability in the model
- Conference reviewers love visualizations
- Can be added as a **section** to any other idea above

### Implementation
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=model, target_layers=[model.efficientnet.features[-1]])
grayscale_cam = cam(input_tensor=input_frames)
visualization = show_cam_on_image(rgb_img, grayscale_cam[0])
```

### Target Venue
- Add-on to any conference paper
- **IEEE Access** supplementary analysis
- **ICCIT 2026**

---

## Idea 9 — Comprehensive Data Augmentation Study

### Overview
Systematic evaluation of various video augmentation strategies on CricShot10k — the original paper only used horizontal flipping.

### Augmentation Strategies to Test

#### Spatial Augmentations
| Augmentation | Description | Cricket Relevance |
|---|---|---|
| Horizontal Flip | Mirror the video | Left/right handed batters ✅ |
| Random Crop | Crop different regions | Simulates different camera positions |
| Color Jitter | Vary brightness/contrast | Day/night matches, different grounds |
| Grayscale | Convert to grayscale | Removes jersey color dependency |
| Gaussian Blur | Blur frames slightly | Simulates camera focus issues |

#### Temporal Augmentations
| Augmentation | Description | Cricket Relevance |
|---|---|---|
| Temporal Jittering | Random frame sampling instead of uniform | Simulates different shot speeds |
| Speed Perturbation | Speed up or slow down video | Fast vs slow bowlers |
| Frame Dropout | Randomly drop frames | Robustness to missing frames |
| Temporal Reversal | Reverse frame order | Check if model uses temporal direction |

#### Advanced Augmentations
| Augmentation | Description |
|---|---|
| MixUp | Blend two videos and their labels |
| CutMix | Cut and paste regions between videos |
| CutOut | Randomly mask regions of frames |
| RandAugment | Automated augmentation policy search |

### Experiments to Run

```
Baseline: Horizontal flip only → 89.09%

Single augmentation tests:
- + Color Jitter
- + Temporal Jittering
- + Random Crop
- + Grayscale
- + Frame Dropout

Combination tests:
- Best spatial + best temporal
- All augmentations combined
- RandAugment policy search
```

### Analysis Questions
1. Which augmentation helps most on rare classes?
2. Does color jitter help generalize across Test (white) vs colored kits?
3. Does temporal jittering improve robustness to different shot speeds?
4. Is there a point of diminishing returns with augmentation stacking?

### Why Novel
- CricShot10k paper used **only horizontal flip** — huge gap to fill
- Cricket-specific augmentation analysis hasn't been done
- Results directly useful for community building on CricShot10k

### Target Venue
- **ICCIT 2026**
- Workshop paper at a computer vision conference

---

## Idea 10 — Ensemble Methods

### Overview
Combine predictions from multiple models to achieve better accuracy than any single model alone.

### Ensemble Strategies

#### Strategy A — Model Ensemble
```
Model 1: EfficientNetV2-S + GRU (their best — 89.09%)
Model 2: Xception + GRU (88.64%)
Model 3: EfficientNetV2-B3 + GRU (88.39%)
Model 4: ConvNeXt-Tiny + GRU (87.20%)

Fusion: Average / Weighted Average / Max Voting of softmax outputs
```

#### Strategy B — Multi-Scale Ensemble
```
Scale 1: 15 frames sampled (original)
Scale 2: 10 frames sampled (fewer, faster)
Scale 3: 20 frames sampled (more, slower)

Each scale trained separately → ensemble predictions
```

#### Strategy C — Test-Time Augmentation (TTA)
```
For each test video:
1. Original video
2. Horizontally flipped
3. Color jittered version
4. Temporal jittered version

Average predictions across all 4 versions
```

#### Strategy D — Two-Stream Late Fusion
```
Stream 1: RGB frames (visual) → prediction
Stream 2: Optical flow (motion) → prediction
Late fusion: weighted average
```

### Experiments to Run

```
Experiment 1: Best single model — 89.09%
Experiment 2: Top-3 model ensemble (avg)
Experiment 3: Top-3 model ensemble (weighted)
Experiment 4: TTA with 3 augmentations
Experiment 5: Multi-scale ensemble
Experiment 6: Two-stream (RGB + Optical Flow)
```

### Why Novel
- Ensemble methods have not been applied to cricket shot classification
- Guaranteed to improve over single model baseline
- Two-stream with optical flow is a well-known strong approach

### Expected Results
- **+1 to +3%** improvement with simple ensembles
- Potentially **+4 to +5%** with optical flow two-stream

### Target Venue
- **ICCIT 2026**
- **ICECE 2026**

---
---

# 🎯 Recommendation For MIST CSE Students

## If You Have 2-3 Months
> **Idea 5 (Temporal Attention) + Idea 8 (XAI)**
- Implement attention on top of their pipeline
- Add GradCAM visualizations
- Strong enough for **ICCIT 2026**

## If You Have 4-6 Months
> **Idea 7 (Edge Deployment) — Best Fit for MIST**
- Leverage MIST's embedded/hardware background
- Deploy on Raspberry Pi or Jetson Nano
- Unique angle no other paper has taken
- Target: **IEEE Access** or **ICECE 2026**

## If You Want Maximum Impact
> **Idea 1 (Transformer) + Idea 5 (Attention) + Idea 8 (XAI)**
- Full pipeline replacement + interpretability
- Target: **IEEE Access journal**

---

## Paper Writing Checklist (Any Idea)

```
□ Reproduce baseline first (89.09%) — before adding anything new
□ Ablation study — show each component's contribution separately
□ Confusion matrix — full 15×15 matrix
□ GradCAM / visualization — at least one visual analysis
□ Cross-dataset test — test on KUCricShot and CricShotClassify too
□ Statistical significance — run 3-5 times, report mean ± std
□ Failure case analysis — show what your model still gets wrong
□ Compare with their paper's results fairly
```

---

*Prepared for MIST CSE Research — Based on CricShot10k (IEEE Access, Vol. 14, Feb 2026)*
