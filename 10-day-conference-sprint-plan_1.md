# 10-Day Sprint Plan: Beating CricShot10k's 89% Baseline
### Target: Conference submission by **August 27, 2026**
### Hardware: Intel Arc B580 (12GB VRAM) + Ryzen 5 7500F, local dev | Kaggle (CUDA T4/P100) for all real training | No conda — pip/venv only

**Workflow in one line:** write and smoke-test code locally on the Arc B580 (XPU backend), push to GitHub, run every real training job on Kaggle (CUDA backend). Full details in §12 — read it alongside §2 on Day 1, since the two setups happen together, not sequentially.

---

## 0. Read this first — honest scope check

You listed six approaches: **multi-scale CNN, GNN, 3D CNN, hybrid CNN, transformers, and other video classification algorithms.** Attempting all six from scratch, trained, ablated, and written up in 10 days on a single 12GB consumer GPU is not realistic — each one is roughly a paper's worth of work on its own, and GNN/3D CNN both carry real risk of *underperforming* the baseline on a 10k-clip dataset (your own reference doc says this too: pure GNN gets 80-82%, worse than 89%; pure 3D CNN often ties or loses to 2D+GRU at this data scale).

**What this plan actually builds:** the two upgrades with the best accuracy-per-day-of-effort ratio, which also happen to be the two your own analysis doc rated highest:

1. **Multi-Scale CNN** (fixes fine-grained bat-angle confusion — Pull vs Hook, Lofted Legside vs Offside)
2. **Lightweight Temporal Transformer replacing GRU** (fixes long-range frame forgetting)

Combined, these are the "Phase 1 + Phase 2" combo your doc estimated at **92–94% accuracy**, fits in 12GB VRAM, and is a clean, defensible "we extend CricShotNet" paper — not an oversized six-architecture demo that's shallow everywhere.

**GNN stream and 3D CNN/SlowFast are explicitly cut from the 10-day critical path.** They're listed as Day 7+ stretch goals only if you're ahead of schedule, and as "Future Work" in the paper otherwise — that's a completely normal, expected move in a conference paper and reviewers will not penalize you for it.

**Full Video Swin / TimeSformer / full SlowFast are cut entirely** — they need 24GB+ VRAM per your own doc's own warning; don't burn a day discovering that the hard way.

If you disagree and want to fight for GNN or 3D CNN too, see §11 for exactly where they'd slot in and what you'd have to give up.

---

## 1. Day-by-Day Plan

| Day | Date | Where | Focus | Deliverable by end of day |
|---|---|---|---|---|
| 1 | Aug 17 | Local + Kaggle | Env setup (§2), repo scaffold, GitHub push, Kaggle account + dataset upload (§12.1) | `torch.xpu.is_available() == True` locally, dataset live as a Kaggle Dataset, repo pushed |
| 2 | Aug 18 | Kaggle | Reproduce CricShotNet baseline (real training run) | Baseline trained on Kaggle, accuracy logged, confusion matrix saved, checkpoint versioned |
| 3 | Aug 19 | Local → Kaggle | Implement Multi-Scale CNN branch locally, smoke-test, push, kick off Kaggle run | Local smoke test passes (no OOM on tiny subset), Kaggle training started |
| 4 | Aug 20 | Kaggle | Multi-Scale CNN finishes training on Kaggle; tune if time allows | Multi-scale accuracy number in hand |
| 5 | Aug 21 | Local → Kaggle | Implement Temporal Transformer locally (replace GRU), smoke-test, push, kick off Kaggle run | Local smoke test passes, Kaggle training started |
| 6 | Aug 22 | Kaggle | Transformer model finishes training; start combined model | Transformer-only accuracy number in hand |
| 7 | Aug 23 | Kaggle | Train combined model (Multi-Scale + Transformer) | Combined model final accuracy + checkpoint saved and versioned |
| 8 | Aug 24 | Local | Ablation table, confusion matrix, error analysis, all figures (pull results/checkpoints down from Kaggle) | Every number and figure the paper needs exists on disk |
| 9 | Aug 25 | Local | Write the paper | Full draft in conference template |
| 10 | Aug 26 | Local | Buffer: proofread, format, submit **a day early** | Submitted (don't wait for Aug 27, servers choke near deadlines) |

Note this leaves Aug 27 itself as true slack — treat Aug 26 as your real deadline.

---

## 2. Day 1 — Environment Setup (no conda)

### 2.1 Enable Resizable BAR (BIOS — do this first, before anything else)
Arc GPUs need Resizable BAR enabled to perform correctly. Reboot into BIOS, enable "Above 4G Decoding" and "Re-Size BAR Support" (naming varies by motherboard vendor). Skipping this causes silent slowdowns that look like a software problem later.

### 2.2 Update Intel Arc drivers
Download the latest driver from Intel's official Arc download page and reboot after installing.

### 2.3 Create a clean venv (Windows PowerShell shown; Linux/WSL2 commands below it)

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / WSL2 (Ubuntu):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> If you hit driver-detection issues on native Windows, WSL2 Ubuntu tends to have more mature Intel GPU compute tooling (Level Zero, compute runtime) — keep this as your Plan B, not your starting point.

### 2.4 Install PyTorch with native XPU support (no IPEX needed, no conda needed)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### 2.5 Verify the GPU is actually visible
```bash
python -c "import torch; print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0))"
```
Expected output: `True` then `Intel(R) Arc(TM) B580 Graphics`. If `False`, recheck the driver install and Resizable BAR before doing anything else — do not proceed to modeling with a broken GPU setup.

### 2.6 Install remaining dependencies

**Two separate requirements files** — this matters. Kaggle notebooks ship with `torch`/`torchvision`/`torchaudio` pre-installed against CUDA; if `requirements.txt` lists them again, `pip install -r requirements.txt` on Kaggle will happily reinstall a CPU or mismatched-CUDA build over the working one and break `torch.cuda.is_available()`. Keep torch itself out of the shared file.

```bash
pip install decord opencv-python-headless ultralytics timm einops scikit-learn pandas matplotlib seaborn tqdm tensorboard pyyaml
```

`requirements.txt` (**local only** — installed once during §2.4/§2.6, includes torch's XPU build):
```
torch
torchvision
torchaudio
decord
opencv-python-headless
ultralytics
timm
einops
scikit-learn
pandas
matplotlib
seaborn
tqdm
tensorboard
pyyaml
```

`requirements-kaggle.txt` (**Kaggle only** — no torch/torchvision/torchaudio, so Kaggle's pre-installed CUDA build is left untouched):
```
decord
opencv-python-headless
ultralytics
timm
einops
scikit-learn
pandas
matplotlib
seaborn
tqdm
tensorboard
pyyaml
```
On Kaggle, install with `!pip install -q -r requirements-kaggle.txt` (§12.2 and §13 below already use this file, not the local one).

### 2.7 Fallback plan — set a hard time limit on debugging
Native XPU support is real and works, but it's a younger ecosystem than CUDA — you may hit an unsupported op or a driver quirk. **Give yourself a 3-hour cap on Day 1 for XPU troubleshooting.** If you blow past it, switch immediately to a free cloud GPU (Kaggle gives ~30 GPU-hours/week on T4/P100, no card required; Colab is a second option) and keep developing there while you fix Arc B580 in the background. A 10-day deadline cannot absorb a multi-day driver rabbit hole.

---

## 3. Folder Structure

Create this exact structure on Day 1 — it maps directly onto the day-by-day plan so nothing gets lost.

```
cricshot-improve/
├── README.md
├── requirements.txt             # local only (includes torch+xpu)
├── requirements-kaggle.txt      # Kaggle only (no torch — don't touch its preinstalled CUDA build)
├── .gitignore
├── configs/
│   ├── baseline.yaml            # local smoke-test config (tiny subset, batch_size 2)
│   ├── multiscale.yaml
│   ├── transformer.yaml
│   ├── combined.yaml
│   ├── baseline_kaggle.yaml     # real training configs — data_root/checkpoint_dir point at /kaggle/...
│   ├── multiscale_kaggle.yaml
│   ├── transformer_kaggle.yaml
│   └── combined_kaggle.yaml
├── data/
│   ├── raw/                    # downloaded CricShot10k clips (gitignored)
│   ├── splits/                 # train.csv / val.csv / test.csv
│   └── cache/                  # optional pre-extracted frames
├── src/
│   ├── datasets/
│   │   └── cricshot_dataset.py
│   ├── models/
│   │   ├── backbone.py         # EfficientNetV2-S wrapper
│   │   ├── temporal_gru.py     # baseline head (for reproduction)
│   │   ├── temporal_transformer.py
│   │   ├── multiscale_cnn.py
│   │   └── combined_model.py
│   ├── utils/
│   │   ├── device.py           # xpu / cuda / cpu auto-picker
│   │   ├── metrics.py
│   │   └── viz.py
│   ├── train.py
│   └── evaluate.py
├── experiments/
│   ├── baseline/
│   ├── multiscale/
│   ├── transformer/
│   └── combined/
├── notebooks/
│   └── error_analysis.ipynb
└── paper/
    ├── figures/
    ├── tables/
    └── draft.md
```

```bash
mkdir -p cricshot-improve/{configs,data/raw,data/splits,data/cache,src/datasets,src/models,src/utils,experiments/baseline,experiments/multiscale,experiments/transformer,experiments/combined,notebooks,paper/figures,paper/tables}
cd cricshot-improve
```

`.gitignore`:
```
.venv/
data/raw/
data/cache/
experiments/*/checkpoints/
__pycache__/
*.pt
```

---

## 4. Day 1 (cont.) — Dataset Acquisition

1. Clone the official CricShot10k repo (paper's code + dataset access): `https://github.com/Dihan69/CricShot10k`
2. Follow its instructions to pull the 10,086 one-second clips across 15 classes. **Check whether the repo ships pre-cropped/pre-segmented frames** — the paper's cropping (YOLOv11 striker+bat detection) and segmentation (semi-transparent overlay) steps add real accuracy (+17pp per their ablation) but re-implementing that pipeline yourself would eat 2+ days you don't have. Reuse their preprocessing artifacts if provided; only rebuild it yourself as a last resort.
3. **Use the authors' original train/val/test split if the repo provides one.** Reproducing your own stratified split introduces a confound when you compare against their reported 89% — a reviewer's first question will be "same split?"
4. Respect the dataset's terms: the paper states it's for **non-commercial research use only**, with copyright of the original footage remaining with the rights holders. Cite the CricShot10k paper (Dihan et al., IEEE Access 2026) properly in your related work.

```
data/raw/<class_name>/<clip_id>.mp4
data/splits/train.csv   # clip_path,label
data/splits/val.csv
data/splits/test.csv
```

---

## 5. Day 2 — Reproduce the Baseline First

Do not skip this. If your reproduction doesn't land near 85–89%, your "improvement" numbers later are meaningless — you need a working, trustworthy baseline to subtract from.

`src/utils/device.py` — **this is the file that makes local-Arc / Kaggle-CUDA portability work.** Same code runs unmodified in both places; it just detects what it's sitting on:
```python
import torch

def get_device():
    """XPU (local Arc B580) takes priority if present; falls back to CUDA
    (Kaggle T4/P100) or CPU. hasattr guards against older torch builds
    that don't ship the xpu module at all."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_amp_settings(device: torch.device):
    """Mixed-precision settings differ by hardware:
    - Arc B580 (xpu): bf16 autocast, no GradScaler needed
    - T4 / P100 (cuda): fp16 autocast + GradScaler (T4 lacks native bf16
      tensor cores on Turing; P100/Pascal has no tensor cores at all, so
      fp16 mainly helps memory, not speed, but is still worth using)
    - cpu: no autocast
    Returns (dtype_or_None, use_scaler: bool)
    """
    if device.type == "xpu":
        return torch.bfloat16, False
    if device.type == "cuda":
        return torch.float16, True
    return None, False
```

Use it in the training loop instead of hardcoding a device string:
```python
device = get_device()
amp_dtype, use_scaler = get_amp_settings(device)
scaler = torch.amp.GradScaler(enabled=use_scaler)

for clip, labels in dataloader:
    clip, labels = clip.to(device), labels.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(clip)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
This is the only change needed anywhere in the codebase to make it run correctly on both machines — never hardcode `"xpu"` or `"cuda"` anywhere else in `src/`.

`src/models/backbone.py`:
```python
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class FrameEncoder(nn.Module):
    """EfficientNetV2-S feature extractor, classifier head removed."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.out_dim = 1280  # EfficientNetV2-S output channels

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x  # (B, out_dim)
```

`src/models/temporal_gru.py` (baseline reproduction):
```python
import torch.nn as nn

class GRUHead(nn.Module):
    def __init__(self, in_dim=1280, hidden=128, num_classes=15):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc1 = nn.Linear(hidden, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: (B, T=15, in_dim)
        _, h = self.gru(x)
        h = h.squeeze(0)
        h = self.bn(h)
        h = self.relu(self.fc1(h))
        return self.fc2(h)
```

**Two config files for this model, per §12's local/Kaggle split:**

`configs/baseline.yaml` — **local smoke test only**, run on the Arc B580 to confirm the code works before trusting it to a Kaggle GPU-hour budget:
```yaml
frames_per_clip: 15
frame_size: 224
batch_size: 2
optimizer: adam
lr: 1e-4
epochs: 1
data_root: data/raw
checkpoint_dir: experiments/baseline/checkpoints
```
Run with `python src/train.py --config configs/baseline.yaml --smoke_test` — confirm loss decreases over the 1 epoch and nothing OOMs, then stop. Don't chase real accuracy here.

`configs/baseline_kaggle.yaml` — **the actual training run, on Kaggle**, matching the paper's reported hyperparameters:
```yaml
frames_per_clip: 15
frame_size: 224
batch_size: 8            # matches the paper; Kaggle's 16GB comfortably fits this
optimizer: adam
lr: 1e-4
lr_decay_factor: 0.1
lr_decay_patience: 4
epochs: 50
early_stop_patience: 10
data_root: /kaggle/input/cricshot10k
checkpoint_dir: /kaggle/working/checkpoints/baseline
```
Run this one on Kaggle (`!python src/train.py --config configs/baseline_kaggle.yaml`, per §12.2/§13). Log Top-1/Top-2/Top-3 accuracy, precision/recall/F1 per class, and save a confusion matrix — you'll directly compare all of these against your new models later. This same local-smoke-test / Kaggle-real-run split applies to every model in §6–§8 below: each gets a plain `configs/<name>.yaml` for local smoke-testing and a `configs/<name>_kaggle.yaml` for the real run.

---

## 6. Day 3–4 — Multi-Scale CNN

Three crops per frame (full body, upper-body/hands, bat-face), each through the **same shared-weight** EfficientNetV2-S (shared weights, not three separate backbones — this keeps parameter count and VRAM manageable):

`src/models/multiscale_cnn.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbone import FrameEncoder

class MultiScaleEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = FrameEncoder(pretrained=pretrained)  # shared weights across scales
        self.out_dim = self.encoder.out_dim * 3

    def _crop(self, x, size):
        # center crop to `size`, then resize back to 224 for the shared backbone
        _, _, h, w = x.shape
        top = (h - size) // 2
        left = (w - size) // 2
        patch = x[:, :, top:top+size, left:left+size]
        return F.interpolate(patch, size=224, mode="bilinear", align_corners=False)

    def forward(self, x):
        # x: (B, C, 224, 224) — full frame already cropped/segmented by CricShotNet's preprocessing
        full = self.encoder(x)
        mid  = self.encoder(self._crop(x, 112))
        fine = self.encoder(self._crop(x, 56))
        return torch.cat([full, mid, fine], dim=1)  # (B, out_dim*3)
```

**VRAM note:** three forward passes per frame roughly triples compute. On 12GB, drop `batch_size` to 4 and use gradient accumulation of 2 to keep the same effective batch size as the baseline:
```yaml
batch_size: 4
grad_accum_steps: 2
```

Wire this into the existing GRU head for now (swap in the transformer on Day 5–6) and train. Compare against baseline confusion matrix — you're specifically checking whether Pull-vs-Hook and Lofted-Legside-vs-Offside error rates drop.

---

## 7. Day 5–6 — Lightweight Temporal Transformer (replaces GRU)

`src/models/temporal_transformer.py`:
```python
import torch
import torch.nn as nn

class TemporalTransformerHead(nn.Module):
    def __init__(self, in_dim=1280, num_layers=2, num_heads=4, num_classes=15, num_frames=15):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, in_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads, dim_feedforward=in_dim * 2,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: (B, T=15, in_dim)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)  # temporal mean pool
        x = self.bn(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

Keep this **shallow (2–4 layers, 4 heads)** — a full ViT-scale transformer will overfit on 10,086 clips. Train it on top of the *single-scale* backbone first (isolate the transformer's effect before combining with multi-scale), then on Day 7 combine both upgrades.

---

## 8. Day 7 — Combined Model

`src/models/combined_model.py`:
```python
import torch.nn as nn
from src.models.multiscale_cnn import MultiScaleEncoder
from src.models.temporal_transformer import TemporalTransformerHead

class CombinedModel(nn.Module):
    def __init__(self, num_classes=15, num_frames=15, pretrained=True):
        super().__init__()
        self.encoder = MultiScaleEncoder(pretrained=pretrained)
        self.head = TemporalTransformerHead(
            in_dim=self.encoder.out_dim, num_classes=num_classes, num_frames=num_frames
        )

    def forward(self, clip):
        # clip: (B, T, C, H, W)
        b, t, c, h, w = clip.shape
        feats = self.encoder(clip.view(b * t, c, h, w))
        feats = feats.view(b, t, -1)
        return self.head(feats)
```

**Use the `get_device()` / `get_amp_settings()` pair from §5** — it already picks bf16-no-scaler on Arc B580 and fp16-with-scaler on Kaggle's T4/P100 automatically. Don't hardcode a device string here.

If you still hit OOM at `batch_size=4` (more likely locally on 12GB than on Kaggle's 16GB cards), add `torch.utils.checkpoint` around the multi-scale encoder's forward pass before dropping batch size further.

---

## 9. Day 8 — Ablation, Confusion Matrix, Error Analysis

Run all four configurations and put them in one table (this table alone justifies the paper):

| Model | Top-1 Acc | Pull→Hook error | Lofted Legside→Offside error |
|---|---|---|---|
| Baseline (EfficientNetV2-S + GRU) — reproduced | your number | 13%* | 8%* |
| + Multi-Scale CNN only | | | |
| + Temporal Transformer only | | | |
| Combined (yours) | | | |

*\*from the original paper's reported confusion matrix — your reproduction should land close to these.*

Also produce, matching the original paper's format so reviewers can directly compare:
- Per-class precision/recall/F1/AUC-ROC table
- Confusion matrix heatmap
- 2–3 qualitative misclassification examples with frames shown (same style as the original paper's Figure 15)

If you have any slack time left today, run a paired significance test (McNemar's test on the test-set predictions, baseline vs. combined model) — conference reviewers increasingly ask for this and it's a 20-minute `scipy` script, not a new experiment.

---

## 10. Day 9 — Write the Paper

**Positioning:** this is an extension paper — "We build on CricShotNet (Dihan et al., IEEE Access 2026) and its CricShot10k dataset, addressing its two dominant residual error modes." That framing is honest, defensible, and gives you an automatic, strong baseline comparison.

Suggested section outline:
1. **Introduction** — motivate from CricShotNet's own documented weaknesses (Pull/Hook and Lofted Legside/Offside confusion), not from scratch
2. **Related Work** — CricShot10k/CricShotNet + the video-action-recognition works it already cites (LRCN, CricShotClassify, pose-based approaches) + your architectural additions' origin papers (multi-scale representation learning, transformer-based temporal modeling)
3. **Method** — architecture diagrams for Multi-Scale CNN branch and Temporal Transformer head; state clearly you reuse CricShotNet's cropping+segmentation preprocessing unchanged
4. **Experimental Setup** — CricShot10k dataset (cite exact stats: 10,086 clips, 15 classes), same train/val/test split as original authors, hardware disclosure (Intel Arc B580 12GB — worth stating plainly, it's a point of practical-reproducibility interest)
5. **Results** — the ablation table from §9, per-class metrics, confusion matrix comparison
6. **Error Analysis** — did the specific errors you targeted actually shrink? This is your strongest evidence of a real contribution rather than noise
7. **Limitations / Future Work** — explicitly name GNN skeleton stream and 3D CNN/SlowFast as promising directions not explored here due to scope — this is normal and expected, not a weakness
8. **Conclusion**

---

## 11. Where GNN / 3D CNN Would Slot In (if you insist, or for future work)

Only attempt these if Days 1–7 finish early:

- **GNN stream**: needs a pose-extraction pass (Mediapipe or YOLO-pose) over all 10,086 clips first (a half-day of preprocessing alone), then an ST-GCN implementation, then a fusion layer with your CNN stream. Realistic minimum: 2.5 extra days. Do not start this unless Day 6 finished with a full day to spare.
- **3D CNN / SlowFast**: your own reference doc already flags this needs 24GB+ for the full SlowFast config — on 12GB you'd be limited to a heavily reduced C3D/I3D variant, which per the same doc may not beat your GRU baseline at this dataset size. Low expected payoff for the risk in a 10-day window.

Both are excellent "Future Work" sentences. Neither is worth risking your submission for.

---

## 12. Local (Arc B580) ↔ Kaggle (CUDA) Hybrid Workflow

**Division of labor:**
- **Local Arc B580** — write code, debug, and run tiny smoke tests only (1–2 epochs on a ~50-clip subset, batch_size=2, just to confirm the forward/backward pass runs and loss drops). Never run a full training job locally — that's what burns your limited days waiting on a 12GB card when a 16GB Kaggle GPU is sitting there free.
- **Kaggle** — every real training run (baseline reproduction, multiscale, transformer, combined, and any hyperparameter reruns).

### 12.1 One-time setup (do this on Day 1, alongside §2)

1. **Create a GitHub repo** for `cricshot-improve/` and push your local folder structure to it (private is fine; make public only if you want easy Kaggle "Add via GitHub" linking — a private repo works too via a personal access token in the clone URL).
   ```bash
   git init
   git remote add origin https://github.com/<you>/cricshot-improve.git
   git add . && git commit -m "init"
   git push -u origin main
   ```
2. **Upload the CricShot10k clips as a Kaggle Dataset** (one-time, not per-session). Kaggle → Create → New Dataset → upload the `data/raw/` folder (or a zipped version). This avoids re-downloading 10k clips every session and doesn't count against your GPU quota.
3. **Verify Kaggle GPU quota**: free tier is ~30 GPU-hours/week, sessions cap at 12 hours, and you get either 1×P100 (16GB) or 2×T4 (16GB each). This is generous but not infinite for 4 model configs — see the budget table below.

### 12.2 Every work session after that

**Local (fast loop):**
```bash
# edit code, then smoke-test
python src/train.py --config configs/baseline.yaml --smoke_test --epochs 1
git add . && git commit -m "add multiscale branch" && git push
```

**On Kaggle (real training):**
1. New Notebook → Settings → Accelerator: GPU → Internet: On
2. Add your uploaded CricShot10k dataset as a Notebook Input
3. Pull your latest code:
   ```python
   !git clone https://github.com/<you>/cricshot-improve.git
   %cd cricshot-improve
   !pip install -q -r requirements.txt   # torch/torchvision already preinstalled with CUDA — don't reinstall those, just the extras
   ```
4. Point the dataset path at the Kaggle input mount (`/kaggle/input/cricshot10k/...`) via the config or a `--data_root` CLI flag — don't hardcode your local `data/raw/` path anywhere.
5. Run training. `get_device()` will silently pick `cuda` here with zero code changes.
6. **Save checkpoints to `/kaggle/working/`** every epoch, then **"Save Version"** at the end of the session — this persists `/kaggle/working/` as a new dataset/output you can attach as an *input* to your next session to resume from, since sessions don't survive shutdown.

### 12.3 GPU-hour budget (don't blow the weekly 30-hour cap)

Rough estimate assuming ~10,086 clips, batch_size 8–16 on a 16GB Kaggle GPU, ~30 epochs to convergence (matches the original paper's observation that runs converge by ~30 epochs):

| Run | Est. hours | Notes |
|---|---|---|
| Baseline reproduction | ~2–3h | Small model (GRU head), fastest of the four |
| Multi-Scale CNN | ~5–6h | ~3× forward-pass cost per frame |
| Temporal Transformer (single-scale) | ~3–4h | Similar cost to baseline, transformer head is cheap |
| Combined model | ~6–7h | Sum of multiscale + transformer overhead |
| **Total** | **~16–20h** | Leaves headroom in one 30h week for ablation reruns/tuning |

This fits inside **one weekly quota reset** if you start Kaggle runs by Day 3–4 as planned. If Kaggle's queue is busy and a session gets preempted, that's exactly why checkpoint-and-resume (§12.2 step 6) matters — don't restart from epoch 0.

### 12.4 Kaggle-specific config

Kaggle's T4/P100 (16GB) has more headroom than your local 12GB Arc card, so give it a separate config rather than reusing `baseline.yaml` unchanged:

`configs/multiscale_kaggle.yaml`:
```yaml
frames_per_clip: 15
frame_size: 224
batch_size: 8            # can go higher than the local 4+accum config
grad_accum_steps: 1
optimizer: adam
lr: 1e-4
lr_decay_factor: 0.1
lr_decay_patience: 4
epochs: 50
early_stop_patience: 10
data_root: /kaggle/input/cricshot10k
checkpoint_dir: /kaggle/working/checkpoints
```

Add a `--data_root` and `--checkpoint_dir` CLI override in `train.py` so the same script takes either the local paths or the Kaggle paths without editing code — the only thing that should differ between machines is which config file you pass in.

---

## 13. Quick Command Reference

```bash
# --- LOCAL (Arc B580) — code + smoke test only ---
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"

python src/train.py --config configs/baseline.yaml --smoke_test --epochs 1
git add . && git commit -m "wip" && git push

# --- KAGGLE (CUDA) — real training, run inside a notebook cell ---
!git clone https://github.com/<you>/cricshot-improve.git
%cd cricshot-improve
!pip install -q -r requirements.txt   # torch/CUDA already present on Kaggle images

!python src/train.py --config configs/baseline_kaggle.yaml
!python src/train.py --config configs/multiscale_kaggle.yaml
!python src/train.py --config configs/transformer_kaggle.yaml
!python src/train.py --config configs/combined_kaggle.yaml

!python src/evaluate.py --checkpoint /kaggle/working/checkpoints/combined/best.pt --split test
```

---

*Plan assumes solo/small-team effort on one Arc B580. Adjust day allocations if you have collaborators who can parallelize dataset prep and model implementation.*
