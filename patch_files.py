"""
Run this ONCE from the project root to fix both issues:
    python patch_files.py

What it does:
  1. Patches src/train.py  — adds encoding='utf-8' to open() in load_config()
  2. Rewrites configs/mimic_author.yaml — ASCII-only, correct author settings
"""
import os
import re

# ─────────────────────────────────────────────────────────────────
# FIX 1: src/train.py  — add encoding='utf-8' to load_config()
# ─────────────────────────────────────────────────────────────────
train_path = os.path.join("src", "train.py")

with open(train_path, encoding="utf-8") as f:
    train_src = f.read()

OLD = "with open(path) as f:"
NEW = "with open(path, encoding='utf-8') as f:"

if OLD in train_src:
    train_src = train_src.replace(OLD, NEW, 1)
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_src)
    print(f"[OK] Patched {train_path}: added encoding='utf-8' to load_config()")
elif NEW in train_src:
    print(f"[OK] {train_path} already has encoding='utf-8' — no change needed.")
else:
    print(f"[!!] Could not find the open() line in {train_path}. "
          f"Please add  encoding='utf-8'  manually to the open() call "
          f"inside load_config().")

# ─────────────────────────────────────────────────────────────────
# FIX 2: configs/mimic_author.yaml — ASCII-only rewrite
# ─────────────────────────────────────────────────────────────────
yaml_path = os.path.join("configs", "mimic_author.yaml")

yaml_content = """\
# configs/mimic_author.yaml
# Exact replication of the author's Keras training notebook.
#
# Key findings from Model_Layer_CNN_RNN_Training_Testing.ipynb:
#
# Architecture:
#   GRU(128) -> BatchNorm -> Dense(1024, relu) -> Dense(15, softmax)
#   NO Dropout anywhere   (dropout=0.0 passed to BaselineModel)
#
# Training (from actual notebook, NOT the paper text):
#   batch_size   = 4     (notebook Cell 4 -- paper says 8, notebook says 4)
#   epochs       = 100   (notebook Cell 9 -- paper says 50, notebook says 100)
#   lr           = 1e-4  (single LR for ALL layers)
#   weight_decay = 0.0   (Keras Adam default)
#   label_smooth = 0.0   (no label smoothing)
#
# Scheduler: ReduceLROnPlateau(factor=0.1, patience=4) -- NOT CosineAnnealing
#
# Augmentation: horizontal flip only (offline), no MixUp/ColorJitter/Erasing

model_type: "baseline"
num_classes: 15
frames_per_clip: 15
pretrained: true

# DataLoader
batch_size: 4
grad_accum_steps: 1
num_workers: 0
cache: false
prefetch_factor: 2

# Optimizer
lr: 0.0001
backbone_lr_multiplier: 1.0
weight_decay: 0.0
label_smoothing: 0.0

# Scheduler
scheduler: reduce_lr_on_plateau
lr_decay_factor: 0.1
lr_decay_patience: 4

# Training
epochs: 100
early_stop_patience: 10

# Architecture flags
mimic_author: true

# Paths
splits_dir: "data/splits"
processed_root: "data/processed"
checkpoint_dir: "experiments/mimic_author/checkpoints"
figure_dir: "experiments/mimic_author/figures"
"""

os.makedirs("configs", exist_ok=True)
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)
print(f"[OK] Rewrote {yaml_path} with ASCII-only content.")

# ─────────────────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────────────────
import yaml
try:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[OK] YAML parses cleanly. Keys: {list(cfg.keys())}")
    assert cfg["mimic_author"]      is True,   "mimic_author should be True"
    assert cfg["scheduler"]         == "reduce_lr_on_plateau"
    assert cfg["dropout"] if "dropout" in cfg else True  # optional key
    assert cfg["weight_decay"]      == 0.0
    assert cfg["label_smoothing"]   == 0.0
    assert cfg["backbone_lr_multiplier"] == 1.0
    print("[OK] All config values verified.")
except Exception as e:
    print(f"[!!] Verification failed: {e}")

print()
print("Done. Now run:")
print("  python -m src.train --config configs/mimic_author.yaml --smoke_test")
