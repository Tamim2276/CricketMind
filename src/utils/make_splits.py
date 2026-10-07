"""
Builds train/val/test CSV splits for CricShot10k.

The official repo (CricShot10k/Codes/Model_Layer_CNN_RNN_Training_Testing.ipynb)
expects pre-made train/test/validation folders on disk but contains no code
that actually produces the split, so there's nothing to reproduce exactly.
This builds my own split instead, following the same principles the paper
describes: stratified by class, 70/15/15, and no source-video leakage across
splits
"""
import os
import re
import random
import argparse
from collections import defaultdict

import pandas as pd

VIDEO_EXTS = {".avi", ".mp4", ".mov"}


def group_id(filename: str) -> str:
    m = re.match(r"(vid\d+)_", filename)
    return m.group(1) if m else filename


def build_splits(data_root, out_dir, train_frac=0.70, val_frac=0.15, seed=42):
    random.seed(seed)
    classes = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )

    rows = {"train": [], "val": [], "test": []}

    for class_name in classes:
        class_dir = os.path.join(data_root, class_name)
        files = sorted(
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        )

        groups = defaultdict(list)
        for f in files:
            groups[group_id(f)].append(f)

        group_ids = sorted(groups.keys())
        random.shuffle(group_ids)

        n_groups = len(group_ids)
        n_train = round(n_groups * train_frac)
        n_val = round(n_groups * val_frac)

        train_groups = set(group_ids[:n_train])
        val_groups = set(group_ids[n_train:n_train + n_val])
        test_groups = set(group_ids[n_train + n_val:])

        for gid, group_files in groups.items():
            split = (
                "train" if gid in train_groups
                else "val" if gid in val_groups
                else "test"
            )
            for f in group_files:
                rows[split].append({
                    "video_path": os.path.join(class_dir, f).replace("\\", "/"),
                    "label": class_name,
                    "class_name": class_name,
                })

    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.DataFrame(rows[split]).sample(frac=1, random_state=seed).reset_index(drop=True)
        df.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        print(f"{split}: {len(df)} clips")

    total = sum(len(v) for v in rows.values())
    print(f"total: {total} clips across {len(classes)} classes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/CricShoot10kShootDataset")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_splits(args.data_root, args.out_dir, args.train_frac, args.val_frac, args.seed)
