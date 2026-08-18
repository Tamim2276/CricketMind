"""
Dataset loader for preprocessed CricShot10k clips.

Reads from data/splits/{train,val,test}.csv and loads clips from
data/processed/{split}/{class_name}/*.pt  (output of build_dataset.py).

Each .pt file is a (T, H, W, C) uint8 RGB tensor — no video decode overhead.
Falls back to .avi video files for backward compatibility.

Each item: (clip_tensor, label_idx)
  clip_tensor: (T, C, H, W) float32, ImageNet-normalized
  label_idx:   int
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet stats — same normalization the EfficientNetV2-S pretrained weights expect.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Cover Drive", "Defensive", "Down The Wicket", "Flick", "Hook",
    "Late Cut", "Lofted Legside", "Lofted Offside", "Pull",
    "Reverse Sweep", "Scoop", "Square Cut", "Straight Drive",
    "Sweep", "Upper Cut",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def _default_frame_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


class CricShotDataset(Dataset):
    """
    Args:
        csv_path:       path to train.csv / val.csv / test.csv
        processed_root: root of preprocessed clips, e.g. "data/processed"
        split:          "train" | "val" | "test"  (used to locate files)
        num_frames:     must match the value used during preprocessing (default 15)
        transform:      per-frame torchvision transform; None → uses default
        smoke_test:     if True, only loads up to 50 existing clips
        cache:          if True, load all clips into RAM on first access
                        (eliminates disk I/O for epoch 2+; needs ~3 GB RAM per 1k clips)
    """

    def __init__(
        self,
        csv_path: str,
        processed_root: str,
        split: str,
        num_frames: int = 15,
        transform=None,
        smoke_test: bool = False,
        cache: bool = False,
    ):
        self.processed_root = processed_root
        self.split          = split
        self.num_frames     = num_frames
        self.transform      = transform or _default_frame_transform(train=(split == "train"))
        self.cache_enabled  = cache
        self._cache: dict   = {}  # idx -> (T, H, W, C) uint8 tensor

        df = pd.read_csv(csv_path)

        # Filter for existing files FIRST, then cap for smoke_test.
        # (Capping first on a shuffled 7k-row CSV almost never hits the tiny
        # preprocessed subset when only --limit N clips were processed.)
        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            class_name = row["class_name"]
            filename   = os.path.basename(row["video_path"])
            stem       = os.path.splitext(filename)[0]
            label_idx  = CLASS_TO_IDX.get(class_name, -1)
            if label_idx == -1:
                continue

            # Prefer .pt (fast tensor), fall back to .avi (legacy video)
            pt_path  = os.path.join(processed_root, split, class_name, stem + ".pt")
            avi_path = os.path.join(processed_root, split, class_name, filename)

            if os.path.exists(pt_path):
                self.samples.append((pt_path, label_idx))
            elif os.path.exists(avi_path):
                self.samples.append((avi_path, label_idx))
            else:
                missing += 1

        if missing and not smoke_test:
            print(f"[CricShotDataset/{split}] WARNING: {missing} clips not found "
                  f"in processed_root '{processed_root}'. "
                  f"Run: python -m src.preprocess.build_dataset")

        # Cap AFTER filtering so smoke_test always finds real clips
        if smoke_test and len(self.samples) > 50:
            self.samples = self.samples[:50]

        print(f"[CricShotDataset/{split}] {len(self.samples)} clips loaded"
              f"{' (smoke_test)' if smoke_test else ''}"
              f"{' [cache=ON]' if cache else ''}.")

        # Pre-load into RAM if cache=True
        if cache and len(self.samples) > 0:
            self._preload_cache()

    def _preload_cache(self):
        from tqdm import tqdm
        bar = tqdm(
            enumerate(self.samples),
            total=len(self.samples),
            desc=f"Caching {self.split}",
            unit="clip",
            dynamic_ncols=True,
            colour="cyan",
        )
        for idx, (path, _) in bar:
            self._cache[idx] = self._load_raw(path)
        print(f"[CricShotDataset/{self.split}] Cache ready — "
              f"{len(self._cache)} clips in RAM.")

    def _load_raw(self, path) -> torch.Tensor:
        """Load clip as (T, H, W, C) uint8 tensor regardless of file type."""
        if path.endswith(".pt"):
            return torch.load(path, weights_only=True)  # (T, H, W, C) uint8 RGB
        else:
            return self._load_avi_raw(path)

    def _load_avi_raw(self, path) -> torch.Tensor:
        """Fallback: decode .avi and return (T, H, W, C) uint8 RGB tensor."""
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            return torch.zeros(self.num_frames, 224, 224, 3, dtype=torch.uint8)

        # Pad or truncate to exactly num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]
        return torch.from_numpy(np.stack(frames, axis=0))  # (T, H, W, C)

    def _apply_transform(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply per-frame transform to (T, H, W, C) uint8 → (T, C, H, W) float32."""
        frames = []
        for t in range(raw.shape[0]):
            frame_np = raw[t].numpy()  # (H, W, C) uint8 RGB — ToPILImage accepts this
            frames.append(self.transform(frame_np))
        return torch.stack(frames, dim=0)  # (T, C, H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        if idx in self._cache:
            raw = self._cache[idx]
        else:
            raw = self._load_raw(path)
            if self.cache_enabled:
                self._cache[idx] = raw  # lazy cache on first access

        clip = self._apply_transform(raw)  # (T, C, H, W)
        return clip, label
