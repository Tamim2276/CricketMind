"""Run the frozen encoder over every clip once, and keep the answers.

The waste this removes: in `mimic_author` the backbone runs 45 times over the
same 9,880 clips. When it is frozen its output for a clip is the *same tensor*
on epoch 1 and on epoch 45, so 44 of those passes are thrown away. Compute it
once, write it down, and Families A, D and V read tensors instead of video.

Shape contract, stated out loud because everything downstream assumes it:

    on disk      (T, 224, 224, 3) uint8      what build_dataset.py wrote
    normalised   (T, 3, 224, 224) float32    ImageNet mean/std, as dataset.py
    encoder      (T, C*H*W)       float32    pool="flatten"
    reshaped     (T, C, H, W)     float32    channel-major -- see grid_shape
    saved        (T, C, H, W)     float16

**Flat, not split-organised.** A frozen backbone's features depend only on the
clip, so one cache serves `splits_author` and `splits_grouped` both -- a split
is just a list of filenames over these files. Writing `train/val/test/` into
the path is the trap: it would force a full re-extraction every time a split
changed, and every Phase 3 experiment runs on two splits.

**Autocast stays on by default.** The baseline trained under bf16 autocast, so
the features its encoder produced were bf16. This cache stands in for that
encoder as the training loop calls it, so it is computed the way the training
loop computes it. `--fp32` turns it off to see the difference.

Measured on the Arc B580 with --workers 4: **9,880 clips in 8.5 min, 19.4
clips/s**. A bare synchronous loop managed only 12.1, and batch 8 was no faster
than batch 4 there -- so the GPU was never the bottleneck, the .npy read and
the float32 normalisation were. That is why --workers matters and a bigger
batch does not.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, PROCESSED
from src.models.encoder import DEFAULT_BACKBONE, DEFAULT_SIZE, FrameEncoder
from src.utils import progress
from src.utils.checkpoint import append_result, atomic_save, read_results
from src.utils.device import describe, get_amp_settings, get_device

__all__ = ["clip_list", "grid_shape", "encode_clip", "extract", "out_path"]

FEATURES = os.path.join("data", "features")
DEFAULT_NAME = "effnetv2s"
FP16_MAX = 65504.0          # anything larger becomes inf, silently


def clip_list(root):
    """Every processed clip under `root`, as "<class>/<name>" without .npy.

    Sorted, so two runs see the same order and the manifest is comparable.
    """
    out = []
    for cls in sorted(os.listdir(root)):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".npy"):
                out.append(f"{cls}/{name[:-4]}")
    return out


def out_path(out_dir, view, rel):
    return os.path.join(out_dir, view, rel.replace("/", os.sep) + ".pt")


def grid_shape(encoder, size=DEFAULT_SIZE):
    """(C, H, W) of the map that pool="flatten" flattens. Probed, not guessed.

    The stride differs per backbone, so deriving H from size/32 is a guess
    that would silently store a transposed cache. Instead: put a real global
    pool back for one forward pass, which gives C, and divide. The
    multiplication is then checked, so a non-square map fails loudly rather
    than rounding to something plausible.
    """
    pool = getattr(encoder.backbone, "avgpool", None)
    if pool is None:
        raise ValueError("backbone has no avgpool; grid_shape needs one")

    # the probe tensor goes wherever the weights already are, or this raises
    # a device mismatch the moment the encoder has been moved to the GPU
    where = next(encoder.parameters()).device
    encoder.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
    try:
        was_training = encoder.backbone.training
        encoder.backbone.eval()
        with torch.no_grad():
            channels = int(encoder.backbone(
                torch.zeros(1, 3, size, size, device=where)).shape[1])
        encoder.backbone.train(was_training)
    finally:
        encoder.backbone.avgpool = pool

    side = int(round(math.sqrt(encoder.out_dim / channels)))
    if channels * side * side != encoder.out_dim:
        raise ValueError(
            f"{encoder.out_dim} is not {channels} x n x n; this backbone's "
            f"feature map is not square and needs its own reshape")
    return channels, side, side


def encode_clip(x, encoder, device, amp, grid, fp32=False):
    """(T, 3, H, W) or (B, T, 3, H, W) -> (T, C, h, w) or (B, T, C, h, w) fp16.

    **The one definition of how a clip becomes a cached tensor.** The verify
    step in 8.5 calls this same function and compares its output to the files,
    so that check tests serialisation and layout rather than re-implementing
    the arithmetic and testing that instead.
    """
    single = x.dim() == 4
    if single:
        x = x[None]

    with torch.no_grad():
        if fp32 or amp.dtype is None:
            feats = encoder(x.to(device))
        else:
            with torch.autocast(device_type=amp.device_type, dtype=amp.dtype):
                feats = encoder(x.to(device))

    feats = feats.float().unflatten(-1, grid)      # (B, T, C*h*w) -> (B,T,C,h,w)
    feats = feats.to("cpu", torch.float16)
    if single:
        return feats[0]
    return feats


def _check_size(path, tensor):
    """The file must be the tensor, not the tensor's neighbours.

    Saving a view writes its whole storage, which is silent, costs a multiple
    of the disk, and only shows up as "the cache is bigger than the arithmetic
    said". One stat call on the first file turns that into an exception.
    """
    want = tensor.numel() * tensor.element_size()
    got = os.path.getsize(path)
    if got > want * 1.05 + 4096:
        raise RuntimeError(
            f"{path} is {got} bytes for a {want}-byte tensor ({got / want:.1f}x). "
            f"Something saved a view instead of a copy.")


class _Clips(Dataset):
    """Every processed clip, normalised. Not split-aware, by design."""

    def __init__(self, root, rels):
        self.root = root
        self.rels = rels
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, i):
        path = os.path.join(self.root,
                            self.rels[i].replace("/", os.sep) + ".npy")
        frames = np.load(path)                      # (T, 224, 224, 3) uint8
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
        return (x - self.mean) / self.std, i


def extract(variant="box", backbone=DEFAULT_BACKBONE, name=None,
            out_dir=None, processed=PROCESSED, limit=None, batch_size=4,
            workers=0, fp32=False, device=None, quiet=False):
    """One pass over every clip. Re-running skips what the manifest lists."""
    name = name or DEFAULT_NAME
    out_dir = out_dir or os.path.join(FEATURES, name)
    root = os.path.join(processed, variant)
    manifest = os.path.join(out_dir, "manifest.jsonl")

    rels = clip_list(root)
    if limit:
        rels = rels[:limit]

    done = set()
    for row in read_results(manifest):
        done.add(row["clip"])
    todo = []
    for rel in rels:
        if rel not in done:
            todo.append(rel)

    device = device or get_device()
    amp = get_amp_settings(device)
    encoder = FrameEncoder(backbone, pretrained=True, freeze=True,
                           pool="flatten").to(device).eval()
    grid = grid_shape(encoder)

    os.makedirs(out_dir, exist_ok=True)
    if not quiet:
        print(describe(device))
        print(f"{backbone} {variant}: {len(rels)} clips, {len(done)} already "
              f"done, {len(todo)} to go")
        print(f"grid {grid}  out_dim {encoder.out_dim}  "
              f"-> {out_path(out_dir, 'grid', '<class>/<clip>')}")
        if fp32:
            print("fp32: autocast off, so these will not match a bf16 run")
        print(flush=True)

    if not todo:
        return 0

    loader = DataLoader(_Clips(root, todo), batch_size=batch_size,
                        shuffle=False, num_workers=workers)

    tty = sys.stdout.isatty()
    started = time.time()
    failed = 0
    seen = 0
    hi = 0.0

    for x, idx in loader:
        feats = encode_clip(x, encoder, device, amp, grid, fp32=fp32)
        for j in range(feats.shape[0]):
            rel = todo[int(idx[j])]
            # .clone() is load-bearing: torch.save writes a view's WHOLE
            # storage, so feats[j] would save all `batch_size` clips into
            # every file -- 4x the bytes, measured
            one = feats[j].clone()
            top = float(one.float().abs().max())
            hi = max(hi, top)
            if not torch.isfinite(one).all():
                # fp16 overflow would be silent; measured headroom is 2278x,
                # so this firing means the backbone changed, not the clip
                row = {"clip": rel, "ok": False,
                       "reason": f"non-finite after fp16 cast, max {top:.1f}"}
                failed += 1
            else:
                path = out_path(out_dir, "grid", rel)
                atomic_save(one, path)
                if seen == 0:
                    _check_size(path, one)       # once, on the first file
                row = {"clip": rel, "ok": True,
                       "shape": list(one.shape), "max": top}
            append_result(row, manifest)
            seen += 1

        if tty:
            rate = seen / max(time.time() - started, 1e-9)
            left = (len(todo) - seen) / max(rate, 1e-9) / 60
            print(progress.line(seen, len(todo),
                                f"{rate:4.1f} clips/s  ~{left:4.1f} min left",
                                prefix="  ", tty=True), end="", flush=True)
        elif seen % 100 == 0 or seen == len(todo):
            print(progress.line(seen, len(todo), "", prefix="  ", tty=False),
                  flush=True)

    secs = time.time() - started
    if not quiet:
        print(f"\n\ndone in {secs / 60:.1f} min   {failed} failed of "
              f"{len(todo)}   {seen / max(secs, 1e-9):.1f} clips/s")
        print(f"largest activation {hi:.1f}, fp16 holds {FP16_MAX:.0f} "
              f"({FP16_MAX / max(hi, 1e-9):.0f}x headroom)")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="box", choices=("box", "seg"))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--name", help="cache directory name, default effnetv2s")
    ap.add_argument("--out", dest="out_dir", help="default data/features/<name>")
    ap.add_argument("--processed", default=PROCESSED)
    ap.add_argument("--limit", type=int, help="first N clips, for a smoke test")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="8 is no faster; the loader is the bottleneck")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--fp32", action="store_true",
                    help="no autocast; will not match a bf16-trained run")
    a = ap.parse_args(argv)
    return extract(**vars(a))


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
