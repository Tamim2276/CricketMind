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

from src.data.augment import ClipFlip
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD, PROCESSED
from src.models.encoder import DEFAULT_BACKBONE, DEFAULT_SIZE, FrameEncoder
from src.utils import progress
from src.utils.checkpoint import append_result, atomic_save, read_results
from src.utils.device import describe, get_amp_settings, get_device

__all__ = ["clip_list", "grid_shape", "encode_clip", "extract", "out_path",
           "verify", "manifest_path", "pool_grid", "derive", "SILU_FLOOR"]

FEATURES = os.path.join("data", "features")
DEFAULT_NAME = "effnetv2s"
FP16_MAX = 65504.0          # anything larger becomes inf, silently
SILU_FLOOR = -0.278464556   # min of x*sigmoid(x); EfficientNet ends in SiLU


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

    # .contiguous() is load-bearing, and not for speed. dataset.py builds its
    # tensor with .permute(0, 3, 1, 2), which makes it channels_last, and that
    # format propagates. The DataLoader then silently decides it: at
    # --workers 0 the batch stays channels_last, at --workers 4 the shared
    # memory collate rewrites it to NCHW. oneDNN picks a different kernel for
    # each, so the cache came out different by 0.39 depending on a flag that
    # is supposed to only affect speed. Pinning the format makes --workers a
    # pure performance knob again.
    x = x.contiguous()

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


def manifest_path(out_dir, view="grid"):
    """One manifest per output view, or the second run skips everything.

    `build_dataset --variants seg` was a silent no-op for exactly this reason:
    it shared a manifest with the run that had already finished, so every clip
    read as done and nothing was written. One ledger per thing being built.
    """
    if view == "grid":
        return os.path.join(out_dir, "manifest.jsonl")
    return os.path.join(out_dir, "manifest_%s.jsonl" % view)


class _Clips(Dataset):
    """Every processed clip, normalised. Not split-aware, by design."""

    def __init__(self, root, rels, mirror=False):
        self.root = root
        self.rels = rels
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        # ClipFlip(1.0) rather than a second torch.flip: one definition of
        # "mirrored" in the codebase, and it is the one training uses
        if mirror:
            self.flip = ClipFlip(1.0)
        else:
            self.flip = None

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, i):
        path = os.path.join(self.root,
                            self.rels[i].replace("/", os.sep) + ".npy")
        frames = np.load(path)                      # (T, 224, 224, 3) uint8
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
        x = (x - self.mean) / self.std
        if self.flip is not None:
            x = self.flip(x)
        return x, i


def extract(variant="box", backbone=DEFAULT_BACKBONE, name=None,
            out_dir=None, processed=PROCESSED, limit=None, batch_size=4,
            workers=0, fp32=False, mirror=False, device=None, quiet=False):
    """One pass over every clip. Re-running skips what the manifest lists.

    `mirror=True` encodes the left-right flipped clip instead, into
    `grid_mirror/`. It has to be a second pass: a flipped clip has its own
    features, and flipping the cached grid instead gives cosine 0.41 against
    the real thing -- measured, not assumed. Because `ClipFlip` is a binary
    choice, caching both halves is **exact** rather than an approximation.
    """
    name = name or DEFAULT_NAME
    out_dir = out_dir or os.path.join(FEATURES, name)
    root = os.path.join(processed, variant)
    if mirror:
        view = "grid_mirror"
    else:
        view = "grid"
    manifest = manifest_path(out_dir, view)

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
        if mirror:
            print("MIRRORED: left-right flipped clips, for flip augmentation")
        print(f"grid {grid}  out_dim {encoder.out_dim}  "
              f"-> {out_path(out_dir, view, '<class>/<clip>')}")
        if fp32:
            print("fp32: autocast off, so these will not match a bf16 run")
        print(flush=True)

    if not todo:
        return 0

    loader = DataLoader(_Clips(root, todo, mirror=mirror),
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers)

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
                path = out_path(out_dir, view, rel)
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


def verify(variant="box", backbone=DEFAULT_BACKBONE, name=None, out_dir=None,
           processed=PROCESSED, clips=20, seed=0, device=None, quiet=False):
    """Check an existing cache against the encoder that is supposed to have
    written it. Returns the number of checks that failed.

    **Not a training run.** A wrong cache still trains to *some* number, so
    accuracy cannot tell "correct" from "wrong in a way the optimiser routed
    around". These compare tensors instead.
    """
    import random

    name = name or DEFAULT_NAME
    out_dir = out_dir or os.path.join(FEATURES, name)
    root = os.path.join(processed, variant)

    rows = []
    for row in read_results(os.path.join(out_dir, "manifest.jsonl")):
        if row.get("ok"):
            rows.append(row["clip"])
    if not rows:
        raise SystemExit(f"no cache to verify at {out_dir}")
    random.Random(seed).shuffle(rows)
    rows = rows[:clips]

    device = device or get_device()
    amp = get_amp_settings(device)
    encoder = FrameEncoder(backbone, pretrained=True, freeze=True,
                           pool="flatten").to(device).eval()
    grid = grid_shape(encoder)
    data = _Clips(root, rows)
    bad = 0

    # ---- check 1: the files are what encode_clip produces -----------------
    # Bit-exact, not approximate. The forward pass is deterministic and
    # encode_clip pins the memory format, so a correct cache reproduces
    # exactly -- at any --workers and any batch size. Anything else means the
    # cache was built by different code or different settings.
    worst_abs = 0.0
    peak = 0.0
    exact = 0
    for start in range(0, len(rows), 4):
        chunk = []
        for i in range(start, min(start + 4, len(rows))):
            chunk.append(data[i][0])
        feats = encode_clip(torch.stack(chunk), encoder, device, amp, grid)
        for j in range(feats.shape[0]):
            rel = rows[start + j]
            saved = torch.load(out_path(out_dir, "grid", rel),
                               map_location="cpu")
            fresh = feats[j]
            if torch.equal(saved, fresh):
                exact += 1
            gap = (saved.float() - fresh.float()).abs()
            worst_abs = max(worst_abs, float(gap.max()))
            peak = max(peak, float(fresh.float().abs().max()))

    ok_1 = exact == len(rows)
    bad += not ok_1

    # ---- check 2: the reshape order, by an independent route --------------
    # In fp32 and with no autocast, because this tests the (C, H, W) vs
    # (H, W, C) convention -- a property of the code, not of the cache -- and
    # bf16 noise would swamp a 1e-7 answer.
    pooled = FrameEncoder(backbone, pretrained=True, freeze=True,
                          pool="avg").to(device).eval()
    worst_fold = 0.0
    with torch.no_grad():
        for i in range(min(3, len(rows))):
            x = data[i][0][None].to(device)
            flat = encoder(x).float()
            derived = flat.unflatten(-1, grid).mean(dim=(-1, -2))
            real = pooled(x).float()
            worst_fold = max(worst_fold,
                             float((derived - real).abs().max()))

    ok_2 = worst_fold < 1e-4
    bad += not ok_2

    # ---- check 3: the activation floor ------------------------------------
    # EfficientNetV2-S ends in SiLU, whose minimum is -0.278464556. A correct
    # cache presses against that and never goes under. A hard 0.0 would mean
    # a ReLU backbone; far below means a pre-activation layer.
    low = 0.0
    high = 0.0
    flat_zero = 0
    for rel in rows:
        t = torch.load(out_path(out_dir, "grid", rel), map_location="cpu")
        low = min(low, float(t.min()))
        high = max(high, float(t.max()))
        if float(t.min()) == 0.0:
            flat_zero += 1

    ok_3 = low >= -0.28 and flat_zero == 0
    bad += not ok_3

    if not quiet:
        def mark(ok):
            if ok:
                return "PASS"
            return "FAIL"

        print(f"verifying {len(rows)} clips from {out_dir}")
        print(f"  1 round-trip against encode_clip   {mark(ok_1)}   "
              f"{exact}/{len(rows)} bit-exact  max abs gap {worst_abs:.4f}  "
              f"(values reach {peak:.1f})")
        if not ok_1 and worst_abs < 0.05 * max(peak, 1e-9):
            print("      the gap is bf16-sized, so the cache is not wrong -- "
                  "it was built with different settings (--fp32? older code?)")
        print(f"  2 reshape order, fp32, vs pool=avg {mark(ok_2)}   "
              f"max abs {worst_fold:.3e}  (float32 noise is ~1e-7)")
        print(f"  3 activation floor                 {mark(ok_3)}   "
              f"min {low:.6f}  max {high:.2f}  "
              f"(SiLU floor -0.278465; {flat_zero} tensors at a hard 0.0)")
        print()
        if bad:
            print(f"{bad} of 3 FAILED -- do not train on this cache")
        else:
            print("all 3 passed")
    return bad


def pool_grid(grid):
    """(T, C, H, W) fp16 -> (T, C) fp16: the average over the spatial grid.

    Two details, each of which still produces a valid-looking tensor when
    wrong. **Sum in float32**: 49 fp16 values added in fp16 pile up rounding
    error, so up-cast, average, then store as fp16. **Reduce over the last two
    axes**: dim=(-1, -2) is the 7x7. dim=(1, 2) would give (T, 7) -- it saves
    without complaint and means nothing.
    """
    if grid.dim() != 4:
        raise ValueError(f"expected (T, C, H, W), got {tuple(grid.shape)}")
    return grid.float().mean(dim=(-1, -2)).half()


def derive(view="avg", name=None, out_dir=None, quiet=False):
    """The 1280-wide view, made from grid/ on the CPU. Returns failures.

    No encoder and no second pass over the video: 8.5 showed averaging the
    cached grid equals what pool="avg" computes, to 3.6e-07. Both halves are
    derived, grid/ -> avg/ and grid_mirror/ -> avg_mirror/, so flip still
    works on the small view. Each output has its own manifest, and reads its
    work list from the source's manifest, so only finished files are used.
    """
    if view != "avg":
        raise ValueError(f"only the avg view can be derived, got {view!r}")
    name = name or DEFAULT_NAME
    out_dir = out_dir or os.path.join(FEATURES, name)
    tty = sys.stdout.isatty()
    failed = 0

    for source in ("grid", "grid_mirror"):
        if not os.path.isdir(os.path.join(out_dir, source)):
            if source == "grid":
                raise SystemExit(f"no grid cache at {out_dir}; run extract first")
            if not quiet:
                print(f"{source}/ not there, skipping its half")
            continue
        target = source.replace("grid", view)

        rels = []
        for row in read_results(manifest_path(out_dir, source)):
            if row.get("ok"):
                rels.append(row["clip"])
        manifest = manifest_path(out_dir, target)
        done = set()
        for row in read_results(manifest):
            done.add(row["clip"])
        todo = []
        for rel in rels:
            if rel not in done:
                todo.append(rel)

        if not quiet:
            print(f"{source}/ -> {target}/   {len(rels)} clips, "
                  f"{len(done)} already done, {len(todo)} to go", flush=True)

        started = time.time()
        for n, rel in enumerate(todo, 1):
            try:
                grid = torch.load(out_path(out_dir, source, rel),
                                  map_location="cpu")
                pooled = pool_grid(grid)
                path = out_path(out_dir, target, rel)
                atomic_save(pooled, path)
                if n == 1:
                    _check_size(path, pooled)
                row = {"clip": rel, "ok": True, "shape": list(pooled.shape)}
            except (OSError, RuntimeError, ValueError) as err:
                row = {"clip": rel, "ok": False,
                       "reason": f"{type(err).__name__}: {err}"}
                failed += 1
            append_result(row, manifest)

            if tty:
                rate = n / max(time.time() - started, 1e-9)
                left = (len(todo) - n) / max(rate, 1e-9) / 60
                print(progress.line(n, len(todo),
                                    f"{rate:5.0f} clips/s  ~{left:4.1f} min left",
                                    prefix="  ", tty=True), end="", flush=True)
            elif n % 1000 == 0 or n == len(todo):
                print(progress.line(n, len(todo), "", prefix="  ", tty=False),
                      flush=True)
        if todo and not quiet:
            print(f"\n  done in {(time.time() - started) / 60:.1f} min\n")

    if not quiet:
        print(f"{failed} failed")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="check an existing cache instead of writing one")
    ap.add_argument("--clips", type=int, default=20,
                    help="--verify only: how many clips to check")
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
    ap.add_argument("--mirror", action="store_true",
                    help="encode the flipped clips, into grid_mirror/ (8.7)")
    ap.add_argument("--derive", choices=("avg",),
                    help="make the pooled view from grid/, no GPU (8.8)")
    a = ap.parse_args(argv)

    opts = vars(a)
    do_verify = opts.pop("verify")
    checked = opts.pop("clips")
    pooled_view = opts.pop("derive")
    if pooled_view:
        return derive(view=pooled_view, name=opts["name"],
                      out_dir=opts["out_dir"])
    if do_verify:
        for unused in ("limit", "batch_size", "workers", "fp32", "mirror"):
            opts.pop(unused)
        return verify(clips=checked, **opts)
    return extract(**opts)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
