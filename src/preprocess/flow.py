"""Optical flow for W01 (Day 32): how every pixel moved between frames.

torchvision's RAFT-small, between consecutive box frames:

    box clip     (15, 224, 224, 3) uint8
    flow         (14, 2, 112, 112) fp16     u (right) and v (down), in pixels
                                            at 112, ~0.7 MB a clip

Computed at 224, where RAFT sees the most detail, then shrunk to 112 for
r3d_18: shrinking a flow field halves its values too, since a 4-pixel move
at 224 is a 2-pixel move at 112.

**The mirror is made on the fly, not stored:** flip left-right and negate
u. Exact for a flow field, and it saves another 7 GB.

Day 14 found a GPU kernel that failed to repeat itself on 1% of clips, so
`--verify` recomputes a spread of clips and compares, and runs one clip
twice.

    python -m src.preprocess.flow              builds data/features/flow/
    python -m src.preprocess.flow --verify 25  recompute 25 clips, compare
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import progress
from src.utils.checkpoint import append_result, atomic_save, read_results

__all__ = ["SIZE", "FLOW_DIR", "shrink", "mirror_flow", "clip_flow",
           "load_raft", "build", "verify"]

SIZE = 112
FLOW_DIR = os.path.join("data", "features", "flow")
PAIRS = 7            # frame pairs per RAFT call


def load_raft(device, pretrained=True):
    """RAFT-small, in eval, on `device`. The weights download once."""
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    if pretrained:
        weights = Raft_Small_Weights.DEFAULT
    else:
        weights = None
    return raft_small(weights=weights).eval().to(device)


def shrink(flow, size=SIZE):
    """(N, 2, H, W) in pixels at H -> (N, 2, size, size) in pixels at size."""
    scale = size / flow.shape[-1]
    small = F.interpolate(flow, size=(size, size), mode="bilinear",
                          antialias=True, align_corners=False)
    return small * scale


def mirror_flow(flow):
    """The flow of the mirrored clip: flipped left-right, u negated."""
    flipped = torch.flip(flow, dims=[-1])
    out = flipped.clone()
    out[..., 0, :, :] = -flipped[..., 0, :, :]
    return out


def clip_flow(model, frames, device):
    """(T, H, W, 3) uint8 RGB -> (T - 1, 2, SIZE, SIZE) float32."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
    x = x.float() / 255.0 * 2.0 - 1.0       # RAFT's range, as its transforms
    first = x[:-1]
    second = x[1:]
    parts = []
    with torch.no_grad():
        for i in range(0, len(first), PAIRS):
            # RAFT refines its guess; the last one is the answer
            out = model(first[i:i + PAIRS].to(device),
                        second[i:i + PAIRS].to(device))[-1]
            parts.append(out.float().cpu())
    return shrink(torch.cat(parts))


def _manifest(out_dir):
    return os.path.join(out_dir, "manifest.jsonl")


def _finished(manifest):
    done = set()
    for row in read_results(manifest):
        if row.get("ok"):
            done.add(row["clip"])
    return done


def _path(out_dir, rel):
    return os.path.join(out_dir, "flow", rel.replace("/", os.sep) + ".pt")


def build(processed, out_dir=FLOW_DIR, limit=None, device=None, model=None,
          quiet=False, ledger=None):
    """Flow for every box clip. Resumable. Returns failures."""
    from src.extract import clip_list
    from src.utils.device import get_device

    device = device or get_device()
    model = model or load_raft(device)
    root = os.path.join(processed, "box")
    clips = clip_list(root)
    if limit:
        clips = clips[:limit]
    manifest = _manifest(out_dir)
    done = _finished(manifest)
    todo = []
    for rel in clips:
        if rel not in done:
            todo.append(rel)
    if not quiet:
        print(f"flow: {len(clips)} clips, {len(done)} done, {len(todo)} to go "
              f"on {device}", flush=True)

    tty = sys.stdout.isatty() and not quiet
    started = time.time()
    failed = 0
    for n, rel in enumerate(todo, 1):
        try:
            frames = np.load(os.path.join(root, rel.replace("/", os.sep)
                                          + ".npy"))
            flow = clip_flow(model, frames, device)
            if not torch.isfinite(flow).all():
                raise ValueError("flow is not finite")
            atomic_save(flow.half(), _path(out_dir, rel))
            size = flow.norm(dim=1)
            row = {"clip": rel, "ok": True, "flows": len(flow),
                   "mean_px": round(float(size.mean()), 3),
                   "max_px": round(float(size.max()), 2)}
        except (OSError, RuntimeError, ValueError) as err:
            row = {"clip": rel, "ok": False,
                   "reason": f"{type(err).__name__}: {err}"}
            failed += 1
        append_result(row, manifest)
        if tty:
            rate = n / max(time.time() - started, 1e-9)
            left = (len(todo) - n) / max(rate, 1e-9) / 60
            print(progress.line(n, len(todo),
                                f"{rate:5.2f} clips/s  ~{left:5.1f} min left",
                                prefix="  ", tty=True), end="", flush=True)
    if todo and not quiet:
        print(f"\n{failed} failed")
    if todo and ledger:
        secs = time.time() - started
        append_result({"kind": "flow", "cache": out_dir, "clips": len(todo),
                       "failed": failed, "minutes": round(secs / 60, 2)},
                      ledger)
    return failed


def verify(processed, out_dir=FLOW_DIR, count=25, device=None, model=None,
           tolerance=0.05):
    """Recompute `count` clips spread over the cache and compare, then run
    one clip twice. Returns the clips that disagree, with how far."""
    from src.extract import clip_list
    from src.utils.device import get_device

    device = device or get_device()
    model = model or load_raft(device)
    root = os.path.join(processed, "box")
    cached = sorted(_finished(_manifest(out_dir)))
    if not cached:
        raise SystemExit(f"no flow cache at {out_dir}")
    step = max(len(cached) // count, 1)
    bad = {}
    for rel in cached[::step][:count]:
        frames = np.load(os.path.join(root, rel.replace("/", os.sep) + ".npy"))
        again = clip_flow(model, frames, device)
        stored = torch.load(_path(out_dir, rel)).float()
        gap = float((again - stored).abs().max())
        if gap > tolerance:
            bad[rel] = round(gap, 4)
    rel = cached[0]
    frames = np.load(os.path.join(root, rel.replace("/", os.sep) + ".npy"))
    gap = float((clip_flow(model, frames, device)
                 - clip_flow(model, frames, device)).abs().max())
    if gap > 1e-4:
        bad[rel + " (twice)"] = round(gap, 6)
    return bad


def main(argv=None):
    from src.extract import CACHE_LEDGER

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", type=int, metavar="N",
                    help="recompute N clips and compare with the cache")
    ap.add_argument("--processed", default=os.path.join("data", "processed"))
    ap.add_argument("--out", default=FLOW_DIR)
    ap.add_argument("--limit", type=int, help="first N clips, for a smoke test")
    args = ap.parse_args(argv)
    if args.verify:
        bad = verify(args.processed, args.out, args.verify)
        if bad:
            print(f"{len(bad)} disagree: {bad}")
            return 1
        print(f"all {args.verify} recomputed clips match, and one clip "
              f"repeats itself")
        return 0
    return build(args.processed, args.out, limit=args.limit,
                 ledger=CACHE_LEDGER)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
