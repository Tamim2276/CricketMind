"""Run the whole preprocessing pipeline over every clip, resumably.

Per clip: read -> drop duplicates -> detect -> crop -> sample 15 -> save.

**Written to survive a power cut**, because a 4-8 hour run on a desktop with
no battery will meet one. Every clip is finished and recorded before the next
starts, so an interruption costs one clip. Re-running the same command skips
whatever the manifest already lists.

Saved as raw uint8 .npy rather than JPEG. Measured on real crops over 10,091
clips: raw 42.4 GB for both variants, PNG 11.3 GB, JPEG q95 3.0 GB. Raw costs
disk but stays lossless, and the expensive thing here is the run, not the
bytes -- raw and PNG convert to each other for free, JPEG is one way.
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

from src.data.video import read_clip, reset_retries, retries
from src.preprocess.crop import crop_clip
from src.preprocess.detect import detect, load_detector
from src.preprocess.sample import TARGET, drop_duplicates, sample
from src.utils import progress
from src.utils.checkpoint import append_result, read_results
from src.utils.device import get_device

__all__ = ["process_clip", "build", "VARIANTS"]

ROOT = os.path.join("data", "CricShoot10kShootDataset")
OUT = os.path.join("data", "processed")
MANIFEST = os.path.join(OUT, "manifest.jsonl")
VARIANTS = ("box", "seg")


def _progress(done: int, total: int, failed: int, t0: float, tty: bool) -> str:
    """One line, rewritten in place on a terminal, appended in a log file."""
    rate = done / max(time.time() - t0, 1e-9)
    if rate:
        left = (total - done) / rate / 3600
    else:
        left = 0.0

    if failed and tty:
        bad = f"{progress.RED}{failed} failed{progress.OFF}"
    else:
        bad = f"{failed} failed"

    tail = f"{rate:4.1f} clips/s  {bad}  ~{left:4.1f} h left"
    return progress.line(done, total, tail, prefix="  ", tty=tty)


def clip_list(root: str = ROOT):
    """Every clip, as (class, relative path). Sorted, so runs are comparable."""
    out = []
    for cls in sorted(os.listdir(root)):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".avi"):
                out.append((cls, f"{cls}/{name}"))
    return out


def out_path(out_dir: str, variant: str, rel: str) -> str:
    return os.path.join(out_dir, variant, rel.replace("/", os.sep)[:-4] + ".npy")


def process_clip(rel, model, device, out_dir=OUT, variants=VARIANTS,
                 frames=TARGET, root=ROOT):
    """One clip, start to finish. Returns the row to record."""
    clip = read_clip(os.path.join(root, rel))
    raw = len(clip)
    clip, _ = drop_duplicates(clip)
    per = detect(list(clip), model=model, device=device,
                 masks="seg" in variants)

    row = {"clip": rel, "raw": raw, "dedup": len(clip)}
    for v in variants:
        r = crop_clip(clip, per, segmented=(v == "seg"))
        if r.frames.shape[0] == 0:
            return {**row, "ok": False, "reason": r.note}
        path = out_path(out_dir, v, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _save(sample(r.frames, frames), path)
        row[f"{v}_kept"] = int(r.frames.shape[0])
        row[f"{v}_note"] = r.note
    return {**row, "ok": True}


def _save(arr, path: str) -> None:
    """np.save, but a cut cannot leave a half-written .npy behind."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def build(limit=None, out_dir=OUT, variants=VARIANTS, root=ROOT,
          manifest=None, frames=TARGET, device=None):
    manifest = manifest or os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    clips = clip_list(root)
    if limit:
        clips = clips[:limit]
    done = set()
    for row in read_results(manifest):
        done.add(row["clip"])

    todo = []
    for cls, rel in clips:
        if rel not in done:
            todo.append((cls, rel))

    model = load_detector()
    device = device or str(get_device())
    reset_retries()
    print(f"{len(clips)} clips, {len(done)} already done, {len(todo)} to go")
    print(f"device={device}  variants={','.join(variants)}  -> {out_dir}\n",
          flush=True)

    tty = sys.stdout.isatty()
    t0, failed = time.time(), 0
    for n, (cls, rel) in enumerate(todo, 1):
        try:
            row = process_clip(rel, model, device, out_dir, variants, frames, root)
        except Exception as e:            # one bad clip must not kill the run
            row = {"clip": rel, "ok": False,
                   "reason": f"{type(e).__name__}: {e}"}
            if tty:
                print()                   # do not scribble over the bar
            traceback.print_exc(limit=1)
        row["class"] = cls
        append_result(row, manifest)
        failed += not row["ok"]

        if tty:
            print(_progress(n, len(todo), failed, t0, True), end="", flush=True)
        elif n % 25 == 0 or n == len(todo):
            print(_progress(n, len(todo), failed, t0, False), flush=True)

    secs = time.time() - t0
    print(f"\ndone in {secs/3600:.2f} h   {failed} failed of {len(todo)}")
    if retries():
        print(f"read_clip needed a second attempt {retries()} times -- "
              f"if that is more than a handful, investigate before trusting this")
    return failed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, help="first N clips, for a smoke test")
    p.add_argument("--out", default=OUT)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--frames", type=int, default=TARGET)
    p.add_argument("--device")
    a = p.parse_args(argv)
    return build(a.limit, a.out, tuple(a.variants.split(",")), a.root,
                 frames=a.frames, device=a.device)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
