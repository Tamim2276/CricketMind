"""The authors' preprocessing, as their notebooks actually run it (7.6).

Their paper describes one pipeline and their code runs another
(CricShot10kAuthorsCode/Codes, read 2026-10-21). This rebuilds a few
hundred author-val clips the way the code does, so their released model and
ours can both be scored on frames like the ones their model learned from:

    crop        every frame on its own: striker box joined with the bat box
                (Bat_Detection_Model, max_det 1), 10% of the width added left
                and right only, then stretched to 224 x 224
    no striker  that frame is dropped
    video       MJPG, written and read back, as each of their steps saves
    sample      15 frames, evenly spaced over what is left
    overlay     ultralytics results.plot() at imgsz 256, max_det 2: striker
                blue (4, 42, 255), bat cyan (11, 219, 235)

Two variants come out, so the crop and the colours can be told apart:

    replica_crop   their crop, no overlay
    replica        their crop and their overlay

One substitute: their striker detector was never released (the file named
for it is a byte copy of the segmentation model), so the striker box is our
own tracked pick from detect.py. Everything after that follows their code.

    python -m tools.replica --build        # 300 author val clips, resumable
    python -m tools.replica --score        # B01 and B03 on box, crop, replica

Their model is scored by tools/authors_model.py, under Keras 3.8.0.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

from src.data.video import read_clip
from src.preprocess.crop import track_picks
from src.preprocess.detect import detect, load_detector
from src.utils import progress
from src.utils.checkpoint import append_result, read_results
from tools.authors_model import balanced_sample, read_split

RAW = os.path.join("data", "CricShoot10kShootDataset")
PROCESSED = os.path.join("data", "processed")
SOURCE = os.path.join("data", "splits_author", "val.csv")
OUT_SPLIT = os.path.join("data", "splits_replica")
MANIFEST = os.path.join(PROCESSED, "manifest_replica.jsonl")
BAT_MODEL = os.path.join("CricShoot10kModels", "Bat_Detection_Model.pt")
SEG_MODEL = os.path.join("CricShoot10kModels",
                         "Striker_Bat_Segmentation_Model.pt")
SIZE = 224
FRAMES = 15
PAD = 0.1           # of the box width, sideways only
SEG_IMGSZ = 256
CLIPS = 300         # 20 a class
CHECKPOINTS = {
    "B01 EfficientNetV2-S": os.path.join("experiments", "mimic_author",
                                         "best.pt"),
    "B03 Xception": os.path.join("experiments", "B03_xception__author",
                                 "best.pt"),
}
VARIANTS = ("box", "replica_crop", "replica")


# --- their geometry -----------------------------------------------------------

def authors_box(striker, bat, frame_w):
    """Their crop box: striker joined with the bat, then 10% of the width
    on each side. The rows are not padded: that is their code, not a slip."""
    x1, y1, x2, y2 = striker
    if bat is not None:
        x1 = min(x1, bat[0])
        y1 = min(y1, bat[1])
        x2 = max(x2, bat[2])
        y2 = max(y2, bat[3])
    pad = int((x2 - x1) * PAD)
    return max(0, x1 - pad), y1, min(frame_w, x2 + pad), y2


def stretch(frame, box, size=SIZE):
    """Cut the box out and resize it to a square, shape be damned; None if
    the box is empty."""
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1):y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def even_indices(n, k=FRAMES):
    """k indices spread evenly over n frames, ends included."""
    return np.linspace(0, n - 1, k).round().astype(int)


def as_ints(box):
    """Their boxes went through .astype(int): truncated, not rounded."""
    out = []
    for v in box:
        out.append(int(v))
    return tuple(out)


def mjpg_round_trip(frames_bgr, folder):
    """Write the frames as an MJPG .avi and read them back, as every one of
    their notebooks saves its output."""
    path = os.path.join(folder, "clip.avi")
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (w, h))
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    cap = cv2.VideoCapture(path)
    back = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        back.append(frame)
    cap.release()
    os.remove(path)
    if len(back) != len(frames_bgr):
        raise RuntimeError(f"MJPG gave back {len(back)} of "
                           f"{len(frames_bgr)} frames")
    return back


# --- their models -------------------------------------------------------------

def load_models():
    from ultralytics import YOLO
    return {"striker": load_detector(), "bat": YOLO(BAT_MODEL),
            "seg": YOLO(SEG_MODEL)}


def bat_boxes(frames_bgr, model, device, batch=8):
    """One bat box per frame or None, as their call: max_det 1, defaults."""
    out = []
    for i in range(0, len(frames_bgr), batch):
        chunk = frames_bgr[i:i + batch]
        results = model.predict(chunk, max_det=1, device=device, verbose=False)
        for result in results:
            found = None
            names = result.names
            for k in range(len(result.boxes)):
                if names[int(result.boxes.cls[k])] == "Cricket Bat":
                    found = as_ints(result.boxes.xyxy[k].tolist())
                    break
            out.append(found)
    return out


def authors_overlay(frame_bgr, model, device):
    """Their segmentation layer: ultralytics draws the masks itself."""
    result = model.predict(frame_bgr, imgsz=SEG_IMGSZ, max_det=2,
                           device=device, verbose=False)[0]
    return result.plot(boxes=False, probs=False, labels=False)


# --- one clip -----------------------------------------------------------------

def replicate(rel, models, device, root=RAW):
    """(replica_crop, replica, note) as (15, 224, 224, 3) RGB uint8, or
    (None, None, why)."""
    clip = read_clip(os.path.join(root, rel))                # RGB
    frames_bgr = []
    for frame in clip:
        frames_bgr.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    h, w = clip.shape[1:3]

    # on the CPU, one frame at a time: four 896px frames at once crashed
    # (segfault) with a training run holding most of the RAM
    batch = 4
    if str(device).startswith("cpu"):
        batch = 1
    picks = track_picks(detect(list(clip), model=models["striker"],
                               device=device, batch=batch), (h, w))
    bats = bat_boxes(frames_bgr, models["bat"], device, batch=batch)

    crops = []
    for frame, pick, bat in zip(frames_bgr, picks, bats):
        if pick is None:
            continue                       # their code drops the frame
        box = authors_box(as_ints(pick.box), bat, w)
        crop = stretch(frame, box)
        if crop is not None:
            crops.append(crop)
    if not crops:
        return None, None, "no striker in any frame"

    with tempfile.TemporaryDirectory() as folder:
        crops = mjpg_round_trip(crops, folder)               # crop notebook
        picked = []
        for i in even_indices(len(crops)):
            picked.append(crops[i])
        picked = mjpg_round_trip(picked, folder)             # 15-frame videos
        drawn = []
        for frame in picked:
            drawn.append(authors_overlay(frame, models["seg"], device))
        drawn = mjpg_round_trip(drawn, folder)               # seg notebook

    plain = []
    for frame in picked:
        plain.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tinted = []
    for frame in drawn:
        tinted.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    note = f"kept {len(crops)}/{len(clip)} frames"
    return np.stack(plain), np.stack(tinted), note


# --- the sample, built and scored ---------------------------------------------

def npy_path(variant, rel, processed=PROCESSED):
    return os.path.join(processed, variant,
                        rel.replace("/", os.sep)[:-4] + ".npy")


def _save(arr, path):
    """np.save, but a cut cannot leave a half-written file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_split(rows, done, out=OUT_SPLIT):
    """The clips that came out, in the split format, with classes.json."""
    os.makedirs(out, exist_ok=True)
    kept = []
    for row in rows:
        if row["clip"] in done:
            kept.append(row)
    path = os.path.join(out, "val.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("clip,class,label,match\n")
        for row in kept:
            fh.write(f"{row['clip']},{row['class']},{row['label']},"
                     f"{row['match']}\n")
    shutil.copy(os.path.join(os.path.dirname(SOURCE), "classes.json"),
                os.path.join(out, "classes.json"))
    return path, len(kept)


def build(n=CLIPS, device=None, manifest=MANIFEST):
    from src.utils.device import get_device

    device = device or str(get_device())
    rows = balanced_sample(read_split(SOURCE), n, seed=0)
    done = set()
    for row in read_results(manifest):
        if row.get("ok"):
            done.add(row["clip"])
    todo = []
    for row in rows:
        if row["clip"] not in done:
            todo.append(row)
    print(f"replica: {len(rows)} author-val clips, {len(done)} done, "
          f"{len(todo)} to go, on {device}", flush=True)

    models = load_models() if todo else None
    started = time.time()
    failed = 0
    for k, row in enumerate(todo, 1):
        rel = row["clip"]
        try:
            plain, tinted, note = replicate(rel, models, device)
        except Exception as err:            # one bad clip must not stop 300
            plain, tinted, note = None, None, f"{type(err).__name__}: {err}"
        if plain is None:
            failed += 1
            append_result({"clip": rel, "ok": False, "reason": note}, manifest)
        else:
            _save(plain, npy_path("replica_crop", rel))
            _save(tinted, npy_path("replica", rel))
            append_result({"clip": rel, "ok": True, "note": note}, manifest)
            done.add(rel)
        rate = k / max(time.time() - started, 1e-9)
        left = (len(todo) - k) / max(rate, 1e-9) / 60
        sys.stdout.write(progress.line(k, len(todo),
                                       f"{rate:4.2f} clips/s  ~{left:4.1f} min "
                                       f"left", prefix="  ",
                                       tty=progress.is_tty()))
        if not progress.is_tty():
            sys.stdout.write("\n")
        sys.stdout.flush()

    path, kept = write_split(rows, done)
    print(f"\n{kept} clips ready, {failed} failed this run -> {path}")
    return kept


def score(checkpoints=None, split_csv=None, device=None,
          out=os.path.join("experiments", "replica_scores.json")):
    """Each of our checkpoints on the same clips, three ways."""
    from src.evaluate import evaluate_checkpoint

    checkpoints = checkpoints or CHECKPOINTS
    split_csv = split_csv or os.path.join(OUT_SPLIT, "val.csv")
    table = {}
    for name, ckpt in checkpoints.items():
        table[name] = {}
        for variant in VARIANTS:
            report = evaluate_checkpoint(ckpt, split_csv, variant=variant,
                                         device=device)
            table[name][variant] = {"top1": report["top1"],
                                    "macro_f1": report["macro_f1"]}
            print(f"  {name:22s} {variant:13s} {report['top1']:.2%}  "
                  f"F1 {report['macro_f1']:.3f}", flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"split_csv": split_csv, "scores": table}, fh, indent=2)
    print(f"\nsaved {out}")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true",
                    help="make replica_crop/ and replica/ for the sample")
    ap.add_argument("--score", action="store_true",
                    help="score B01 and B03 on box, replica_crop, replica")
    ap.add_argument("--clips", type=int, default=CLIPS)
    ap.add_argument("--device", help="default: the GPU if there is one")
    args = ap.parse_args(argv)
    if not args.build and not args.score:
        ap.error("give --build, --score or both")
    if args.build:
        build(args.clips, args.device)
    if args.score:
        score(device=args.device)


if __name__ == "__main__":
    main()
