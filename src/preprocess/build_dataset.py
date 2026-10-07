"""
Full preprocessing pipeline: raw clip -> crop -> sample 15 frames -> segment
-> written as .pt tensor files into train/val/test/<class_name>/ folders.

Output format: torch.Tensor of shape (T, H, W, C) uint8 RGB, saved with torch.save().
This eliminates per-epoch video decode overhead during training (~3-5x speedup).

Usage (local full run, Arc B580 XPU):
    python -m src.preprocess.build_dataset --device xpu

Usage (smoke test, 2 videos per class, CPU):
    python -m src.preprocess.build_dataset --limit 2 --device cpu

Usage (Kaggle CUDA):
    python -m src.preprocess.build_dataset \\
        --data_root /kaggle/input/cricshot10k-clips/CricShoot10kShootDataset \\
        --splits_dir /kaggle/working/cricshot-improve/data/splits \\
        --output_dir /kaggle/working/preprocessed \\
        --striker_model /kaggle/input/cricshot10k-models/Player_Type_Detection_Model.pt \\
        --bat_model /kaggle/input/cricshot10k-models/Bat_Detection_Model.pt \\
        --seg_model /kaggle/input/cricshot10k-models/Striker_Bat_Segmentation_Model.pt \\
        --device cuda
"""
import os
import csv
import argparse
import tempfile
import time

import numpy as np
import torch
from tqdm import tqdm

from src.preprocess.crop import crop_video
from src.preprocess.sample import sample_video
from src.preprocess.segment import segment_video_to_frames

VIDEO_EXTS = {".avi", ".mp4", ".mov"}


def load_split_lookup(splits_dir):
    """Returns {(class_name, filename): split} built from train/val/test.csv."""
    lookup = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(splits_dir, f"{split}.csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                filename = os.path.basename(row["video_path"])
                lookup[(row["class_name"], filename)] = split
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="data/raw/CricShoot10kShootDataset")
    parser.add_argument("--splits_dir",    default="data/splits")
    parser.add_argument("--output_dir",    default="data/processed")
    parser.add_argument("--striker_model", default="data/CricShoot10kModels/Player_Type_Detection_Model.pt")
    parser.add_argument("--bat_model",     default="data/CricShoot10kModels/Bat_Detection_Model.pt")
    parser.add_argument("--seg_model",     default="data/CricShoot10kModels/Striker_Bat_Segmentation_Model.pt")
    parser.add_argument("--device",        default="xpu",
                        help="Device for YOLO models: xpu (Arc B580), cuda (Kaggle), cpu (fallback)")
    parser.add_argument("--num_frames",    type=int, default=15)
    parser.add_argument("--limit",         type=int, default=None,
                        help="Cap videos per class — for smoke testing only")
    parser.add_argument("--classes",       nargs="*", default=None,
                        help="Restrict to these class names — for smoke testing only")
    args = parser.parse_args()

    from ultralytics import YOLO  # deferred import

    print(f"Loading YOLO models on device={args.device!r}...")
    striker_model = YOLO(args.striker_model)
    bat_model     = YOLO(args.bat_model)
    seg_model     = YOLO(args.seg_model)
    striker_model.to(args.device)
    bat_model.to(args.device)
    seg_model.to(args.device)
    print("Models loaded.")

    split_lookup = load_split_lookup(args.splits_dir)

    classes = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    )
    if args.classes:
        classes = [c for c in classes if c in args.classes]

    skipped_log = []
    total_done = 0
    total_skipped_existing = 0
    t_start = time.time()

    # Count total videos upfront for the outer progress bar
    all_videos = []
    for class_name in classes:
        class_dir = os.path.join(args.data_root, class_name)
        vids = sorted(
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        )
        if args.limit:
            vids = vids[:args.limit]
        for v in vids:
            all_videos.append((class_name, v))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cropped = os.path.join(tmp_dir, "cropped.avi")
        tmp_sampled = os.path.join(tmp_dir, "sampled.avi")

        outer_bar = tqdm(
            total=len(all_videos),
            desc="Total clips",
            unit="clip",
            dynamic_ncols=True,
            colour="green",
        )

        for class_name in classes:
            class_dir = os.path.join(args.data_root, class_name)
            videos = sorted(
                f for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS
            )
            if args.limit:
                videos = videos[:args.limit]

            inner_bar = tqdm(
                videos,
                desc=f"{class_name[:18]:<18}",
                unit="clip",
                leave=False,
                dynamic_ncols=True,
            )

            for video_name in inner_bar:
                split = split_lookup.get((class_name, video_name))
                if split is None:
                    skipped_log.append(f"{class_name}/{video_name}: not found in any split CSV")
                    continue

                out_dir = os.path.join(args.output_dir, split, class_name)
                os.makedirs(out_dir, exist_ok=True)

                # Output is a .pt tensor file, not .avi
                stem       = os.path.splitext(video_name)[0]
                final_path = os.path.join(out_dir, stem + ".pt")

                if os.path.exists(final_path):
                    total_skipped_existing += 1
                    continue

                input_path = os.path.join(class_dir, video_name)

                # Step 1: Crop to striker+bat region
                n1 = crop_video(input_path, tmp_cropped, striker_model, bat_model)
                if n1 == 0:
                    skipped_log.append(f"{class_name}/{video_name}: 0 frames after crop (no striker detected)")
                    outer_bar.update(1)
                    continue

                # Step 2: Uniformly sample num_frames frames
                n2 = sample_video(tmp_cropped, tmp_sampled, num_frames=args.num_frames)
                if n2 == 0:
                    skipped_log.append(f"{class_name}/{video_name}: 0 frames after sampling")
                    outer_bar.update(1)
                    continue

                # Step 3: Segment (returns RGB numpy frames, not a video file)
                frames = segment_video_to_frames(tmp_sampled, seg_model)
                if len(frames) == 0:
                    skipped_log.append(f"{class_name}/{video_name}: 0 frames after segmentation")
                    outer_bar.update(1)
                    continue

                # Pad/truncate to exactly num_frames
                while len(frames) < args.num_frames:
                    frames.append(frames[-1])
                frames = frames[:args.num_frames]

                # Save as (T, H, W, C) uint8 tensor — no float conversion, saves ~4x space
                clip_tensor = torch.from_numpy(np.stack(frames, axis=0))  # (T, H, W, C) uint8
                torch.save(clip_tensor, final_path)

                total_done += 1
                outer_bar.update(1)
                outer_bar.set_postfix(
                    done=total_done,
                    skipped=len(skipped_log),
                    existing=total_skipped_existing,
                )
                if total_done % 50 == 0:
                    elapsed = time.time() - t_start
                    rate    = total_done / elapsed
                    outer_bar.write(
                        f"  {total_done} done | {rate:.1f} clip/s | "
                        f"~{(len(all_videos) - total_done - total_skipped_existing) / max(rate, 1e-9) / 60:.0f} min left"
                    )

            inner_bar.close()
        outer_bar.close()

    elapsed = time.time() - t_start
    print(f"\nDone. {total_done} clips processed, {total_skipped_existing} already existed (resumed), "
          f"{len(skipped_log)} skipped. Total time: {elapsed/60:.1f} min")

    if skipped_log:
        log_path = os.path.join(args.output_dir, "skipped_videos.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_log))
        print(f"Skipped-clip details written to {log_path}")


if __name__ == "__main__":
    main()
