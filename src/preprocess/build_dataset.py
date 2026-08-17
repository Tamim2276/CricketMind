"""
Full preprocessing pipeline: raw clip -> crop -> sample 15 frames -> segment
-> written into train/val/test/<class_name>/ folders matching data/splits/*.csv,
so CricShot10k's own create_dataframe()/training notebook can be reused unmodified.

Needs a GPU for the YOLO detection/segmentation models to run at a usable speed
(Kaggle, not the local smoke test) -- see --device.

Usage (smoke test, a couple videos per class, CPU is fine for this):
    python src/preprocess/build_dataset.py --limit 2 --device cpu

Usage (real run, Kaggle, GPU):
    python src/preprocess/build_dataset.py \
        --data_root /kaggle/input/cricshot10k-clips/CricShoot10kShootDataset \
        --splits_dir /kaggle/working/cricshot-improve/data/splits \
        --output_dir /kaggle/working/preprocessed \
        --striker_model /kaggle/input/cricshot10k-models/Player_Type_Detection_Model.pt \
        --bat_model /kaggle/input/cricshot10k-models/Bat_Detection_Model.pt \
        --seg_model /kaggle/input/cricshot10k-models/Striker_Bat_Segmentation_Model.pt \
        --device cuda
"""
import os
import csv
import shutil
import argparse
import tempfile

from src.preprocess.crop import crop_video
from src.preprocess.sample import sample_video
from src.preprocess.segment import segment_video

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
    parser.add_argument("--data_root", default="data/raw/CricShoot10kShootDataset")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--striker_model", default="data/CricShoot10kModels/Player_Type_Detection_Model.pt")
    parser.add_argument("--bat_model", default="data/CricShoot10kModels/Bat_Detection_Model.pt")
    parser.add_argument("--seg_model", default="data/CricShoot10kModels/Striker_Bat_Segmentation_Model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_frames", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None, help="cap videos per class, for smoke testing")
    parser.add_argument("--classes", nargs="*", default=None, help="restrict to these class names, for smoke testing")
    args = parser.parse_args()

    from ultralytics import YOLO  # deferred: only needed once args are parsed

    print("Loading models...")
    striker_model = YOLO(args.striker_model)
    bat_model = YOLO(args.bat_model)
    seg_model = YOLO(args.seg_model)
    striker_model.to(args.device)
    bat_model.to(args.device)
    seg_model.to(args.device)

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

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cropped = os.path.join(tmp_dir, "cropped.avi")
        tmp_sampled = os.path.join(tmp_dir, "sampled.avi")

        for class_name in classes:
            class_dir = os.path.join(args.data_root, class_name)
            videos = sorted(
                f for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS
            )
            if args.limit:
                videos = videos[:args.limit]

            print(f"\n=== {class_name}: {len(videos)} videos ===")

            for video_name in videos:
                split = split_lookup.get((class_name, video_name))
                if split is None:
                    skipped_log.append(f"{class_name}/{video_name}: not found in any split CSV")
                    continue

                out_dir = os.path.join(args.output_dir, split, class_name)
                os.makedirs(out_dir, exist_ok=True)
                final_path = os.path.join(out_dir, video_name)

                if os.path.exists(final_path):
                    total_skipped_existing += 1
                    continue

                input_path = os.path.join(class_dir, video_name)

                n1 = crop_video(input_path, tmp_cropped, striker_model, bat_model)
                if n1 == 0:
                    skipped_log.append(f"{class_name}/{video_name}: 0 frames after crop (no striker detected)")
                    continue

                n2 = sample_video(tmp_cropped, tmp_sampled, num_frames=args.num_frames)
                if n2 == 0:
                    skipped_log.append(f"{class_name}/{video_name}: 0 frames after sampling")
                    continue

                n3 = segment_video(tmp_sampled, final_path, seg_model)
                if n3 == 0:
                    skipped_log.append(f"{class_name}/{video_name}: 0 frames after segmentation")
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    continue

                total_done += 1
                if total_done % 50 == 0:
                    print(f"  processed {total_done} clips so far...")

    print(f"\nDone. {total_done} clips processed, {total_skipped_existing} already existed (resumed), "
          f"{len(skipped_log)} skipped.")

    if skipped_log:
        log_path = os.path.join(args.output_dir, "skipped_videos.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_log))
        print(f"Skipped-clip details written to {log_path}")


if __name__ == "__main__":
    main()
