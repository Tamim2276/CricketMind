"""
Step 2 of preprocessing: uniformly sample a fixed number of frames per clip.

Not part of the published CricShot10k code -- their training notebook expects
videos that already contain exactly `num_frame` (15) frames each (see the
"15fs" path segment and the "All Shots Cropped 15frames" folder name in their
segmentation notebook), but the step that produces that isn't published.
This fills that gap with plain uniform sampling, including repeats if a clip
has fewer than `num_frames` frames after cropping.
"""
import cv2
import numpy as np


def sample_video(input_video_path, output_video_path, num_frames=15, output_size=(224, 224)):
    """Returns the number of frames written."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_video_path}")
        return 0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total = len(frames)
    if total == 0:
        return 0

    indices = np.linspace(0, total - 1, num_frames).round().astype(int)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_video_path, fourcc, 15.0, output_size)

    for idx in indices:
        out.write(frames[idx])

    out.release()
    return len(indices)
