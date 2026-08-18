"""
Step 3 of preprocessing: segment striker (blue) and bat (green) overlay.

Ported from CricShot10k/Codes/Model_Layer_Segmentation_of_Cropped_Video.ipynb
with the same inference/plot settings.
"""
import cv2
import numpy as np


def segment_video_to_frames(input_video_path, seg_model, output_size=(224, 224)):
    """Returns a list of annotated frames as RGB numpy arrays (H, W, C) uint8.
    Used by build_dataset.py to produce .pt tensor outputs instead of .avi files.
    Returns an empty list if the video cannot be opened.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_video_path}")
        return []

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = seg_model(
            frame, imgsz=256, max_det=2,
            show_boxes=False, show_labels=False, show_conf=False, verbose=False,
        )
        annotated = results[0].plot(boxes=False, probs=False, labels=False)  # BGR
        # Resize if needed (shouldn't be needed after crop/sample, but guard anyway)
        if (annotated.shape[1], annotated.shape[0]) != output_size:
            annotated = cv2.resize(annotated, output_size)
        # Convert BGR → RGB for consistency with torchvision transforms
        frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def segment_video(input_video_path, output_video_path, seg_model, output_size=(224, 224)):
    """Legacy function: writes segmented frames to an .avi video file.
    Kept for backward compatibility. New code should use segment_video_to_frames().
    Returns the number of frames written.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_video_path}")
        return 0

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_video_path, fourcc, 15.0, output_size)

    frames_written = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = seg_model(
            frame, imgsz=256, max_det=2,
            show_boxes=False, show_labels=False, show_conf=False, verbose=False,
        )
        annotated = results[0].plot(boxes=False, probs=False, labels=False)
        out.write(annotated)
        frames_written += 1

    cap.release()
    out.release()
    return frames_written
