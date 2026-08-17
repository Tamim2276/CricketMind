"""
Step 3 of preprocessing: segment striker (blue) and bat (green) overlay.

Ported from CricShot10k/Codes/Model_Layer_Segmentation_of_Cropped_Video.ipynb
with the same inference/plot settings.
"""
import cv2


def segment_video(input_video_path, output_video_path, seg_model, output_size=(224, 224)):
    """Returns the number of frames written."""
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
