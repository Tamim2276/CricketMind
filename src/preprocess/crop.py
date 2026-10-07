"""
Step 1 of preprocessing: crop each frame to the striker+bat region.

Ported from CricShot10k/Codes/Model_Layer_Bat_Striker_Cropping_of_Full_video.ipynb
with the same detection/combination/padding logic, only the I/O paths and
model-loading are adapted. Frames where no striker is detected are dropped,
matching the original -- this can leave a clip with fewer output frames than
input frames (or, rarely, zero).
"""
import cv2


def crop_video(input_video_path, output_video_path, striker_model, bat_model, output_size=(224, 224)):
    """Returns the number of frames written (0 means the clip had no usable frames)."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    frames_written = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = striker_model(frame, verbose=False)
        bat_results = bat_model(frame, verbose=False, max_det=1)

        striker_detected = False
        striker_box = None
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0].cpu().numpy())
                label = striker_model.names[cls]
                if label == "Striker":
                    striker_detected = True
                    striker_box = (x1, y1, x2, y2)
                    break
            if striker_detected:
                break

        if not striker_detected:
            continue  # matches original: frame is silently dropped

        bat_detected = False
        bat_box = None
        for bat_result in bat_results:
            for bat_box_item in bat_result.boxes:
                bx1, by1, bx2, by2 = bat_box_item.xyxy[0].cpu().numpy().astype(int)
                bat_cls = int(bat_box_item.cls[0].cpu().numpy())
                bat_label = bat_model.names[bat_cls]
                if bat_label == "Cricket Bat":
                    bat_detected = True
                    bat_box = (bx1, by1, bx2, by2)
                    break
            if bat_detected:
                break

        if bat_detected:
            final_x1 = min(striker_box[0], bat_box[0])
            final_y1 = min(striker_box[1], bat_box[1])
            final_x2 = max(striker_box[2], bat_box[2])
            final_y2 = max(striker_box[3], bat_box[3])
        else:
            final_x1, final_y1, final_x2, final_y2 = striker_box

        width = final_x2 - final_x1
        padding = int(width * 0.1)
        final_x1_expanded = max(0, final_x1 - padding)
        final_x2_expanded = min(frame.shape[1], final_x2 + padding)

        cropped_frame = frame[final_y1:final_y2, final_x1_expanded:final_x2_expanded]
        if cropped_frame.size == 0:
            continue

        resized_frame = cv2.resize(cropped_frame, output_size)
        out.write(resized_frame)
        frames_written += 1

    cap.release()
    out.release()
    return frames_written
