# Strategies to Break 90% Accuracy

To go from ~73% to >90% on a complex video dataset of 6,700 clips, we have to move beyond standard ImageNet-pretrained CNNs. If your current `multiscale`, `transformer`, and `combined` models peak around 80-85%, here are the top 5 most feasible, state-of-the-art (SOTA) techniques we can implement next to break the 90% barrier.

## 1. Skeleton / Keypoint Extraction (Pose Estimation)
**The Concept:** Right now, the model looks at raw RGB pixels, which means it gets distracted by the color of the jersey, the grass, and the lighting. Cricket shots are defined by human biomechanics (the angle of the elbows, knees, and spine).
**The Implementation:** 
- We run `YOLO-Pose` or `MediaPipe` on the dataset to extract the (x,y) coordinates of the batsman's joints.
- We train a lightweight **Spatial-Temporal Graph Convolutional Network (ST-GCN)** on these coordinates. 
- **Why it works:** A skeleton model is 100% blind to the background. It only sees pure motion. Fusing a Skeleton model with your RGB model is the most guaranteed way to hit >90% in modern action recognition.

## 2. Test-Time Augmentation (TTA)
**The Concept:** A "free" accuracy boost that requires zero retraining. 
**The Implementation:** 
- During `evaluate.py`, instead of running the test video through the model once, we run it 5 times:
  1. Original
  2. Horizontally Flipped (Left-handed vs Right-handed batsman)
  3. Slightly zoomed in
  4. Slightly zoomed out
  5. Different 15 frames sampled
- We average the 5 predictions together.
- **Why it works:** It acts like a mini-ensemble. If the model is confused by one angle, the flipped/zoomed angle often corrects it. This typically yields a free **+2% to +4%** accuracy boost.

## 3. Video-Pretrained Backbones (Kinetics-400)
**The Concept:** We are currently using `EfficientNetV2-S`, which was pretrained on ImageNet (static pictures of dogs, cats, cars). It doesn't inherently understand "motion".
**The Implementation:**
- We swap the backbone to a 3D-CNN (like `X3D` or `SlowFast`) or a Video Transformer (like `VideoMAE`) that was pre-trained on **Kinetics-400** (a dataset of 400,000 human action videos).
- **Why it works:** A Kinetics-pretrained model already knows how to track a human swinging an object through time. Fine-tuning it on Cricket shots takes a fraction of the data and yields massively higher accuracy.

## 4. Optical Flow (Two-Stream Networks)
**The Concept:** Some shots (like an Upper Cut vs a Late Cut) look almost identical in static frames; the only difference is the exact velocity and direction the bat is moving.
**The Implementation:**
- We compute **Optical Flow** (a mathematical map of how pixels move between frame 1 and frame 2).
- We feed the RGB frames into one CNN, and the Optical Flow into a second CNN, and fuse them.
- **Why it works:** Optical flow explicitly forces the network to look at motion vectors (velocity) rather than just spatial pixels.

## 5. Model Ensembling
**The Concept:** You are currently training 4 different models for your paper (`baseline`, `multiscale`, `transformer`, `combined`). 
**The Implementation:**
- When all 4 are done, we write a script that loads all 4 models into memory.
- For a test video, we ask all 4 models for their probabilities and average the scores.
- **Why it works:** The `multiscale` model might be great at wrists, while the `transformer` is great at timing. Ensembling them combines their strengths and masks their weaknesses, almost always pushing the final accuracy up by **+3% to +5%**.
