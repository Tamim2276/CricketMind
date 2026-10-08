"""Find the striker and the bat in a frame, using the authors' YOLO model.

Detection only -- picking which of several people is the striker is 3.2's job.
"""
import os
from functools import lru_cache
from typing import NamedTuple

import cv2
import numpy as np

from src.utils.device import get_device

__all__ = ["Detection", "load_detector", "detect", "MODEL_PATH", "CLASSES"]

# Player_Type_Detection_Model.pt is byte-identical to the segmentation one
MODEL_PATH = os.path.join("CricShoot10kModels", "Player_Type_Detection_Model.pt")
CLASSES = ("Striker", "Bat")

# The checkpoint says imgsz=224, and ultralytics uses that unless told
# otherwise -- squashing a 896x540 broadcast frame until the batter is a few
# pixels. Measured over five clips, striker detection went 0-75% at 224 and
# 71-100% at 896. Native width it is.
IMGSZ = 896
# Measured on an idle Arc at 896: batches of 4, 8 and 16 all run at ~21 fps,
# peaking at 1.0 / 1.8 / 4.1 GB. 24 fails outright. 4 is the cheapest of the
# equally-fast options, which leaves the card free for anything else running.
BATCH = 4


class Detection(NamedTuple):
    cls: str
    conf: float
    box: tuple          # x1, y1, x2, y2 in pixels

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def centre(self) -> tuple:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@lru_cache(maxsize=2)
def load_detector(path: str = MODEL_PATH):
    """Cached -- the weights are 124 MB and loading them is not free."""
    from ultralytics import YOLO
    if not os.path.exists(path):
        raise FileNotFoundError(f"detector not found: {path}")
    return YOLO(path)


def detect(frames, conf: float = 0.25, model=None, device=None,
           rgb: bool = True, imgsz: int = IMGSZ, batch: int = BATCH):
    """Detections per frame. Pass one frame or a list; always get a list back.

    `rgb=True` means the frames came from read_clip. Ultralytics reads numpy
    arrays as BGR, so they are converted here -- handing it RGB silently costs
    accuracy rather than raising.
    """
    model = model or load_detector()
    device = device or str(get_device())

    single = isinstance(frames, np.ndarray) and frames.ndim == 3
    imgs = [frames] if single else list(frames)
    if rgb:
        imgs = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in imgs]

    out = []
    for i in range(0, len(imgs), batch):
        for r in model.predict(imgs[i:i + batch], device=device, conf=conf,
                               imgsz=imgsz, verbose=False):
            out.append([
                Detection(model.names[int(c)], float(p), tuple(map(float, b)))
                for c, p, b in zip(r.boxes.cls, r.boxes.conf,
                                   r.boxes.xyxy.tolist())
            ])
    return out[0] if single else out
