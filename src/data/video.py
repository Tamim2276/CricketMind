"""Read a clip off disk. The only place in the project that opens a video.

Returns frames as they are on disk -- no resizing, no frame selection, no
normalising. Those belong to the preprocessing steps that follow.
"""
import os

import cv2
import numpy as np

__all__ = ["read_clip", "clip_shape", "ClipError", "retries", "reset_retries"]

# Counts reads that failed once and succeeded on a second attempt. Behind a
# function, not a bare global: `from video import RETRIES` would bind the
# number at import time and always report 0.
_RETRIES = 0


def retries() -> int:
    """Reads that needed a second attempt. Print this at the end of a long run."""
    return _RETRIES


def reset_retries() -> None:
    global _RETRIES
    _RETRIES = 0


class ClipError(RuntimeError):
    """A clip could not be read. Carries the path, so logs are useful."""


def read_clip(path, rgb: bool = True, retry: bool = True) -> np.ndarray:
    """One clip as (frames, height, width, 3) uint8.

    Frame count comes from what actually decodes, not from the file header --
    headers lie, and in this dataset clip length varies from 22 to 49 anyway.

    One retry, because a clip that reads fine every other time once decoded 0
    frames during a test run and the cause was never found. Over 10,091 clips
    that would silently drop a good one. The retry is counted, not swallowed:
    check `retries()` after a long run, and if it is more than a handful, stop
    and investigate rather than trusting the output.
    """
    global _RETRIES
    try:
        return _read_once(path, rgb)
    except ClipError:
        if not retry or not os.path.exists(os.fspath(path)):
            raise                      # missing or unreadable file: no point
        _RETRIES += 1
        return _read_once(path, rgb)


def _read_once(path, rgb: bool) -> np.ndarray:
    path = os.fspath(path)
    if not os.path.exists(path):
        raise ClipError(f"no such clip: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ClipError(f"cannot open: {path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # OpenCV hands back BGR; every ImageNet-pretrained model wants RGB
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ClipError(f"opened but decoded 0 frames: {path}")

    shapes = set()
    for frame in frames:
        shapes.add(frame.shape)
    if len(shapes) > 1:
        raise ClipError(f"frames differ in size {sorted(shapes)}: {path}")

    return np.stack(frames)


def clip_shape(path):
    """(frames, height, width) from the header, without decoding.

    Cheap, for scanning 10,091 files. The frame count is the header's claim --
    use read_clip when it has to be right.
    """
    path = os.fspath(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ClipError(f"cannot open: {path}")
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    finally:
        cap.release()
