"""Save a clip as one picture, so you can actually look at it.

Used in step 2.4 on raw clips, and again in 3.4 to judge crops by eye. Judging
a crop by eye is not laziness -- there is no number that tells you the box is
round the batter rather than the keeper.
"""
import os

import cv2
import numpy as np

from src.data.video import read_clip

__all__ = ["frame_grid", "compare"]


def frame_grid(clip, out, cols: int = 6, width: int = 300, label: bool = True):
    """Tile every frame of a clip into a single PNG. Returns the path.

    `clip` is an array from read_clip, or a path.
    """
    if isinstance(clip, (str, os.PathLike)):
        clip = read_clip(clip)

    scale = width / clip.shape[2]
    h = int(round(clip.shape[1] * scale))
    tiles = []
    for frame in clip:
        tiles.append(cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA))

    if label:
        for i, tile in enumerate(tiles):
            cv2.putText(tile, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)       # outline first
            cv2.putText(tile, str(i), (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 1, cv2.LINE_AA)

    rows = int(np.ceil(len(tiles) / cols))
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * cols:                           # pad the last row
        tiles.append(blank)

    stacked = []
    for r in range(rows):
        stacked.append(np.hstack(tiles[r * cols:(r + 1) * cols]))
    sheet = np.vstack(stacked)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))  # imwrite wants BGR
    return out


def compare(before, after, out, width: int = 420, titles=("before", "after")):
    """One frame from each of two clips, side by side. The thesis figure."""
    if isinstance(before, (str, os.PathLike)):
        before = read_clip(before)
    if isinstance(after, (str, os.PathLike)):
        after = read_clip(after)

    middles = [before[len(before) // 2], after[len(after) // 2]]
    panes = []
    for img, title in zip(middles, titles):
        h = int(round(img.shape[0] * width / img.shape[1]))
        pane = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        cv2.putText(pane, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(pane, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 1, cv2.LINE_AA)
        panes.append(pane)

    # the two clips need not be the same height -- pad the shorter one
    tall = 0
    for pane in panes:
        if pane.shape[0] > tall:
            tall = pane.shape[0]
    padded = []
    for pane in panes:
        padded.append(np.pad(pane, ((0, tall - pane.shape[0]), (0, 0), (0, 0))))
    panes = padded

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(np.hstack(panes), cv2.COLOR_RGB2BGR))
    return out


def draw_boxes(frame, dets, out=None, colors=None):
    """Draw detections on a frame. Returns the annotated copy."""
    if colors is None:
        colors = {"Striker": (0, 200, 255), "Bat": (0, 255, 80)}
    img = frame.copy()
    for det in dets:
        x1 = int(det.box[0])
        y1 = int(det.box[1])
        x2 = int(det.box[2])
        y2 = int(det.box[3])
        colour = colors.get(det.cls, (255, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        tag = f"{det.cls} {det.conf:.2f}"
        cv2.putText(img, tag, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, tag, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1, cv2.LINE_AA)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img
