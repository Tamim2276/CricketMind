"""Day 42: where a model looks when it answers, by Grad-CAM.

    python -m tools.gradcam                     C06's first run: 5 right, 5 wrong
    python -m tools.gradcam --run F02_multiscale_tf__grouped
    python -m tools.gradcam --clips Pull/vid445_27.avi --device cpu

Grad-CAM weighs a layer's maps by how much each one pushes the answer and
keeps the part that pushes it up. The stronger the orange, the more that spot
made the model give the answer named on the left.

    MViT     the tokens going into its last block. After that block only the
             class token reaches the head, so a later layer would show nothing.
             Its 8 time steps are stretched over the 15 frames.
    F02      EfficientNet's last stage, each frame on its own. F02 also reads
             two earlier stages, which this does not show.

Picking the clips needs every val answer: the first time, about a minute on
the GPU, kept as tuning's val_probs.npz. With --clips nothing else is scored,
so it runs on the CPU while training. Pictures go to
notes/figures/gradcam/<run>/, rebuilt on every run. Val only; test stays
closed (rule 5).
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import to_rgb

from src.evaluate import _restore
from src.experiments import registry as R
from src.models.model import ShotModel
from src.models.video_classifier import VideoClassifier
from src.utils.device import get_device
from tools.tuning import SPLIT, val_probs

__all__ = ["target_layer", "cam_of", "grad_cam", "pick", "draw", "main"]

RUN = "C06_mvitv2_s__grouped"
COUNT = 5
SHOWN = 8
FIGURES = os.path.join("notes", "figures", "gradcam")
HEAT = "#eb6834"
STRONGEST = 0.75


class _Keep:
    """Holds what the hooked layer last returned, or the args it was given."""

    def __init__(self):
        self.value = None

    def __call__(self, module, inputs, output=None):
        if output is None:
            self.value = inputs
        else:
            self.value = output


def target_layer(model):
    """(layer, kind): which maps Grad-CAM reads for this model."""
    if isinstance(model, VideoClassifier) and model.name == "mvit_v2_s":
        return model.backbone.blocks[-1].norm1, "mvit"
    if isinstance(model, ShotModel):
        features = getattr(model.encoder.backbone, "features", None)
        if features is not None:
            return features[-1], "frames"
    raise SystemExit(f"Grad-CAM here knows MViT and the frame-CNN models, "
                     f"not {type(model).__name__}")


def cam_of(maps, grads, kind, frames, size, thw=None):
    """One map per stored frame, (frames, size, size), not yet scaled."""
    if kind == "mvit":
        # the class token first, then one token per (time, row, column)
        maps = maps[0, 1:]
        grads = grads[0, 1:]
        weights = grads.mean(dim=0)
        cam = torch.relu((maps * weights).sum(dim=1))
        t, h, w = thw
        cam = cam.view(1, 1, t, h, w)
        cam = F.interpolate(cam, size=(frames, size, size), mode="trilinear",
                            align_corners=False)[0, 0]
    else:
        # (frames, channels, h, w): each frame weighs its own channels
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((maps * weights).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=(size, size),
                            mode="bilinear", align_corners=False)[:, 0]
    # trilinear can dip a hair under zero
    return cam.clamp(min=0).cpu().numpy()


def grad_cam(model, clip, device, label=None):
    """(probabilities, {class: map}) for the model's answer, and for `label`
    too when the answer is wrong. clip is (T, 3, H, W), as dataset.py hands
    it over. Full precision, no autocast; no weight learns anything.

    The maps share one scale, 0 to 1, so weak evidence for the true shot
    looks weak beside the answer's."""
    layer, kind = target_layer(model)
    keep = _Keep()
    keep_thw = _Keep()
    hooks = [layer.register_forward_hook(keep)]
    if kind == "mvit":
        hooks.append(model.backbone.blocks[-1].register_forward_pre_hook(keep_thw))
    model.eval()
    try:
        with torch.enable_grad():
            logits = model(clip.unsqueeze(0).to(device))
            probs = torch.softmax(logits.detach().float()[0], dim=0).cpu().numpy()
            guess = int(np.argmax(probs))
            targets = [guess]
            if label is not None and label != guess:
                targets.append(label)
            cams = {}
            thw = None
            if kind == "mvit":
                thw = keep_thw.value[1]
            for target in targets:
                # autograd.grad leaves every weight's .grad untouched
                grads = torch.autograd.grad(logits[0, target], keep.value,
                                            retain_graph=True)[0]
                cams[target] = cam_of(keep.value.detach(), grads, kind,
                                      clip.shape[0], clip.shape[-1], thw)
    finally:
        for hook in hooks:
            hook.remove()
    peak = 0.0
    for target in cams:
        peak = max(peak, float(cams[target].max()))
    if peak > 0:
        for target in cams:
            cams[target] = cams[target] / peak
    return probs, cams


def pick(part, count=COUNT):
    """(right, wrong): clip numbers, most confident first, one per true class."""
    probs = part["probs"]
    labels = np.asarray(part["labels"])
    order = []
    for i in range(len(labels)):
        guess = int(np.argmax(probs[i]))
        order.append((-float(probs[i, guess]), i))
    order.sort()
    right = []
    wrong = []
    right_classes = set()
    wrong_classes = set()
    for _, i in order:
        label = int(labels[i])
        if int(np.argmax(probs[i])) == label:
            if len(right) < count and label not in right_classes:
                right.append(i)
                right_classes.add(label)
        elif len(wrong) < count and label not in wrong_classes:
            wrong.append(i)
            wrong_classes.add(label)
    return right, wrong


def _shown(frames):
    """SHOWN frame numbers, evenly spread, first and last included."""
    if frames <= SHOWN:
        return list(range(frames))
    found = []
    for k in range(SHOWN):
        found.append(round(k * (frames - 1) / (SHOWN - 1)))
    return found


def _heat(cam):
    """One frame's map as an orange layer whose strength is the map."""
    layer = np.zeros(cam.shape + (4,))
    red, green, blue = to_rgb(HEAT)
    layer[..., 0] = red
    layer[..., 1] = green
    layer[..., 2] = blue
    layer[..., 3] = cam * STRONGEST
    return layer


def draw(frames, rows, title, path):
    """frames (T, H, W, 3) uint8; rows [(name, map or None)], one picture."""
    shown = _shown(len(frames))
    fig, axes = plt.subplots(len(rows), len(shown), squeeze=False,
                             figsize=(1.55 * len(shown), 1.7 * len(rows) + 0.5))
    for r, (name, cam) in enumerate(rows):
        for c, t in enumerate(shown):
            ax = axes[r][c]
            ax.imshow(frames[t])
            if cam is not None:
                ax.imshow(_heat(cam[t]))
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
            if r == 0:
                ax.set_title(f"frame {t + 1}", fontsize=8, color="#52514e")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def _fresh_folder(folder):
    """Make the folder, and clear the pictures a previous run left there."""
    os.makedirs(folder, exist_ok=True)
    for old in glob.glob(os.path.join(folder, "*.png")):
        os.remove(old)


def _explain(model, data, index, device, folder, number):
    """Draw one clip, return its line for the index."""
    path, label, clip_name = data.items[index]
    clip, _ = data[index]
    probs, cams = grad_cam(model, clip, device, label)
    guess = int(np.argmax(probs))
    classes = data.classes
    rows = [("the crop", None), (f"for {classes[guess]}", cams[guess])]
    if guess != label:
        rows.append((f"for {classes[label]}", cams[label]))
    title = (f"{clip_name}: a {classes[label]}; the model says "
             f"{classes[guess]} {probs[guess]:.0%}")
    picture = f"{number:02d}.png"
    draw(np.load(path), rows, title, os.path.join(folder, picture))
    return (f"{number}. **{clip_name}**: a {classes[label]}, the model says "
            f"{classes[guess]} {probs[guess]:.0%}\n\n"
            f"![clip {number}]({picture})\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=RUN, help="a run's name, seed included")
    ap.add_argument("--clips", nargs="+", help="val clips, as val.csv names them")
    ap.add_argument("--count", type=int, default=COUNT)
    ap.add_argument("--device", help="cpu runs beside training, slowly")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--figures", default=FIGURES)
    args = ap.parse_args(argv)

    parsed = R.parse_name(args.run)
    if parsed is None:
        raise SystemExit(f"{args.run} is not a run's name")
    split = parsed[1]
    split_csv = os.path.join(R.SPLITS[split], "val.csv")
    ckpt = os.path.join(args.out, args.run, "best.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"{ckpt} does not exist")
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    sections = []
    if args.clips:
        sections.append(("Chosen clips", args.clips))
    else:
        cached = val_probs(args.run, args.out, score=False, split=split)
        if cached is None and device.type == "cpu":
            print("picking clips needs every val answer, hours on the CPU; "
                  "use --clips, or run this when the GPU is free")
            return 1
        part = val_probs(args.run, args.out, score=True, split=split)
        right, wrong = pick(part, args.count)
        named = []
        for i in right:
            named.append(part["clips"][i])
        sections.append(("Right, most confident first", named))
        named = []
        for i in wrong:
            named.append(part["clips"][i])
        sections.append(("Wrong, most confident first", named))

    _, data, model, _ = _restore(ckpt, split_csv, device=device)
    where = {}
    for i, item in enumerate(data.items):
        where[item[2]] = i
    for _, names in sections:
        for name in names:
            if name not in where:
                raise SystemExit(f"{name} is not a {split} val clip")

    folder = os.path.join(args.figures, args.run)
    _fresh_folder(folder)
    lines = [f"# Where {args.run} looks\n",
             f"Grad-CAM on {split} val. The top row is the crop the model saw. "
             "Below it, the stronger the orange, the more that spot pushed the "
             "model towards the shot named on the left. A mistake gets a row "
             "for its true shot too. Made by `tools/gradcam.py`; rerun it "
             "rather than editing a picture.\n"]
    number = 0
    for heading, names in sections:
        lines.append(f"## {heading}\n")
        for name in names:
            number += 1
            print(f"{number}: {name}")
            lines.append(_explain(model, data, where[name], device, folder,
                                  number))
    with open(os.path.join(folder, "index.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {number} pictures to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
