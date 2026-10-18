"""Which frames does A11's attention pooling lean on, per shot class?

    python -m tools.attention
    python -m tools.attention --exp A11_attnpool

Runs the val clips through each split's best checkpoint, averages the
attention weights by true class, saves them as report_attention.json (kept by
git, so the figure can be redrawn without the checkpoint) and draws one
heatmap per split. Validation only.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import methods
from src.experiments import registry as R
from src.plot import FIGURES, THEMES, sequential_cmap
from src.utils.checkpoint import atomic_write_json
from src.utils.device import get_device

__all__ = ["attention_by_class", "draw", "main"]


def _load(ckpt_path, device):
    from src.train import DEFAULTS

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(DEFAULTS)
    cfg.update(saved["cfg"])
    method = methods.load(cfg["method"])
    return saved, cfg, method


def attention_by_class(ckpt_path, split_csv, device=None):
    """{'classes', 'frames', 'weights' (classes x frames), 'clips'}."""
    device = device or get_device()
    saved, cfg, method = _load(ckpt_path, device)
    data = method.eval_set(cfg, split_csv, saved["classes"])
    model = method.model(cfg, data, len(data.classes))
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    if not hasattr(model, "last_weights"):
        raise ValueError(f"{ckpt_path} is not an attention-pooling head")

    loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=False)
    sums = None
    counts = np.zeros(len(data.classes))
    with torch.no_grad():
        for x, y in loader:
            model(x.to(device))
            w = model.last_weights.float().cpu().numpy()     # (B, T)
            if sums is None:
                sums = np.zeros((len(data.classes), w.shape[1]))
            for i in range(len(y)):
                sums[int(y[i])] += w[i]
                counts[int(y[i])] += 1
    means = np.zeros_like(sums)
    for c in range(len(counts)):
        if counts[c]:
            means[c] = sums[c] / counts[c]
    return {"classes": list(data.classes), "frames": int(sums.shape[1]),
            "weights": means.tolist(), "clips": counts.astype(int).tolist(),
            "checkpoint": os.path.abspath(ckpt_path),
            "split_csv": os.path.abspath(split_csv)}


def draw(panels, path, theme="light"):
    """panels: [(split, result)]. One heatmap per split, peak frame marked."""
    ink = THEMES[theme]
    classes = panels[0][1]["classes"]
    frames = panels[0][1]["frames"]
    top = 0.0
    for split, result in panels:
        top = max(top, float(np.max(result["weights"])))

    height = 2.4 + 0.32 * len(classes)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), height),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(ink["surface"])
    mesh = None
    for k, (split, result) in enumerate(panels):
        ax = axes[0][k]
        ax.set_facecolor(ink["surface"])
        grid = np.array(result["weights"])
        mesh = ax.pcolormesh(grid, cmap=sequential_cmap(ink), vmin=0, vmax=top,
                             edgecolors=ink["surface"], linewidth=1.0)
        # a dot on each class's most-watched frame
        for c in range(len(classes)):
            peak = int(np.argmax(grid[c]))
            ax.plot(peak + 0.5, c + 0.5, marker="o", markersize=4,
                    color=ink["surface"], linestyle="none")
        ax.set_ylim(len(classes), 0)
        ax.set_yticks([c + 0.5 for c in range(len(classes))])
        labels = []
        for c in range(len(classes)):
            labels.append(f"{classes[c]} (n={result['clips'][c]})")
        ax.set_yticklabels(labels, fontsize=7.5, color=ink["secondary"])
        ax.set_xticks([t + 0.5 for t in range(frames)])
        ticks = []
        for t in range(frames):
            ticks.append(str(t + 1))
        ax.set_xticklabels(ticks, fontsize=7.5, color=ink["secondary"])
        ax.tick_params(length=0)
        for side in ax.spines:
            ax.spines[side].set_visible(False)
        ax.set_xlabel("frame", color=ink["secondary"], fontsize=8.5)
        ax.set_title(f"{split} split", color=ink["secondary"], fontsize=9,
                     loc="left", pad=6)

    cax = fig.add_axes([0.93, 1.2 / height, 0.012, 1 - 2.2 / height])
    bar = fig.colorbar(mesh, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=ink["muted"], labelsize=7.5, length=0)
    bar.set_label(f"mean attention weight (even = {1 / frames:.3f})",
                  color=ink["muted"], fontsize=7.5)
    fig.text(0.02, 1 - 0.14 / height, "Which frames A11 attends to",
             color=ink["primary"], fontsize=12.5, ha="left", va="top",
             fontweight="medium")
    fig.text(0.02, 1 - 0.42 / height,
             "validation clips; mean weight per true class; dot = the "
             "most-watched frame", color=ink["muted"], fontsize=8.5,
             ha="left", va="top")
    fig.subplots_adjust(left=0.16, right=0.9, top=1 - 0.95 / height,
                        bottom=0.9 / height, wspace=0.06)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=ink["surface"])
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", default="A11_attnpool")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--figures", default=FIGURES)
    args = ap.parse_args(argv)

    panels = []
    for split in R.SPLITS:
        name = R.run_name(args.exp, split)
        folder = os.path.join(args.out, name)
        ckpt = os.path.join(folder, "best.pt")
        if not os.path.exists(ckpt):
            print(f"{name}: no checkpoint, skipped")
            continue
        csv_path = os.path.join(R.SPLITS[split], "val.csv")
        result = attention_by_class(ckpt, csv_path)
        atomic_write_json(result, os.path.join(folder, "report_attention.json"))
        panels.append((split, result))
        print(f"{name}: {sum(result['clips'])} val clips averaged")
    if not panels:
        return 1
    light = draw(panels, os.path.join(args.figures, f"{args.exp}_attention.png"))
    dark = draw(panels, os.path.join(args.out, R.run_name(args.exp, panels[0][0]),
                                     "attention_dark.png"), theme="dark")
    print(f"wrote {light}\nwrote {dark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
