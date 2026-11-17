"""The paper's results: the final system's confusion matrix and per-class
table, and one table of every experiment on both splits.

Reads the reports the runs and tools.final already wrote. Nothing here
scores a model.

    python -m tools.paper_results
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = os.path.join("experiments", "G01_final__grouped", "report_test.json")
FIG_DIR = os.path.join("paper", "cricshot", "figures", "results")
TAB_DIR = os.path.join("paper", "cricshot", "tables")

INK = "#2b2a28"
MUTED = "#6b6a66"


def confusion_figure(report, path):
    classes = report["classes"]
    matrix = np.array(report["confusion"])
    n = len(classes)

    # colour is the share of each true class, so small classes read the same
    shares = np.zeros(matrix.shape)
    for i in range(n):
        total = matrix[i].sum()
        if total:
            shares[i] = matrix[i] / total

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(5.1, 4.9))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            count = matrix[i, j]
            if count == 0:
                continue
            if shares[i, j] > 0.5:
                colour = "white"
            else:
                colour = INK
            ax.text(j, i, str(count), ha="center", va="center",
                    fontsize=5.5, color=colour)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=6.5)
    ax.set_yticklabels(classes, fontsize=6.5)
    ax.set_xlabel("Predicted", fontsize=7.5, color=MUTED)
    ax.set_ylabel("True", fontsize=7.5, color=MUTED)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # thin white lines between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def per_class_table(report, path):
    classes = report["classes"]
    scores = report["per_class"]

    # hardest class first
    order = list(range(len(classes)))
    order.sort(key=lambda i: scores["f1"][i])

    lines = []
    lines.append("% Written by tools/paper_results.py from G01_final__grouped's test report.")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{The final system on the grouped test set, per class, "
                 "hardest first. $n$ is the number of test clips.}")
    lines.append("\\label{tab:per_class}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}lrccc@{}}")
    lines.append("\\toprule")
    lines.append("Shot & $n$ & Precision & Recall & F1 \\\\")
    lines.append("\\midrule")
    for i in order:
        lines.append(f"{classes[i]} & {scores['support'][i]} & "
                     f"{scores['precision'][i]:.3f} & {scores['recall'][i]:.3f} & "
                     f"{scores['f1'][i]:.3f} \\\\")
    lines.append("\\midrule")
    lines.append(f"Macro average & {report['clips']} & "
                 f"{report['macro_precision']:.3f} & {report['macro_recall']:.3f} & "
                 f"{report['macro_f1']:.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# every experiment, by family, in the paper's words
FAMILIES = [
    ("Linear probes on frozen features", [
        ("P01", "EfficientNetV2-S, ImageNet"),
        ("P02", "ConvNeXt-T, ImageNet"),
        ("P03", "R3D-18, Kinetics-400"),
        ("P04", "VideoMAE-B, Kinetics-400"),
        ("P05", "CLIP ViT-B/16"),
    ]),
    ("Temporal heads on frozen EfficientNetV2-S frames", [
        ("A01", "Mean pooling"),
        ("A02", "Max pooling"),
        ("A03", "GRU-128 (the authors' head)"),
        ("A04", "GRU-64"),
        ("A05", "BiGRU-128"),
        ("A06", "LSTM-64"),
        ("A07", "BiLSTM-64"),
        ("A08", "Transformer, 2 layers, 4 heads"),
        ("A09", "Transformer, 4 layers, 8 heads"),
        ("A10", "Temporal convolution (TCN)"),
        ("A11", "Attention pooling"),
        ("A12", "NetVLAD"),
        ("A13", "GRU-128, layer-normalised input"),
        ("A14", "GRU-128 on averaged feature maps"),
    ]),
    ("Classical classifiers on frame-averaged features", [
        ("D01", "Logistic regression"),
        ("D02", "SVM, RBF kernel"),
        ("D03", "Random forest"),
        ("D04", "XGBoost"),
        ("D05", "k-nearest neighbours"),
        ("D06", "MLP"),
    ]),
    ("Frozen backbones with an MLP", [
        ("V01", "R3D-18"),
        ("V02", "MC3-18"),
        ("V03", "R(2+1)D-18"),
        ("V04", "S3D"),
        ("V05", "Video Swin-T"),
        ("V06", "MViTv2-S"),
        ("V07", "VideoMAE-B"),
        ("V08", "TimeSformer-B"),
        ("V09", "CLIP ViT-B/16, per frame"),
        ("V10", "DINOv2-B, per frame"),
    ]),
    ("Image backbones with GRU-128, fine-tuned", [
        ("B01", "EfficientNetV2-S (the authors' model)"),
        ("B02", "EfficientNetV2-B3"),
        ("B03", "Xception"),
        ("B04", "ConvNeXt-T"),
        ("B05", "MobileNetV3-L"),
        ("B06", "ResNet-50"),
        ("B07", "RegNetY-16GF"),
        ("B08", "Swin-T"),
    ]),
    ("Video backbones, fine-tuned", [
        ("C01", "R(2+1)D-18"),
        ("C02", "R3D-18"),
        ("C03", "I3D, ResNet-50"),
        ("C04", "SlowFast, ResNet-50"),
        ("C05", "X3D-M"),
        ("C06", "MViTv2-S"),
        ("C07", "Video Swin-T"),
        ("C08", "VideoMAE-B"),
    ]),
    ("Reproduction of the authors' method", [
        ("R01", "Their recipe on their preprocessing"),
        ("R02", "Their recipe on our preprocessing"),
        ("R03", "Our B01 on their preprocessing"),
    ]),
    ("Pose", [
        ("E01", "Keypoints with GRU-128"),
        ("E02", "ST-GCN on keypoints"),
        ("E03", "A09 and E01, averaged"),
        ("E04", "Cross-attention, pose and RGB"),
    ]),
    ("Bat, ball and motion", [
        ("W01", "Optical flow (RAFT), R3D-18"),
        ("W02", "Bat track, GRU-128"),
        ("W03", "Ball track, GRU-128"),
        ("W04", "Bat, ball and pose tracks"),
        ("W05", "Frame differences, R3D-18"),
    ]),
    ("Multi-scale and hybrid", [
        ("F01", "B01 with multi-scale features"),
        ("F02", "F01 with a transformer head"),
        ("F03", "C02 and W01, averaged"),
        ("F04", "B01 with a second resolution"),
        ("F05", "B01 with attention over GRU steps"),
    ]),
    ("Tuning C06, one change each", [
        ("T01", "Learning rate $3\\times10^{-5}$"),
        ("T02", "Learning rate $3\\times10^{-4}$"),
        ("T03", "Backbone learning rate $\\times0.1$"),
        ("T04", "Backbone learning rate $\\times0.5$"),
        ("T05", "Label smoothing 0.05"),
        ("T06", "Label smoothing 0.1"),
        ("T07", "Class-balanced loss"),
        ("T08", "Stochastic depth 0.4"),
        ("T09", "Head dropout 0.7"),
        ("T10", "AdamW, weight decay 0.05"),
        ("T11", "Colour jitter"),
        ("T12", "Random crop"),
        ("T13", "Time jitter, one frame"),
    ]),
    ("Tuning F02, one change each", [
        ("T14", "Learning rate $3\\times10^{-5}$"),
        ("T15", "Learning rate $3\\times10^{-4}$"),
        ("T16", "Backbone learning rate $\\times0.1$"),
        ("T17", "Backbone learning rate $\\times0.5$"),
        ("T18", "Label smoothing 0.05"),
        ("T19", "Label smoothing 0.1"),
        ("T20", "Class-balanced loss"),
        ("T21", "Stochastic depth 0.4"),
        ("T22", "Head dropout 0.3"),
        ("T23", "AdamW, weight decay 0.01"),
        ("T24", "Colour jitter"),
        ("T25", "Random crop"),
        ("T26", "Time jitter, one frame"),
        ("T27", "20 frames instead of 15"),
    ]),
]

# runs repeated with seeds 1 and 2; the table shows seed 42
REPEATED = ["C04", "C06", "F02", "T07", "T18", "T27"]


def _runs_by_id():
    """{(experiment ID, split): run name} for seed-42 runs in results.jsonl."""
    runs = {}
    with open(os.path.join("experiments", "results.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            name = row.get("name") or row["cfg"]["name"]
            if "__seed" in name or "__" not in name:
                continue
            exp, split = name.split("__", 1)
            runs[(exp.split("_")[0], split)] = name
    return runs


def _top1(run, part):
    """Top-1 in percent from a run's report, or None if it was not scored."""
    if run is None:
        return None
    path = os.path.join("experiments", run, f"report_{part}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["top1"] * 100


def _cell(value):
    if value is None:
        return "--"
    return f"{value:.2f}"


def all_results_table(final_report, path):
    runs = _runs_by_id()

    lines = []
    lines.append("% Written by tools/paper_results.py from every run's reports.")
    lines.append("{\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{longtable}{@{}llcccc@{}}")
    lines.append("\\caption{Every experiment: top-1 accuracy (\\%) on both "
                 "splits. Single runs, no test-time augmentation. Test was "
                 "read once, after the final system was chosen on grouped "
                 "validation. Runs marked $^{\\dagger}$ were repeated with two "
                 "more seeds (Table~\\ref{tab:final_system}); the seed-42 run "
                 "is shown.}\\label{tab:all_results}\\\\")
    header = []
    header.append("\\toprule")
    header.append(" & & \\multicolumn{2}{c}{Author split} & "
                  "\\multicolumn{2}{c}{Grouped split} \\\\")
    header.append("\\cmidrule(lr){3-4}\\cmidrule(l){5-6}")
    header.append("ID & Model & Val & Test & Val & Test \\\\")
    header.append("\\midrule")
    lines.extend(header)
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{6}{@{}l}{\\textit{Table~\\ref{tab:all_results}, "
                 "continued}} \\\\")
    lines.extend(header)
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")

    for family, members in FAMILIES:
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{family}}}}} \\\\")
        for exp_id, label in members:
            author = runs.get((exp_id, "author"))
            grouped = runs.get((exp_id, "grouped"))
            mark = ""
            if exp_id in REPEATED:
                mark = "$^{\\dagger}$"
            lines.append(f"{exp_id}{mark} & {label} & "
                         f"{_cell(_top1(author, 'val'))} & "
                         f"{_cell(_top1(author, 'test'))} & "
                         f"{_cell(_top1(grouped, 'val'))} & "
                         f"{_cell(_top1(grouped, 'test'))} \\\\")
        lines.append("\\addlinespace")

    with open(os.path.join("experiments", "G01_final__grouped",
                           "report_choice.json"), encoding="utf-8") as fh:
        final_val = json.load(fh)["val"]["chosen"] * 100
    lines.append("\\midrule")
    lines.append(f"G01 & \\textbf{{Final: C06 + F02, 3 seeds, flip}} & -- & -- & "
                 f"\\textbf{{{final_val:.2f}}} & "
                 f"\\textbf{{{final_report['top1'] * 100:.2f}}} \\\\")
    lines.append("\\end{longtable}")
    lines.append("}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def top_confusions(report, count=8):
    classes = report["classes"]
    matrix = np.array(report["confusion"])
    mistakes = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                mistakes.append((int(matrix[i, j]), classes[i], classes[j]))
    mistakes.sort(reverse=True)
    return mistakes[:count]


def main():
    with open(REPORT, encoding="utf-8") as fh:
        report = json.load(fh)
    os.makedirs(FIG_DIR, exist_ok=True)

    confusion_figure(report, os.path.join(FIG_DIR, "fig_confusion.pdf"))
    per_class_table(report, os.path.join(TAB_DIR, "tab_per_class.tex"))
    all_results_table(report, os.path.join(TAB_DIR, "tab_all_results.tex"))

    print("most common mistakes (true -> predicted)")
    for count, true, pred in top_confusions(report):
        print(f"  {true:16s} -> {pred:16s} {count}")


if __name__ == "__main__":
    main()
