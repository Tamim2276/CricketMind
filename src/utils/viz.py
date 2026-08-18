"""
Visualization helpers: confusion matrix heatmap, per-class bar chart.
All figures are saved to disk (for paper/figures/) and not displayed
interactively so they work inside Kaggle notebooks and headless runs.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns


def save_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Saves a normalized confusion matrix heatmap to save_path."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {save_path}")


def save_accuracy_bar(history, save_path, title="Training Accuracy"):
    """Saves a train/val accuracy curve.
    history: {"train_acc": [...], "val_acc": [...]}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, history["train_acc"], label="Train Top-1")
    ax.plot(epochs, history["val_acc"], label="Val Top-1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved accuracy curve → {save_path}")
