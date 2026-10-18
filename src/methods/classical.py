"""Family D: classical models on cached features, fitted once.

    linear   logistic regression (D01)
    svm      RBF support vector machine (D02)
    rf       random forest, 500 trees (D03)
    xgb      gradient-boosted trees (D04)
    knn      10 nearest neighbours (D05)
    mlp      two hidden layers, 512 and 256 (D06)

Each clip becomes one vector: the 15 frames of the avg view averaged into
1,280 numbers, standardised with the training clips only. No epochs, so no
resume and no flip; the runner calls `fit` instead of train.py's loop, and
`score` writes the same report format as evaluate.py, so the leaderboard,
report cards and figures need no change. Settings are sensible defaults,
fixed for all six: Family D asks how separable the features already are,
not how far each model can be tuned.
"""
import os
import sys
import threading
import time

import joblib
import numpy as np

from src.data.features import FeatureDataset
from src.evaluate import confusion, file_fingerprint, per_class
from src.utils import progress
from src.utils.checkpoint import append_result

NAME = "classical"

READS = ("features", "view", "head", "flip")

HEADS = ("linear", "svm", "rf", "xgb", "knn", "mlp")

CHECKPOINT = "model.joblib"


def check(cfg):
    """Why this config cannot start, or None."""
    if cfg["head"] not in HEADS:
        return f"unknown classical head {cfg['head']!r}"
    if cfg["view"] not in ("avg", "video"):
        return "classical models need one vector per clip: the avg or video view"
    if cfg["flip"]:
        return "a single fit has no epochs to flip across; set flip 0"
    return None


def vectors(cfg, split_csv, classes=None, label=None):
    """(clips, D) numbers and (clips,) labels for one split. With a label,
    a progress line while the files load."""
    data = FeatureDataset(split_csv, features=cfg["features"], view=cfg["view"],
                          classes=classes)
    show = label is not None and progress.is_tty()
    rows = []
    labels = []
    for i in range(len(data)):
        x, y = data[i]
        if x.dim() == 2:
            x = x.mean(dim=0)              # 15 frames -> one vector
        rows.append(x.numpy())
        labels.append(y)
        if show and (i % 50 == 0 or i == len(data) - 1):
            sys.stdout.write(progress.line(i + 1, len(data), prefix=label))
            sys.stdout.flush()
    if show:
        sys.stdout.write("\r" + " " * 100 + "\r")
    return np.stack(rows), np.array(labels), data


def _fit_with_clock(model, x, y, head, quiet):
    """sklearn's fit is one call with nothing to count, so show a clock."""
    if quiet or not progress.is_tty():
        model.fit(x, y)
        return
    done = threading.Event()

    def clock():
        start = time.time()
        while not done.wait(1.0):
            gone = int(time.time() - start)
            sys.stdout.write(f"\r  fitting {head} ... {gone // 60}:{gone % 60:02d}"
                             f" elapsed")
            sys.stdout.flush()

    ticker = threading.Thread(target=clock, daemon=True)
    ticker.start()
    try:
        model.fit(x, y)
    finally:
        done.set()
        ticker.join()
        sys.stdout.write("\r" + " " * 60 + "\r")


def estimator(cfg):
    """The scaler and the model, as one sklearn pipeline."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    seed = cfg["seed"]
    head = cfg["head"]
    if head == "linear":
        model = LogisticRegression(max_iter=3000, random_state=seed)
    elif head == "svm":
        model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    elif head == "rf":
        model = RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                       random_state=seed)
    elif head == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", n_jobs=-1, random_state=seed)
    elif head == "knn":
        model = KNeighborsClassifier(n_neighbors=10)
    else:
        model = MLPClassifier(hidden_layer_sizes=(512, 256), early_stopping=True,
                              max_iter=300, random_state=seed)
    return make_pipeline(StandardScaler(), model)


def _scores(model, x):
    """(clips, classes): probabilities when the model has them, else margins."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(x)
        except AttributeError:          # SVC without probability=True
            pass
    return model.decision_function(x)


def fit(cfg, quiet=False):
    """Fit, save model.joblib, append the ledger row. Returns the row."""
    from src.train import _row_backbone, run_dir, split_name
    from src.utils.seed import set_seed

    set_seed(cfg["seed"])
    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)

    load_label = None
    if not quiet:
        load_label = "  loading train "
    x, y, data = vectors(cfg, os.path.join(cfg["split"], "train.csv"),
                         label=load_label)
    if not quiet:
        load_label = "  loading val   "
    xv, yv, _ = vectors(cfg, os.path.join(cfg["split"], "val.csv"),
                        data.classes, label=load_label)
    if not quiet:
        print(f"{cfg['name']} [classical {cfg['head']}]: {len(y)} train, "
              f"{len(yv)} val, {x.shape[1]} numbers per clip", flush=True)

    started = time.time()
    model = estimator(cfg)
    _fit_with_clock(model, x, y, cfg["head"], quiet)
    minutes = (time.time() - started) / 60
    val_top1 = float((model.predict(xv) == yv).mean())

    joblib.dump({"model": model, "cfg": cfg, "classes": list(data.classes),
                 "val_top1": val_top1}, os.path.join(out, CHECKPOINT))

    row = {"name": cfg["name"], "method": cfg["method"],
           "split": cfg["split"], "split_name": split_name(cfg["split"]),
           "variant": cfg["variant"],
           "features": cfg["features"], "view": cfg["view"],
           "backbone": _row_backbone(cfg), "frozen": cfg["freeze"],
           "best_val_top1": val_top1, "epochs_run": 1,
           "stopped": "fitted once", "minutes": minutes,
           "resumed": False,
           "deterministic": True,       # fixed random_state; tested
           "device": "cpu", "cfg": cfg}
    append_result(row, os.path.join(cfg["out"], "results.jsonl"))
    if not quiet:
        print(f"  fitted in {minutes:.1f} min; val top-1 {val_top1:.2%}")
    return row


def score(ckpt_path, split_csv):
    """The same report evaluate.py writes, for a fitted classical model."""
    saved = joblib.load(ckpt_path)
    cfg = saved["cfg"]
    classes = saved["classes"]
    model = saved["model"]
    part = os.path.splitext(os.path.basename(split_csv))[0]
    x, y, data = vectors(cfg, split_csv, classes,
                         label=f"  scoring {part:<6}")

    scores = _scores(model, x)
    order = np.argsort(-scores, axis=1)
    preds = order[:, 0]
    hits = [0, 0, 0]
    for k in range(3):
        for i in range(len(y)):
            if y[i] in order[i, :k + 1]:
                hits[k] += 1
    loss = None
    if scores.min() >= 0 and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3):
        picked = scores[np.arange(len(y)), y]
        loss = float(-np.log(np.clip(picked, 1e-12, None)).mean())

    matrix = confusion(y, preds, len(classes))
    stats = per_class(matrix)
    return {
        "checkpoint": file_fingerprint(ckpt_path),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_csv": os.path.abspath(split_csv),
        "method": cfg["method"],
        "variant": cfg["variant"],
        "features": cfg["features"],
        "view": cfg["view"],
        "clips": int(len(y)),
        "dropped": len(data.missing),
        "classes": list(classes),
        "loss": loss,
        "top1": hits[0] / len(y),
        "top2": hits[1] / len(y),
        "top3": hits[2] / len(y),
        "macro_f1": float(np.mean(stats["f1"])),
        "macro_precision": float(np.mean(stats["precision"])),
        "macro_recall": float(np.mean(stats["recall"])),
        "per_class": stats,
        "confusion": matrix.tolist(),
        "trained_epoch": 1,
        "trained_val_top1": saved["val_top1"],
    }
