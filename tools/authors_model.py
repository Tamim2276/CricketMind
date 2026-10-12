"""Score the authors' own trained model on our preprocessed clips.

This is the experiment that stops the guessing. Our reimplementation reaches
78.14% on their test split where they report 89.09%. The gap is either in the
preprocessing or in the training, and nothing measured so far separates the
two. Their shipped weights run on our crops do:

    ~89%  -> our preprocessing is faithful; the gap is our training
    ~78%  -> our crops are not their crops; look at crop geometry and sampling
    ~7%   -> the weights did not load correctly; ignore the number

That last branch is why no separate weight check is needed. The .keras file
stores its backbone under auto-generated layer names, so `load_weights` falls
back to positional order there and cannot be verified by name. A scrambled
backbone cannot be rescued by any class permutation, so a score well above
chance is itself the proof that loading worked.

**Their class order is unknown.** The file carries no label list, so the
permutation from their output units to our labels is fitted on a sample of
*train* and then applied to test. Fitting it on test would be scoring the
answer against itself.

TensorFlow 2.11+ has no GPU on native Windows and the card is an Intel Arc
anyway, so this is CPU. Measured: about 13 minutes for 1,976 clips.
"""
import argparse
import csv
import json
import os
import sys
import time
import zipfile

import numpy as np

KERAS = os.path.join("CricShoot10kModels",
                     "Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras")
SPLITS = os.path.join("data", "splits_author")
PROCESSED = os.path.join("data", "processed")
NUM_CLASSES = 15


class PositionalLoad(RuntimeError):
    """The weights could only be matched by position, so they are not trusted."""


def _rebuild(cache):
    """Their architecture, rebuilt from what their config.json states."""
    import keras
    from keras import layers

    base = keras.applications.EfficientNetV2S(
        include_top=False, input_shape=(224, 224, 3), weights=None)
    inp = keras.Input((15, 224, 224, 3))
    x = layers.TimeDistributed(base)(inp)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.GRU(128, reset_after=True)(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.99)(x)
    x = layers.Dense(1024, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.load_weights(cache)
    return model


def load_authors_model(path=KERAS, cache=None, allow_positional=False):
    """Load their trained model, preferring the way that cannot go wrong.

    `keras.saving.load_model` rebuilds the graph from their own config and
    matches weights to it, so it is right by construction. It needs **Keras
    3.8.0**, the version that wrote the file (metadata.json); 3.12 raises
    "Cannot convert '15' to a shape" on TimeDistributed(Functional).

    The fallback rebuilds the architecture and calls `load_weights`. That is
    **not safe here**: their config.json names the backbone layers
    canonically (`stem_conv`) while their weights.h5 stores them generically
    (`conv2d`, `batch_normalization`), so nothing matches by name and Keras
    falls back to positional order. Measured, that path scores 16% -- barely
    above the 6.7% chance line -- so it is refused unless asked for.
    """
    import keras

    if cache is None:
        cache = os.path.join(os.path.dirname(path), "_authors.weights.h5")
    if not os.path.exists(cache):
        with zipfile.ZipFile(path) as z:
            data = z.read("model.weights.h5")
        with open(cache, "wb") as fh:
            fh.write(data)

    try:
        model = keras.saving.load_model(path, compile=False)
        print(f"  loaded directly (keras {keras.__version__})")
        return model
    except Exception as err:
        first = str(err).strip().splitlines()[0]
        print(f"  direct load failed: {first}")

    if not allow_positional:
        raise PositionalLoad(
            f"keras {keras.__version__} cannot deserialise this file; it was "
            f"written by 3.8.0. Falling back to load_weights matches by "
            f"position, which scores 16% and means nothing. Run this under "
            f"keras 3.8.0, or pass --allow_positional to see the bad number "
            f"on purpose.")

    print("  WARNING: positional load, the result is not trustworthy")
    return _rebuild(cache)


def read_split(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def balanced_sample(rows, n, seed=0):
    """Across classes, because the CSVs are ordered by class."""
    import random

    by_class = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row)
    rng = random.Random(seed)
    for name in by_class:
        rng.shuffle(by_class[name])

    out = []
    names = sorted(by_class)
    i = 0
    while len(out) < n:
        took = False
        for name in names:
            if i < len(by_class[name]) and len(out) < n:
                out.append(by_class[name][i])
                took = True
        if not took:
            break
        i += 1
    return out


def predict(model, rows, variant, processed, classes, quiet=False):
    """Returns (true label index, their output unit) for each readable clip."""
    truth = []
    units = []
    started = time.time()

    for n, row in enumerate(rows, 1):
        rel = row["clip"].replace("/", os.sep)[:-4] + ".npy"
        path = os.path.join(processed, variant, rel)
        if not os.path.exists(path):
            continue
        clip = np.load(path).astype("float32")
        if clip.shape != (15, 224, 224, 3):
            continue

        prob = model.predict(clip[None], verbose=0)[0]
        truth.append(classes.index(row["class"]))
        units.append(int(prob.argmax()))

        if not quiet and (n % 25 == 0 or n == len(rows)):
            gone = time.time() - started
            left = gone / n * (len(rows) - n)
            print("  %4d/%4d  %.1f min elapsed, ~%.1f min left"
                  % (n, len(rows), gone / 60, left / 60), flush=True)

    return np.array(truth), np.array(units)


def fit_permutation(truth, units, n_classes=NUM_CLASSES):
    """Their output unit -> our label index, by maximum agreement."""
    counts = np.zeros((n_classes, n_classes), dtype=int)
    for t, u in zip(truth, units):
        counts[u, t] += 1

    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-counts)
        mapping = {}
        for u, t in zip(rows, cols):
            mapping[int(u)] = int(t)
        return mapping, counts
    except ImportError:
        pass

    # greedy fallback: take the most confident pairing first
    mapping = {}
    used = set()
    order = np.dstack(np.unravel_index(np.argsort(-counts, axis=None),
                                       counts.shape))[0]
    for u, t in order:
        if int(u) in mapping or int(t) in used:
            continue
        mapping[int(u)] = int(t)
        used.add(int(t))
    return mapping, counts


def score(truth, units, mapping):
    mapped = []
    for u in units:
        mapped.append(mapping.get(int(u), -1))
    mapped = np.array(mapped)
    return float((mapped == truth).mean()), mapped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="box", choices=("box", "seg"))
    ap.add_argument("--fit_clips", type=int, default=450,
                    help="train clips used to fit the class permutation")
    ap.add_argument("--test_clips", type=int, default=0,
                    help="0 means the whole test split")
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--processed", default=PROCESSED)
    ap.add_argument("--out", default=os.path.join("experiments",
                                                  "authors_model.json"))
    ap.add_argument("--allow_positional", action="store_true",
                    help="load by position anyway; the number means nothing")
    args = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    with open(os.path.join(args.splits, "classes.json"), encoding="utf-8") as fh:
        classes = json.load(fh)

    print("loading the authors' weights", flush=True)
    try:
        model = load_authors_model(allow_positional=args.allow_positional)
    except PositionalLoad as err:
        print()
        print(err)
        return 2

    print(f"\nfitting the class permutation on {args.fit_clips} train clips "
          f"(variant={args.variant})", flush=True)
    train = balanced_sample(read_split(os.path.join(args.splits, "train.csv")),
                            args.fit_clips)
    t_truth, t_units = predict(model, train, args.variant, args.processed,
                               classes)
    mapping, counts = fit_permutation(t_truth, t_units)
    fit_acc, _ = score(t_truth, t_units, mapping)
    print(f"  agreement on the fitting sample: {fit_acc:.2%}")

    test = read_split(os.path.join(args.splits, "test.csv"))
    if args.test_clips:
        test = balanced_sample(test, args.test_clips, seed=1)
    print(f"\nscoring {len(test)} test clips", flush=True)
    truth, units = predict(model, test, args.variant, args.processed, classes)
    acc, mapped = score(truth, units, mapping)

    print()
    print("=" * 62)
    print(f"  authors' model on OUR {args.variant} crops")
    print(f"  test top-1  {acc:.2%}   over {len(truth)} clips")
    print("=" * 62)
    print()
    print("  for reference")
    print("    our model, same split, box   78.14%")
    print("    our model, same split, seg   75.51%")
    print("    their paper                  89.09%")
    print()
    if acc < 0.12:
        print("  At chance. Either the load failed or the inputs are wrong;")
        print("  the number says nothing about preprocessing.")
    elif acc > 0.85:
        print("  Near their reported figure: our preprocessing is faithful,")
        print("  and the gap is in our training rather than our crops.")
    elif acc > 0.70:
        print("  Close to our own model's score: our crops are in the same")
        print("  distribution as theirs, so look at training, not cropping.")
    else:
        print("  Well above chance but far below what our own model gets on")
        print("  these same crops. Their model partially transfers, so the")
        print("  frames are cricket but not the frames it was trained on:")
        print("  crop geometry, zoom, padding or frame sampling differ.")

    report = {
        "variant": args.variant,
        "fit_clips": len(t_truth),
        "fit_agreement": fit_acc,
        "test_clips": len(truth),
        "test_top1": acc,
        "mapping": {str(k): v for k, v in sorted(mapping.items())},
        "classes": classes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
