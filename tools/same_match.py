"""Day 16: does sharing a match with training help a val clip, or hurt it?

The author split keeps a clip's own match out of training only for the same
shot type, so nearly every author val clip has clips from its own match in
train, and every one of them is a *different* shot. k-NN shows what that
does: score the same val clips with all train clips, then with each clip's
own match left out. Same features and model as D05.

    python -m tools.same_match

Val only, CPU only, a minute or two.
"""
import csv

import numpy as np

from src.experiments import registry as R
from src.methods.classical import estimator, vectors

K = 10
POOL = 300      # neighbours fetched, so K are left after dropping the match


def matches(split_csv, data):
    """The match of every clip `data` kept, in its order."""
    by_clip = {}
    with open(split_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_clip[row["clip"]] = row["match"]
    out = []
    for _, _, clip in data.items:
        out.append(by_clip[clip])
    return np.array(out)


def vote(labels, n_classes):
    """Most common label; a tie goes to the first class, as sklearn does."""
    return int(np.argmax(np.bincount(labels, minlength=n_classes)))


def compare(neighbours, y_val, m_val, y_train, m_train, n_classes, k=K):
    """neighbours: (val clips, POOL) train indices, nearest first."""
    right_all = 0
    right_without = 0
    nearest_same = 0
    same_count = 0
    same_wrong = 0
    for i in range(len(y_val)):
        nb = neighbours[i]
        same = m_train[nb] == m_val[i]
        top = nb[:k]
        if vote(y_train[top], n_classes) == y_val[i]:
            right_all += 1
        others = nb[~same][:k]
        if vote(y_train[others], n_classes) == y_val[i]:
            right_without += 1
        if same[0]:
            nearest_same += 1
        for j in top[same[:k]]:
            same_count += 1
            if y_train[j] != y_val[i]:
                same_wrong += 1
    n = len(y_val)
    wrong = None
    if same_count:
        wrong = same_wrong / same_count
    return {"clips": n, "top1_all": right_all / n,
            "top1_without_own_match": right_without / n,
            "nearest_is_own_match": nearest_same / n,
            "own_match_share_of_top_k": same_count / (n * k),
            "own_match_neighbours_wrong": wrong}


def measure(split, exp_id="D05_knn"):
    cfg = R.resolve(R.get(exp_id), split, {})
    root = cfg["split"]
    x_tr, y_tr, d_tr = vectors(cfg, f"{root}/train.csv")
    x_va, y_va, d_va = vectors(cfg, f"{root}/val.csv", classes=d_tr.classes)
    model = estimator(cfg)
    model.fit(x_tr, y_tr)
    scaler = model[0]
    knn = model[-1]
    _, neighbours = knn.kneighbors(scaler.transform(x_va), n_neighbors=POOL)
    found = compare(neighbours, y_va, matches(f"{root}/val.csv", d_va), y_tr,
                    matches(f"{root}/train.csv", d_tr), len(d_tr.classes),
                    k=knn.n_neighbors)
    found["sklearn_top1"] = float((model.predict(x_va) == y_va).mean())
    return cfg, found


def main():
    for split in R.SPLITS:
        cfg, f = measure(split)
        print(f"\n{split} val, {f['clips']} clips, k-NN on {cfg['features']} "
              f"{cfg['view']}")
        print(f"  all train clips (D05)            {f['top1_all']:.2%}  "
              f"(sklearn {f['sklearn_top1']:.2%})")
        print(f"  own match left out               "
              f"{f['top1_without_own_match']:.2%}")
        print(f"  nearest clip from its own match  "
              f"{f['nearest_is_own_match']:.1%}")
        print(f"  own-match share of the top {K}     "
              f"{f['own_match_share_of_top_k']:.1%}")
        if f["own_match_neighbours_wrong"] is not None:
            print(f"  ...of which a different shot     "
                  f"{f['own_match_neighbours_wrong']:.1%}")


if __name__ == "__main__":
    main()
