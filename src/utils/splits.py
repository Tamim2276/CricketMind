"""Which match a clip came from, and the two train/val/test splits.

Clips are named `vid<match>_<shot>.avi`, and clips sharing a match share the
batter, ground, lighting and camera. Two split schemes:

  grouped  key = match_id          a whole match goes to one split
  author   key = (match_id, class) reproduces Dihan et al., who kept the same
                                   batter's same shot type off both sides but
                                   let a match straddle otherwise

Both are written, because `author` is what 89.09% is comparable to and
`grouped` is the number to believe.
"""
import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict

__all__ = [
    "match_id", "CLIP_RE", "load_clips", "split_clips",
    "write_splits", "make_all_splits", "SCHEMES",
]

# vid100_41.avi, and vid100_41_flip.avi once Day 7.2 adds augmented copies
CLIP_RE = re.compile(r"^(vid\d+)_(\d+)((?:_[a-z0-9]+)*)\.avi$")

def match_id(name: str) -> str:
    """'Cover Drive/vid100_41.avi' -> 'vid100'. Raises on anything unexpected.

    Raising rather than guessing, because a silently wrong grouping key puts
    clips from one match on both sides of a split and nothing in a results
    table would show it.
    """
    base = os.path.basename(str(name).replace("\\", "/"))
    m = CLIP_RE.match(base)
    if not m:
        raise ValueError(
            f"not a CricShot10k clip name: {name!r} "
            f"(expected vid<digits>_<digits>.avi)"
        )
    return m.group(1)


def _grouped_key(clip: str, cls: str):
    """A whole match goes to one split."""
    return match_id(clip)


def _author_key(clip: str, cls: str):
    """The authors kept one batter's one shot type off both sides, no more."""
    return match_id(clip), cls


SCHEMES = {"grouped": _grouped_key, "author": _author_key}


def load_clips(data_root: str):
    """Every clip as (relpath, class). Classes are the sorted folder names."""
    classes = []
    for name in os.listdir(data_root):
        if os.path.isdir(os.path.join(data_root, name)):
            classes.append(name)
    classes.sort()

    items = []
    for cls in classes:
        for name in sorted(os.listdir(os.path.join(data_root, cls))):
            match_id(name)                      # fail now, not in Phase 3
            items.append((f"{cls}/{name}", cls))
    return items, classes


def split_clips(items, scheme="grouped", test_frac=0.2, val_frac=0.2, seed=42):
    """Assign clips to train/val/test, keeping each group intact.

    `val_frac` is a fraction of what is left after test, matching the authors:
    20% test, then 20% of the remainder for val.

    Groups are placed largest first, each into the split whose *worst-filled*
    class it would leave least full. Exact proportions are impossible once
    groups move whole -- a 53-clip match lands in one place or another -- so
    this gets close and the caller checks how close.
    """
    key_of = SCHEMES[scheme]
    fracs = {
        "test": test_frac,
        "val": (1 - test_frac) * val_frac,
        "train": (1 - test_frac) * (1 - val_frac),
    }

    groups = defaultdict(list)
    for clip, cls in items:
        groups[key_of(clip, cls)].append((clip, cls))

    totals = Counter()
    for _, cls in items:
        totals[cls] += 1

    want = {}
    have = {}
    out = {}
    for split in fracs:
        want[split] = {}
        for cls, n in totals.items():
            want[split][cls] = fracs[split] * n
        have[split] = Counter()
        out[split] = []

    def group_size(key):
        return -len(groups[key])

    # seed decides ties between equal-sized groups; size decides the rest
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    keys.sort(key=group_size)

    for key in keys:
        counts = Counter()
        for _, cls in groups[key]:
            counts[cls] += 1

        # fill ratio: 1.0 means this split has exactly its share of that class.
        # Minimising the worst one keeps all three splits advancing in step --
        # summing the deviations instead fills the smallest target first and
        # leaves train with whatever groups happen to remain.
        best = None
        best_fill = None
        for split in fracs:
            worst = 0.0
            for cls in totals:
                fill = (have[split][cls] + counts[cls]) / want[split][cls]
                if fill > worst:
                    worst = fill
            if best_fill is None or worst < best_fill:
                best_fill = worst
                best = split

        have[best].update(counts)
        out[best].extend(groups[key])

    for split in out:
        out[split].sort()
    return out


def write_splits(out_dir: str, assignment, classes):
    """One CSV per split, plus the frozen class order.

    The label index is written into the file so nothing downstream re-derives
    it from a directory listing -- that is how predictions end up mapped to the
    wrong class names.
    """
    os.makedirs(out_dir, exist_ok=True)
    idx = {}
    for i, cls in enumerate(classes):
        idx[cls] = i
    for split, rows in assignment.items():
        with open(os.path.join(out_dir, f"{split}.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["clip", "class", "label", "match"])
            for clip, cls in rows:
                w.writerow([clip, cls, idx[cls], match_id(clip)])
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as fh:
        json.dump(classes, fh, indent=2)


def leaks(assignment, key=match_id):
    """Groups appearing in more than one split, worst first."""
    where = defaultdict(set)
    for split, rows in assignment.items():
        for clip, _ in rows:
            where[key(clip)].add(split)

    out = {}
    for group, splits in where.items():
        if len(splits) > 1:
            out[group] = sorted(splits)
    return out


def make_all_splits(data_root, out_root="data", test_frac=0.2, val_frac=0.2,
                    seed=42, verbose=True):
    items, classes = load_clips(data_root)
    report = {}
    for scheme in SCHEMES:
        assignment = split_clips(items, scheme, test_frac, val_frac, seed)
        out_dir = os.path.join(out_root, f"splits_{scheme}")
        write_splits(out_dir, assignment, classes)

        sizes = {}
        for split, rows in assignment.items():
            sizes[split] = len(rows)
        straddling = len(leaks(assignment))
        report[scheme] = {"dir": out_dir, "sizes": sizes,
                          "straddling_matches": straddling}

        if verbose:
            print(f"\n{scheme}  -> {out_dir}")
            for split in ("train", "val", "test"):
                n = len(assignment[split])
                print(f"   {split:5s} {n:5d}  ({n / len(items):5.1%})")
            print(f"   matches in >1 split: {straddling}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default="data/CricShoot10kShootDataset")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    make_all_splits(a.data_root, a.out, a.test_frac, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
