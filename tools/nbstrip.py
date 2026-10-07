"""Strip outputs and execution counts from notebooks, so git diffs stay readable.

Saved outputs made the old single notebook 1.1 MB, of which almost all was
base64 images and captured stdout; every re-run produced a diff nobody could
review.

Usage:
    python tools/nbstrip.py notebooks/*.ipynb      # rewrite in place
    python tools/nbstrip.py --check notebooks/*.ipynb   # exit 1 if any has output

To strip automatically on every commit, register it once as a git filter:

    git config filter.nbstrip.clean "python tools/nbstrip.py --stdin"
    git config filter.nbstrip.smudge cat

`.gitattributes` already points *.ipynb at that filter.
"""
import argparse
import json
import sys


def strip(nb: dict) -> tuple[dict, int]:
    """Remove outputs and execution counts. Returns (notebook, cells changed)."""
    changed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            changed += 1
        cell["outputs"] = []
        cell["execution_count"] = None
        # these drift on every run and carry no information
        cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb, changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="notebooks to strip")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any notebook carries output")
    ap.add_argument("--stdin", action="store_true",
                    help="read one notebook on stdin, write it to stdout (git filter)")
    args = ap.parse_args()

    if args.stdin:
        nb, _ = strip(json.load(sys.stdin))
        json.dump(nb, sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    dirty = 0
    for path in args.paths:
        with open(path, encoding="utf-8") as fh:
            nb = json.load(fh)
        nb, changed = strip(nb)
        if not changed:
            print(f"clean  {path}")
            continue
        dirty += 1
        if args.check:
            print(f"OUTPUT {path}  ({changed} cells)")
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(nb, fh, indent=1, ensure_ascii=False)
                fh.write("\n")
            print(f"strip  {path}  ({changed} cells)")

    return 1 if (args.check and dirty) else 0


if __name__ == "__main__":
    raise SystemExit(main())
