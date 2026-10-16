"""Run experiments from the registry.

    python -m src.experiments.run --list
    python -m src.experiments.run --list --family A
"""
import argparse
import sys

from src.experiments import registry as R

__all__ = ["main", "print_list"]


def print_list(family=None, out=sys.stdout):
    ready = 0
    shown = 0
    for fam in R.FAMILIES:
        if family is not None and fam != family:
            continue
        chosen = R.entries(fam)
        question, day = R.FAMILIES[fam]
        out.write(f"\n{fam}  {question}  (Day {day}, {len(chosen)} IDs)\n")
        for entry in chosen:
            state = R.status(entry)
            if state == "ready":
                ready += 1
            shown += 1
            out.write(f"  {entry['id']:<20} {state:<37} {entry['why']}\n")
    runs = shown * len(R.SPLITS)
    out.write(f"\n{shown} IDs, {runs} runs ({', '.join(R.SPLITS)}), "
              f"{ready} IDs ready now\n")
    return shown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true",
                    help="every experiment, its status and its question")
    ap.add_argument("--family", choices=list(R.FAMILIES))
    args = ap.parse_args(argv)

    if args.list:
        print_list(args.family)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
