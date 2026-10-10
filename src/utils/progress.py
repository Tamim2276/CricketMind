"""A one-line progress bar, rewritten in place.

On a terminal it redraws one line with a carriage return. Anywhere else -- a
log file, a pipe, pytest -- it prints plain lines instead, because a log full
of escape codes and carriage returns is worse than no bar at all.
"""
import sys

__all__ = ["line", "dots", "is_tty", "WIDTH", "GREEN", "RED", "DIM", "OFF"]

WIDTH = 28
GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
OFF = "\033[0m"
FULL = chr(0x25CF)      # a filled circle
EMPTY = chr(0xB7)       # a middle dot


def is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def dots(done: int, total: int, width: int = WIDTH) -> str:
    """The bar itself: green for done, dim for the rest."""
    if total > 0:
        filled = round(done / total * width)
    else:
        filled = width
    filled = max(0, min(width, filled))
    return f"{GREEN}{FULL * filled}{OFF}{DIM}{EMPTY * (width - filled)}{OFF}"


def line(done: int, total: int, tail: str = "", prefix: str = "",
         width: int = WIDTH, tty=None) -> str:
    """One progress line. Starts with \\r and stays on one line on a terminal."""
    if tty is None:
        tty = is_tty()
    if not tty:
        return f"{prefix}{done}/{total}  {tail}"
    if total > 0:
        share = done / total
    else:
        share = 1.0
    return (f"\r{prefix}[{dots(done, total, width)}] {done}/{total} "
            f"{share:5.1%}  {tail} ")
