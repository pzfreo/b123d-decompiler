"""Compare two batch summaries part by part, and fail when a part got worse.

    uv run python tools/compare_runs.py baselines/realistic-summary.csv results/summary.csv

Either argument may be a summary.csv or the batch output directory holding one. A part
got worse when its IoU fell by more than the tolerance, when it scored before and does
not now (a crash or a timeout), or when it is missing from the new run. Gains and new
parts are listed too, but never fail the comparison.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

#: IoU a part may lose before the comparison calls it a regression. Sampled measures
#: move by a few ten-thousandths between runs of the same code.
TOLERANCE = 0.005


def load(path: Path) -> dict[str, dict]:
    if path.is_dir():
        path = path / "summary.csv"
    with path.open(newline="") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def _iou(row: dict | None) -> float | None:
    value = (row or {}).get("iou")
    return float(value) if value not in (None, "") else None


def compare(before: dict[str, dict], after: dict[str, dict], tolerance: float = TOLERANCE):
    """(regressions, improvements, notes), each a list of one-line descriptions."""
    regressions, improvements, notes = [], [], []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if new is None:
            regressions.append(f"{name}: missing from the new run")
            continue
        if old is None:
            notes.append(f"{name}: new part, {new.get('status')} {new.get('iou') or ''}".rstrip())
            continue
        was, now = _iou(old), _iou(new)
        if was is not None and now is None:
            regressions.append(f"{name}: scored {was:.4f}, now {new.get('status')}")
        elif was is None and now is not None:
            improvements.append(f"{name}: was {old.get('status')}, now {now:.4f}")
        elif was is not None and now is not None:
            change = now - was
            if change < -tolerance:
                regressions.append(f"{name}: {was:.4f} -> {now:.4f} ({change:+.4f})")
            elif change > tolerance:
                improvements.append(f"{name}: {was:.4f} -> {now:.4f} ({change:+.4f})")
    return regressions, improvements, notes


def _mean(rows: dict[str, dict]) -> str:
    values = [v for v in (_iou(row) for row in rows.values()) if v is not None]
    return f"{sum(values) / len(values):.4f} over {len(values)}" if values else "nothing scored"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args(argv)

    before, after = load(args.before), load(args.after)
    regressions, improvements, notes = compare(before, after, args.tolerance)
    print(f"IoU mean {_mean(before)} -> {_mean(after)}")
    for title, lines in (("worse", regressions), ("better", improvements), ("other", notes)):
        if lines:
            print(f"\n{title} ({len(lines)}):")
            print("\n".join(f"  {line}" for line in lines))
    if not (regressions or improvements or notes):
        print(f"no part moved by more than {args.tolerance}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
