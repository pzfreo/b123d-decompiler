"""Aggregate a `b123d-decompile batch` run, or compare two of them.

Answers the question that decides what to build next: which unmodelled family is
holding back the most parts, and how much material is still being left behind.

    uv run python tools/corpus_summary.py results/

Given a second directory it prints the difference instead, over the parts both runs
scored. The comparison that matters most is a run against the same corpus with
`--no-trim`, which separates what quiddity's named features rebuild from what tracing
unnamed faces adds on top. Those are different achievements and one number hides which
of them moved.

    uv run b123d-decompile batch corpus/ -o with-tracing/
    uv run b123d-decompile batch corpus/ -o features-only/ --no-trim
    uv run python tools/corpus_summary.py features-only/ with-tracing/
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(results: Path) -> tuple[list[dict], list[dict]]:
    rows = list(csv.DictReader((results / "summary.csv").open()))
    reports = [json.loads(path.read_text()) for path in sorted(results.glob("*/report.json"))]
    return rows, reports


def compare(first: Path, second: Path) -> int:
    """Print how two runs differ over the parts both of them scored."""
    left = {row["file"]: row for row in csv.DictReader((first / "summary.csv").open())}
    right = {row["file"]: row for row in csv.DictReader((second / "summary.csv").open())}
    shared = [
        name for name in left
        if name in right and numeric(left[name]["iou"]) is not None
        and numeric(right[name]["iou"]) is not None
    ]
    if not shared:
        print("the two runs share no scored part")
        return 1

    print(f"{len(shared)} parts scored by both runs\n")
    rows = [
        ("IoU mean", lambda v: sum(v) / len(v)),
        ("IoU median", lambda v: sorted(v)[len(v) // 2]),
    ]
    for label, reduce in rows:
        before = reduce([numeric(left[name]["iou"]) for name in shared])
        after = reduce([numeric(right[name]["iou"]) for name in shared])
        print(f"  {label:22} {before:.3f}  ->  {after:.3f}   {after - before:+.3f}")
    for column, label in (("extra_pct", "extra material, mean"),
                          ("missing_pct", "missing material, mean")):
        before = [numeric(left[n][column]) for n in shared if numeric(left[n][column]) is not None]
        after = [numeric(right[n][column]) for n in shared if numeric(right[n][column]) is not None]
        if before and after:
            first_mean, second_mean = sum(before) / len(before), sum(after) / len(after)
            print(f"  {label:22} {first_mean:7.2f}% -> {second_mean:7.2f}%"
                  f"  {second_mean - first_mean:+7.2f}%")

    moved = sorted(
        shared, key=lambda n: numeric(left[n]["iou"]) - numeric(right[n]["iou"])
    )
    print("\n  most improved:")
    for name in moved[:3]:
        print(f"    {name:34} {numeric(left[name]['iou']):.3f} -> "
              f"{numeric(right[name]['iou']):.3f}")
    regressed = [
        name for name in moved if numeric(right[name]["iou"]) < numeric(left[name]["iou"]) - 1e-6
    ]
    if regressed:
        print("  most regressed:")
        for name in reversed(regressed[-3:]):
            print(f"    {name:34} {numeric(left[name]['iou']):.3f} -> "
                  f"{numeric(right[name]['iou']):.3f}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2:
        return compare(Path(argv[0]), Path(argv[1]))
    if len(argv) != 1:
        print(__doc__)
        return 2
    results = Path(argv[0])
    rows, reports = load(results)
    if not rows:
        print(f"no summary.csv in {results}")
        return 1

    ious = sorted(v for v in (numeric(r["iou"]) for r in rows) if v is not None)
    statuses = collections.Counter(r["status"] for r in rows)
    print(f"{len(rows)} parts: " + ", ".join(f"{k} {v}" for k, v in statuses.most_common()))
    if ious:
        print(
            f"IoU  mean {sum(ious) / len(ious):.4f}  median {ious[len(ious) // 2]:.4f}"
            f"  worst {ious[0]:.4f}  best {ious[-1]:.4f}"
        )
        for limit in (0.9, 0.8, 0.6):
            print(f"  at or above {limit:.1f}: {sum(1 for v in ious if v >= limit)}")

    for column, label in (("missing_pct", "missing material"), ("extra_pct", "extra material")):
        # Skip parts whose intersection could not be computed rather than counting
        # them as zero, which would flatter the average.
        values = [v for v in (numeric(r[column]) for r in rows) if v is not None]
        if not values:
            print(f"{label}: nothing measurable")
            continue
        skipped = len(rows) - len(values)
        tail = f"  ({skipped} not measurable)" if skipped else ""
        print(f"{label}: mean {sum(values) / len(values):.3f}%  worst {max(values):.3f}%{tail}")

    totals = collections.Counter()
    for row in rows:
        for key in ("ops", "emitted", "overlaps", "failed", "inert", "skipped_total"):
            totals[key] += int(row[key] or 0)
    print("ops: " + ", ".join(f"{k} {v}" for k, v in totals.items()))

    skipped = collections.Counter()
    notes = collections.Counter()
    for report in reports:
        skipped.update((report.get("plan") or {}).get("skipped") or {})
        if report.get("boolean_note"):
            notes[report["boolean_note"]] += 1
    if skipped:
        print("\nfamilies with no adapter, most common first:")
        for family, count in skipped.most_common():
            parts = sum(
                1 for r in reports if family in ((r.get("plan") or {}).get("skipped") or {})
            )
            print(f"  {family:24} {count:4} records across {parts} parts")
    if notes:
        print("\nboolean warnings:")
        for note, count in notes.most_common():
            print(f"  {count:3}  {note}")

    ranked = sorted(rows, key=lambda r: numeric(r["iou"]) if numeric(r["iou"]) is not None else -1)
    print("\nworst five:")
    for row in ranked[:5]:
        print(f"  {row['file']:16} IoU {row['iou']:>6}  extra {row['extra_pct']:>7}%"
              f"  skipped {row['skipped_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
