"""The part-by-part comparison that decides whether a change may merge."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "compare_runs", Path(__file__).parents[1] / "tools" / "compare_runs.py"
)
compare_runs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_runs)


def _row(status: str, iou: str) -> dict:
    return {"status": status, "iou": iou}


def test_a_drop_past_the_tolerance_is_a_regression_and_a_small_one_is_not():
    before = {"a.step": _row("ok", "0.9500"), "b.step": _row("ok", "0.9000")}
    after = {"a.step": _row("ok", "0.9400"), "b.step": _row("ok", "0.8970")}
    regressions, improvements, _ = compare_runs.compare(before, after)
    assert regressions == ["a.step: 0.9500 -> 0.9400 (-0.0100)"]
    assert improvements == []


def test_a_part_that_stops_scoring_or_goes_missing_is_a_regression():
    before = {"a.step": _row("ok", "0.9"), "b.step": _row("ok", "0.8")}
    after = {"a.step": _row("timeout", "")}
    regressions, _, _ = compare_runs.compare(before, after)
    assert regressions == ["a.step: scored 0.9000, now timeout", "b.step: missing from the new run"]


def test_gains_and_new_parts_never_fail():
    before = {"a.step": _row("timeout", "")}
    after = {"a.step": _row("ok", "0.7"), "c.step": _row("ok", "0.5")}
    regressions, improvements, notes = compare_runs.compare(before, after)
    assert regressions == []
    assert improvements == ["a.step: was timeout, now 0.7000"]
    assert notes == ["c.step: new part, ok 0.5"]


def test_a_drop_of_exactly_the_tolerance_is_allowed_and_nan_is_unscored():
    before = {"a.step": _row("ok", "0.9500"), "b.step": _row("ok", "0.9000"), "c.step": _row("ok", "0.8")}
    after = {"a.step": _row("ok", "0.9450"), "b.step": _row("ok", "0.8950"), "c.step": _row("ok", "nan")}
    regressions, _, _ = compare_runs.compare(before, after)
    assert regressions == ["c.step: scored 0.8000, now ok"]


def test_main_exits_non_zero_on_a_regression_and_reads_a_run_directory(tmp_path):
    def write(path: Path, rows: dict) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "status", "iou"])
            writer.writeheader()
            for name, row in rows.items():
                writer.writerow({"file": name, **row})

    write(tmp_path / "before.csv", {"a.step": _row("ok", "0.95")})
    run = tmp_path / "run"
    run.mkdir()
    write(run / "summary.csv", {"a.step": _row("ok", "0.90")})
    assert compare_runs.main([str(tmp_path / "before.csv"), str(run)]) == 1
    write(run / "summary.csv", {"a.step": _row("ok", "0.951")})
    assert compare_runs.main([str(tmp_path / "before.csv"), str(run)]) == 0
