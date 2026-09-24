"""The stages joined up: STEP in, source and a score out."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from quiddity import import_step_geometry

from . import model
from .compare import compare_parts
from .emit import render
from .execute import execute
from .model import BuildPlan
from .plan import build_plan
from .recognise import recognise


def decompile(
    step_path: Path,
    script_path: Path,
    *,
    keep_frame: bool = False,
    drop_overlapping: bool = False,
    verify: bool = True,
    trim_faces: bool = True,
    good_enough: float | None = None,
) -> tuple[BuildPlan, str]:
    document, _caller, local = recognise(str(step_path))
    plan = build_plan(
        document, local, step_path.name, verify=verify, trim_faces=trim_faces,
        good_enough=good_enough,
    )
    source = render(
        plan,
        keep_frame=keep_frame,
        drop_overlapping=drop_overlapping,
        default_output=f"{step_path.stem}.rebuilt.step",
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(source)
    return plan, source


def load_target(target: Path, step_out: Path, timeout: float = 900.0) -> tuple[object, dict]:
    """A .py target is executed; a .step target is read. Either way, a solid comes back.

    A script that runs cleanly can still write a file with nothing in it, and the
    reader rejects that outright. That is a result to report, not an exception to
    let escape into the middle of a batch.
    """
    if target.suffix == ".py":
        outcome = execute(target, step_out, timeout=timeout)
        if outcome["status"] != "ok":
            return None, outcome
        source = step_out
    else:
        source, outcome = target, {"status": "ok", "step": str(target)}
    try:
        return import_step_geometry(str(source)), outcome
    except Exception as error:  # noqa: BLE001 - an unreadable rebuild is an outcome
        return None, dict(outcome, status="invalid_solid",
                          stderr=f"{type(error).__name__}: {error}")


def analyse(
    reference: Path, target: Path, step_out: Path, *, samples: int = 2000, surface: bool = True
) -> dict:
    rebuilt, outcome = load_target(target, step_out)
    if rebuilt is None:
        return {"status": outcome["status"], "execution": outcome}
    # A script can run to the end and still export nothing usable; comparing against
    # that produces numbers rather than an error, which is worse than a failure.
    if not rebuilt.solids() or rebuilt.volume <= 0.0:
        outcome = dict(outcome, stderr="the script exported no solid with volume")
        return {"status": "invalid_solid", "execution": outcome}
    report = compare_parts(
        import_step_geometry(str(reference)), rebuilt, samples=samples, surface=surface
    )
    report["status"] = "ok" if report.get("iou") is not None else "unmeasured"
    report["execution"] = outcome
    return report


def plan_summary(plan: BuildPlan) -> dict:
    counts = plan.counts()
    trims = [op for op in plan.ops if op.speculative]
    return {
        "ops": len(plan.ops),
        "emitted": sum(1 for op in plan.ops if op.emitted),
        "trims": len(trims),
        "trims_kept": sum(1 for op in trims if op.emitted),
        "unproved": counts.get(model.UNPROVED, 0),
        "ok": counts.get(model.OK, 0),
        "overlaps": counts.get(model.OVERLAPS, 0),
        "inert": counts.get(model.INERT, 0),
        "failed": counts.get(model.FAILED, 0),
        "skipped": dict(sorted(plan.skipped.items())),
        "skipped_total": sum(plan.skipped.values()),
        "refusals": len(plan.refusals),
        "association_area": (plan.association or {}).get("surface_area", {}).get("ratio"),
    }


def _code_lines(source: str) -> int:
    """Lines of the script that do something: no docstring, comments or entry point."""
    body = source.split('"""', 2)[-1].split("if __name__")[0]
    return sum(
        1
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith(("#", "from ", "import "))
    )


def readability(plan: BuildPlan, source: str) -> dict:
    """How the rebuild reads, next to how well it matches.

    The aim is the designer's approach, not just the shape, and a score for the shape
    alone rewards tracing: a billet drawn as dozens of extrusions can match a part
    closely and read like nothing anyone would draw. So every result also says how long
    its script is, what the billet is and how long that takes to say, and how many of
    its cuts come from features the recogniser named against faces traced without one.
    """
    label = plan.stock_label
    kind = (
        "turned" if "turned" in label
        else "outline" if "outline" in label
        else "box" if "bounding box" in label
        else "other"
    )
    stock_text = "\n".join(plan.stock_code)
    emitted = [op for op in plan.ops if op.emitted]
    features = sum(1 for op in emitted if not op.speculative)
    trims = sum(1 for op in emitted if op.speculative)
    return {
        "script_lines": _code_lines(source),
        "stock_kind": kind,
        "stock_lines": sum(1 for line in stock_text.splitlines() if line.strip()),
        "feature_ops": features,
        "trim_ops": trims,
        "feature_share": round(features / (features + trims), 3) if features + trims else None,
    }


def run_one(step_path: Path, out_dir: Path, *, samples: int = 1000, **kwargs) -> dict:
    """Decompile and score one file into its own result directory."""
    result_dir = out_dir / step_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"file": step_path.name}
    try:
        plan, source = decompile(step_path, result_dir / f"{step_path.stem}.py", **kwargs)
    except Exception as error:  # noqa: BLE001 - recognition failure is a result
        record["status"] = "recognition_failed"
        record["error"] = f"{type(error).__name__}: {error}"
        (result_dir / "report.json").write_text(json.dumps(record, indent=2))
        return record

    (result_dir / "plan.json").write_text(json.dumps(asdict(plan), indent=2))
    record["plan"] = plan_summary(plan)
    record["readability"] = readability(plan, source)
    record["timings"] = plan.timings
    record["stock_trials"] = plan.stock_trials
    report = analyse(
        step_path,
        result_dir / f"{step_path.stem}.py",
        result_dir / f"{step_path.stem}.rebuilt.step",
        samples=samples,
    )
    record.update(report)
    (result_dir / "report.json").write_text(json.dumps(record, indent=2))
    return record


SUMMARY_COLUMNS = [
    "file", "status", "iou", "missing_pct", "extra_pct", "volume_error_pct",
    "com_offset", "hausdorff_max", "ops", "emitted", "overlaps", "failed",
    "inert", "skipped_total", "association_area",
    "stock_kind", "stock_lines", "script_lines", "feature_ops", "trim_ops", "feature_share",
    "trials", "stock_search_s", "first_trial_s", "extra_trials_s", "plan_s",
]


def summary_row(record: dict) -> dict:
    plan = record.get("plan", {})
    haus = record.get("hausdorff") or {}
    def rounded(value, places=4):
        return round(value, places) if isinstance(value, (int, float)) else None
    return {
        "file": record["file"],
        "status": record.get("status", "?"),
        "iou": rounded(record.get("iou")),
        "missing_pct": rounded(record.get("missing_pct"), 3),
        "extra_pct": rounded(record.get("extra_pct"), 3),
        "volume_error_pct": rounded(record.get("volume_error_pct"), 3),
        "com_offset": rounded(record.get("centre_of_mass_offset")),
        "hausdorff_max": rounded(haus.get("hausdorff")),
        "ops": plan.get("ops"),
        "emitted": plan.get("emitted"),
        "overlaps": plan.get("overlaps"),
        "failed": plan.get("failed"),
        "inert": plan.get("inert"),
        "skipped_total": plan.get("skipped_total"),
        "association_area": rounded(plan.get("association_area"), 3),
        **{key: (record.get("readability") or {}).get(key) for key in (
            "stock_kind", "stock_lines", "script_lines", "feature_ops", "trim_ops",
            "feature_share",
        )},
        "trials": (record.get("timings") or {}).get("trials"),
        "stock_search_s": (record.get("timings") or {}).get("stock_search"),
        "first_trial_s": (record.get("timings") or {}).get("first_trial"),
        "extra_trials_s": (record.get("timings") or {}).get("extra_trials"),
        "plan_s": (record.get("timings") or {}).get("total"),
    }


def write_summary(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(summary_row(record))
