"""Command line: decompile, analyse, batch."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from . import model
from .compare import format_report
from .pipeline import (
    analyse,
    decompile,
    plan_summary,
    run_one,
    summary_row,
    write_summary,
)

SUBCOMMANDS = ("decompile", "analyse", "batch", "_part", "_stock")


def _report_plan(plan, stream) -> None:
    summary = plan_summary(plan)
    if summary["trims"]:
        print(
            f"  {summary['trims_kept']} of {summary['trims']} stock trims proved empty",
            file=stream,
        )
    print(
        f"  {summary['emitted'] - summary['trims_kept']} of"
        f" {summary['ops'] - summary['trims']} features modelled"
        f" ({summary['ok']} clean, {summary['overlaps']} overlapping,"
        f" {summary['inert']} inert, {summary['failed']} failed)",
        file=stream,
    )
    if plan.skipped:
        pairs = ", ".join(f"{k} x{v}" for k, v in sorted(plan.skipped.items()))
        print(f"  no adapter: {pairs}", file=stream)
    if plan.refusals:
        print(f"  {len(plan.refusals)} recognition refusals", file=stream)
    for op in plan.ops:
        if op.status in (model.OVERLAPS, model.FAILED):
            print(f"  {op.status}: {op.family} #{op.index} {op.label} - {op.note}", file=stream)


def cmd_decompile(args) -> int:
    step = Path(args.step)
    script = Path(args.output) if args.output else step.with_suffix(".py")
    plan, _ = decompile(
        step,
        script,
        keep_frame=args.keep_frame,
        drop_overlapping=args.drop_overlapping,
        verify=not args.no_verify,
        trim_faces=not args.no_trim,
    )
    print(f"{script}", file=sys.stderr)
    _report_plan(plan, sys.stderr)
    if args.plan:
        from dataclasses import asdict

        Path(args.plan).write_text(json.dumps(asdict(plan), indent=2))
    if args.analyse:
        rebuilt = Path(args.rebuilt) if args.rebuilt else script.with_suffix(".rebuilt.step")
        report = analyse(step, script, rebuilt, samples=args.samples)
        if "reference" not in report:
            print(f"  execution {report['status']}: "
                  f"{report['execution'].get('stderr', '')}", file=sys.stderr)
            return 1
        print(format_report(report), file=sys.stderr)
    return 0


def cmd_analyse(args) -> int:
    reference, target = Path(args.reference), Path(args.target)
    rebuilt = Path(args.rebuilt) if args.rebuilt else target.with_suffix(".rebuilt.step")
    report = analyse(reference, target, rebuilt, samples=args.samples,
                     surface=not args.no_surface)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    if "reference" not in report:
        print(f"execution {report['status']}: {report['execution'].get('stderr','')}",
              file=sys.stderr)
        return 1
    # An unmeasurable intersection still leaves volume, envelope and surface readable.
    print(f"{reference.name} vs {target.name}", file=sys.stderr)
    print(format_report(report), file=sys.stderr)
    return 0 if report["status"] == "ok" else 1


def cmd_part(args) -> int:
    """One part, in its own process, so a kernel crash cannot take the batch with it."""
    run_one(
        Path(args.step), Path(args.output), samples=args.samples,
        keep_frame=args.keep_frame, drop_overlapping=args.drop_overlapping,
        trim_faces=not args.no_trim, verify=not args.no_verify,
    )
    return 0


def cmd_stock(args) -> int:
    """The silhouette stock search, run apart so the parent can put a clock on it."""
    from .plan import silhouette_stock_from_file

    silhouette_stock_from_file(args.shape, args.answer)
    return 0


def _run_isolated(step: Path, out_dir: Path, args) -> dict:
    """Run one part in a child process and read back what it wrote.

    Realistic parts can crash the CAD kernel outright rather than raising, and an
    in-process batch dies with it part way through a corpus. A child process turns
    that into one bad row.
    """
    command = [
        sys.executable, "-m", "b123d_decompiler.cli", "_part", str(step), str(out_dir),
        "--samples", str(args.samples),
    ]
    if args.keep_frame:
        command.append("--keep-frame")
    if args.drop_overlapping:
        command.append("--drop-overlapping")
    if args.no_trim:
        command.append("--no-trim")

    report = out_dir / step.stem / "report.json"
    report.unlink(missing_ok=True)  # so a report left by an earlier run is never read back
    # Verification runs the whole sequence of ops as booleans to catch the one that
    # empties the part, and on some geometry that is where the kernel dies rather
    # than raises. A soundness check on the tool does not catch it: the shape the
    # boolean chokes on is the part being built up, not the tool going in. So when
    # the child dies, try once more without that pass. The plan then goes out with
    # its ops unchecked, which the script itself will show, and that is a great deal
    # better than no result at all for the part.
    attempts = []
    for extra in ([], ["--no-verify"]):
        try:
            done = subprocess.run(
                command + extra, capture_output=True, text=True, check=False,
                timeout=args.part_timeout,
            )
        except subprocess.TimeoutExpired:
            # A part that never finishes is a result too, and without a limit it holds
            # the whole corpus: one sheet-metal part once ran for three hours. It is
            # not retried, since there is no reason to think a second run is quicker.
            record = {
                "file": step.name, "status": "timeout",
                "error": f"no result within {args.part_timeout:.0f}s",
            }
            _keep(record, report)
            return record
        if report.exists():
            record = json.loads(report.read_text())
            if extra:
                record["unverified"] = "the kernel crashed while checking the ops"
            return record
        tail = "\n".join(done.stderr.strip().splitlines()[-6:])
        attempts.append(f"exit {done.returncode}{' without verification' if extra else ''}: {tail}")
        if done.returncode >= 0:
            break  # it failed without crashing, so running it again will not help

    reason = "the CAD kernel crashed" if done.returncode < 0 else "no report was written"
    # Keep what the child said. A crash that only happens under load cannot be
    # reproduced afterwards, so this is the only evidence there will be.
    record = {"file": step.name, "status": "crashed", "error": reason, "attempts": attempts}
    _keep(record, report)
    return record


def _keep(record: dict, report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(record, indent=2))


def cmd_batch(args) -> int:
    source = Path(args.directory)
    files = sorted(source.glob("*.step")) + sorted(source.glob("*.stp"))
    if args.limit:
        files = files[: args.limit]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as pool:
        # Each part already runs in its own process, so the only thing to share out is
        # the waiting. Results arrive as they finish rather than in order.
        pending = {pool.submit(_run_isolated, step, out_dir, args): step for step in files}
        for future in concurrent.futures.as_completed(pending):
            step = pending[future]
            record = future.result()
            records.append(record)
            done += 1
            index = done
            row = summary_row(record)
            iou = f"{row['iou']:.4f}" if row["iou"] is not None else "   -  "
            missing = (
                f"{row['missing_pct']:6.2f}%" if row["missing_pct"] is not None else "     -"
            )
            extra = f"{row['extra_pct']:6.2f}%" if row["extra_pct"] is not None else "     -"
            print(
                f"[{index}/{len(files)}] {step.name:<16} {row['status']:<18} IoU {iou}"
                f"  missing {missing}  extra {extra}",
                file=sys.stderr,
                flush=True,
            )
            write_summary(records, out_dir / "summary.csv")
    scored = [r for r in records if isinstance(r.get("iou"), float)]
    print(f"\n{len(scored)}/{len(records)} scored", file=sys.stderr)
    if scored:
        ious = sorted(r["iou"] for r in scored)
        mean = sum(ious) / len(ious)
        print(f"IoU mean {mean:.4f}  median {ious[len(ious)//2]:.4f}"
              f"  worst {ious[0]:.4f}  best {ious[-1]:.4f}", file=sys.stderr)
    print(f"summary: {out_dir / 'summary.csv'}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="b123d-decompile",
        description="Decompile STEP files into build123d source and measure the result.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("decompile", help="STEP to build123d source")
    one.add_argument("step")
    one.add_argument("-o", "--output", help="script path (default: alongside the STEP file)")
    one.add_argument("--plan", help="also write the build plan as JSON")
    one.add_argument("--rebuilt", help="STEP path for --analyse")
    one.add_argument("--keep-frame", action="store_true",
                     help="leave the model in the recognition frame")
    one.add_argument("--drop-overlapping", action="store_true",
                     help="omit cuts that eat material the reference keeps")
    one.add_argument("--no-verify", action="store_true",
                     help="skip per-op validation (faster, no diagnostics)")
    one.add_argument("--no-trim", action="store_true",
                     help="model only recognised features, without trimming the stock "
                          "back to faces no feature claimed")
    one.add_argument("--analyse", action="store_true", help="run and score the result")
    one.add_argument("--samples", type=int, default=2000)
    one.set_defaults(func=cmd_decompile)

    two = subparsers.add_parser("analyse", help="score a reconstruction against a reference")
    two.add_argument("reference")
    two.add_argument("target", help="a .py to execute or a .step to read")
    two.add_argument("--rebuilt", help="where a .py target should export its STEP")
    two.add_argument("--json", help="write the full report here")
    two.add_argument("--samples", type=int, default=2000)
    two.add_argument("--no-surface", action="store_true", help="skip the Hausdorff pass")
    two.set_defaults(func=cmd_analyse)

    three = subparsers.add_parser("batch", help="decompile and score a directory")
    three.add_argument("directory")
    three.add_argument("-o", "--output", default="results")
    three.add_argument("--limit", type=int)
    three.add_argument("--samples", type=int, default=1000)
    three.add_argument("--keep-frame", action="store_true")
    three.add_argument("--drop-overlapping", action="store_true")
    three.add_argument("--no-trim", action="store_true")
    three.add_argument("--jobs", type=int, default=max((os.cpu_count() or 4) // 2, 1),
                       help="parts to run at once (each already has its own process)")
    three.add_argument("--part-timeout", type=float, default=1800.0,
                       help="seconds one part may take before it is recorded as a timeout")
    three.set_defaults(func=cmd_batch)

    part = subparsers.add_parser("_part", help=argparse.SUPPRESS)
    part.add_argument("step")
    part.add_argument("output")
    part.add_argument("--samples", type=int, default=1000)
    part.add_argument("--keep-frame", action="store_true")
    part.add_argument("--drop-overlapping", action="store_true")
    part.add_argument("--no-trim", action="store_true")
    part.add_argument("--no-verify", action="store_true")
    part.set_defaults(func=cmd_part)

    stock = subparsers.add_parser("_stock", help=argparse.SUPPRESS)
    stock.add_argument("shape")
    stock.add_argument("answer")
    stock.set_defaults(func=cmd_stock)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `b123d-decompile part.step` is the common case; let it skip the verb.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "decompile")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
