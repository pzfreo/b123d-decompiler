"""Stage 3: build plan to build123d source.

Algebra mode, because it maps one-to-one onto the plan and needs no context
manager around every cut. The script is standalone: it imports build123d and
nothing from this package.
"""

from __future__ import annotations

from . import model
from .model import BuildPlan, fmt_tuple

HEADER = '''"""build123d reconstruction of {source}.

Decompiled by b123d-decompiler from quiddity feature recognition.
{summary}
"""

from build123d import *  # noqa: F403
'''


def _summary(plan: BuildPlan, emitted) -> str:
    trims = [op for op in emitted if op.speculative]
    features = len(emitted) - len(trims)
    modelled = len([op for op in plan.ops if not op.speculative])
    lines = [f"{features} operations from {modelled} features this tool can model."]
    if trims:
        lines.append(
            f"{len(trims)} further cuts trim the stock back to flat faces that no\n"
            "feature claimed. Each one proved it removes empty space alone."
        )
    counts = plan.counts()
    for status in (model.INERT, model.OVERLAPS, model.FAILED):
        if counts.get(status):
            words = {
                model.INERT: "changed nothing and were dropped",
                model.OVERLAPS: "cut into material the reference keeps",
                model.FAILED: "could not be built",
            }[status]
            lines.append(f"{counts[status]} {words}.")
    if plan.skipped:
        pairs = ", ".join(f"{name} x{count}" for name, count in sorted(plan.skipped.items()))
        lines.append(f"Not modelled, no adapter: {pairs}.")
    if plan.refusals:
        lines.append(f"{len(plan.refusals)} recognition refusals carried no usable geometry.")
    ratio = (plan.association or {}).get("surface_area", {}).get("ratio")
    if ratio is not None:
        lines.append(f"Quiddity associated {ratio:.0%} of the surface area with a feature.")
    return "\n".join(lines)


def render(
    plan: BuildPlan,
    *,
    keep_frame: bool = False,
    drop_overlapping: bool = False,
    default_output: str = "rebuilt.step",
) -> str:
    emitted = [
        op
        for op in plan.ops
        if op.emitted and not (drop_overlapping and op.status == model.OVERLAPS)
    ]
    out = [HEADER.format(source=plan.source, summary=_summary(plan, emitted)), ""]
    out.append(f"# {plan.stock_label}")
    out.extend(plan.stock_code)

    for op in emitted:
        out.append("")
        flag = "  <-- " + op.note if op.status == model.OVERLAPS else ""
        out.append(f"# {op.family} #{op.index}: {op.label}{flag}")
        if op.note and op.status != model.OVERLAPS:
            out.append(f"# note: {op.note}")
        out.extend(op.code)
        out.append("part -= tool" if op.kind == "cut" else "part += tool")

    if not keep_frame:
        frame = plan.frame
        out += [
            "",
            "# back to the coordinate system of the source file",
            (
                f"part = Plane(origin={fmt_tuple(frame['origin'])}, "
                f"x_dir={fmt_tuple(frame['x'])}, z_dir={fmt_tuple(frame['z'])}) * part"
            ),
        ]

    out += [
        "",
        "",
        'if __name__ == "__main__":',
        "    import sys",
        "",
        f'    export_step(part, sys.argv[1] if len(sys.argv) > 1 else "{default_output}")',
        "",
    ]
    return "\n".join(out)
