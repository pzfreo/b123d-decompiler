"""Stage 2: turn a recognition document into a validated build plan.

Every op is validated by executing its own generated source and measuring the
result against the reference solid. A cut whose tool eats material the reference
keeps is wrong, and saying so here is what makes a bad number later traceable to
one adapter rather than to "the pipeline".
"""

from __future__ import annotations

import time

from build123d import Part

from . import model
from .adapters import ADAPTERS, EVIDENCE_ONLY, propose_face_trims
from .geom import (
    Context,
    common_volume,
    is_sound,
    material_volume,
    run_source,
    shared_material,
)
from .model import BuildPlan, Op
from .stock import stock_candidates

#: A cut tool may clip this fraction of its own volume out of material the part
#: keeps before we call it wrong rather than tolerance noise.
OVERLAP_LIMIT = 0.02

#: ...and no more than this fraction of the whole part, whatever the tool's size.
#: Without it a tool made deliberately oversized, as an open profile's closure is,
#: passes the relative test while cutting away a real piece of the part.
PART_LIMIT = 0.002

#: Everything the speculative trims eat between them, together. Each one can sit just
#: inside the per-op limit and still ruin a part when forty of them do it at once, so
#: the cheapest are taken first and the rest are dropped once the budget is gone.
TRIM_BUDGET_SHARE = 0.005


def _build_tools(ops: list[Op]) -> dict[int, Part]:
    """Execute each op's own source. A failure is recorded on the op, not raised."""
    tools: dict[int, Part] = {}
    for position, op in enumerate(ops):
        if op.status != model.PLANNED:
            continue  # an adapter already refused it and said why
        try:
            tool = run_source(op.code)
            op.volume = round(float(tool.volume), 6)
            if op.volume <= 1e-9:
                op.status = model.INERT
                op.note = "; ".join(filter(None, [op.note, "tool has no volume"]))
                continue
            # Check the tool is a sound solid before anything subtracts it. A tool
            # with a self-intersecting face or an unsewn edge measures a plausible
            # volume and then takes the whole process down inside the boolean, which
            # costs the part its result entirely and leaves nothing to report. The
            # check is a fraction of the cost of the boolean it stands in front of.
            if not is_sound(tool):
                op.status = model.FAILED
                op.note = "; ".join(
                    filter(None, [op.note, "the tool is not a sound solid"])
                )
                continue
            tools[position] = tool
        except Exception as error:  # noqa: BLE001 - any failure is a reportable outcome
            op.status = model.FAILED
            op.note = "; ".join(filter(None, [op.note, f"{type(error).__name__}: {error}"]))
    return tools


def _classify_fuse(op: Op, tool: Part, ctx: Context, stock: Part, cut_tools: list[Part]) -> None:
    """Decide whether an additive op earns a place in the script.

    Under bounding-box stock every additive feature already sits inside the stock,
    so the only additive op that can matter is one that puts back material a cut
    removed. Whether it should is a question about the reference, not about which
    family won: if the region the two tools share is material there, the fuse
    restores it; if it is open space, refilling it would be wrong.
    """
    shared = material = 0.0
    for cut in cut_tools:
        overlap, solid = shared_material(tool, cut, ctx.part)
        shared += overlap
        material += solid

    if shared <= 1e-9:
        outside_stock = op.volume - common_volume(tool, stock)
        if outside_stock / op.volume < 1e-3:
            op.status = model.INERT
            op.note = "; ".join(filter(None, [op.note, "already inside the stock"]))
        else:
            op.status = model.OK
        return

    if material / shared > 0.5:
        op.status = model.OK
        op.note = "; ".join(filter(None, [op.note, "restores material removed by a cut"]))
    else:
        op.status = model.INERT
        op.note = "; ".join(
            filter(None, [op.note, "would refill a void the reference keeps open"])
        )


def _spend_trim_budget(plan: BuildPlan, ctx: Context) -> None:
    """Keep the cleanest speculative cuts and drop the rest once they add up.

    Every trim here passed the per-op test on its own. That is not enough: on one NIST
    part forty-two of them each sat just inside the limit and between them took most of
    the part away. Spend a fixed share of the part on tracing, cheapest cuts first.
    """
    budget = TRIM_BUDGET_SHARE * ctx.part.volume
    candidates = sorted(
        (op for op in plan.ops if op.speculative and op.status == model.OK),
        key=lambda op: op.overlap or 0.0,
    )
    spent = 0.0
    for op in candidates:
        cost = op.overlap or 0.0
        if spent + cost > budget:
            op.status = model.UNPROVED
            op.note = "; ".join(
                filter(None, [op.note, "dropped: tracing had already spent its budget"])
            )
            continue
        spent += cost


def _reject_destructive(plan: BuildPlan, tools: dict[int, Part], stock: Part):
    """Apply the ops in order and drop any that leave nothing behind.

    Building a tool and measuring it is not the same as using it. A malformed tool can
    measure plausibly and still empty the part when it is subtracted, and the script
    would then fail at a line the plan called clean. Running the sequence here is the
    only check that matches what the script does.
    """
    order = sorted(
        (position for position, op in enumerate(plan.ops) if plan.ops[position].emitted),
        key=lambda position: (
            plan.ops[position].kind == "fuse",
            -(plan.ops[position].volume or 0.0),
        ),
    )
    part = stock
    for position in order:
        op = plan.ops[position]
        tool = tools.get(position)
        if tool is None:
            continue
        try:
            before = float(part.volume)
            candidate = part - tool if op.kind == "cut" else part + tool
            if not candidate.solids() or candidate.volume <= 0.0:
                raise ValueError("the part would be left empty")
            # A boolean can succeed, keep the volume and still leave a solid the kernel
            # calls invalid, typically a cut whose face lands on one of the part's own.
            # Every op after it inherits the damage, and STEP export silently drops an
            # invalid solid, so the script runs clean and the rebuild comes out empty.
            if not is_sound(candidate):
                raise ValueError("the part would be left malformed")
            # A cut cannot take away more than its own tool, and a fuse cannot add
            # more. When the kernel says otherwise the boolean has come apart, and
            # the sequence has to go on without it.
            moved = abs(before - float(candidate.volume))
            if moved > float(tool.volume) * 1.5 + 1e-6:
                raise ValueError(
                    f"it moved {moved:.4g} mm\u00b3 with a tool of only {tool.volume:.4g}"
                )
        except Exception as error:  # noqa: BLE001 - a destructive op is a reportable outcome
            op.status = model.FAILED
            op.note = "; ".join(
                filter(None, [op.note, f"not applied: {type(error).__name__}: {error}"])
            )
            continue
        part = candidate
    return part


def _accept_unverified(plan: BuildPlan) -> None:
    """Settle ops without checking them against the reference, when that is not possible.

    This is the plan a part gets when checking crashed the kernel. Every tool is still
    built, which needs no boolean against the part, so one that will not build or has
    no volume is left out. The recognised features then go in: they are the design
    intent, and a plan without them is just a billet. The speculative trims stay out,
    since the only thing that made them safe was the check that could not run.
    """
    _build_tools(plan.ops)
    for op in plan.ops:
        if op.status != model.PLANNED:
            continue
        if op.speculative:
            op.status = model.UNPROVED
            op.note = "; ".join(filter(None, [op.note, "not checked, so not used"]))
        else:
            op.status = model.OK
            op.note = "; ".join(filter(None, [op.note, "not checked against the part"]))


def _verify(plan: BuildPlan, ctx: Context):
    """Judge every op against the reference, then against the sequence it belongs to.

    Fuses are settled first because a cut is only wrong where no fuse puts the material
    back, and the whole sequence is run last because an op can be individually sound and
    still ruin the part when it is applied in order.
    """
    stock = run_source(plan.stock_code, "part")
    tools = _build_tools(plan.ops)
    cut_tools = [tools[i] for i, op in enumerate(plan.ops) if op.kind == "cut" and i in tools]

    for position, op in enumerate(plan.ops):
        tool = tools.get(position)
        if tool is not None and op.status == model.PLANNED and op.kind == "fuse":
            _classify_fuse(op, tool, ctx, stock, cut_tools)

    restorers = [
        tools[i]
        for i, op in enumerate(plan.ops)
        if op.kind == "fuse" and op.status == model.OK and i in tools
    ]
    for position, op in enumerate(plan.ops):
        tool = tools.get(position)
        if tool is None or op.kind != "cut" or op.status != model.PLANNED:
            continue
        op.overlap = round(material_volume(tool, ctx.part, restorers), 6)
        if (
            op.overlap / op.volume > OVERLAP_LIMIT
            or op.overlap > PART_LIMIT * ctx.part.volume
        ):
            share = (
                f"cuts {op.overlap:.3g} mm\u00b3 "
                f"({100 * op.overlap / ctx.part.volume:.2f}% of the part) "
                "out of material it keeps"
            )
            op.status = model.UNPROVED if op.speculative else model.OVERLAPS
            if op.speculative:
                share = f"not proved empty: it {share}"
            op.note = "; ".join(filter(None, [op.note, share]))
        else:
            op.status = model.OK

    _spend_trim_budget(plan, ctx)
    return _reject_destructive(plan, tools, stock)


def _cylinder_catalogue(document: dict):
    """Every round feature a turned treatment might sit on: holes, bosses, shafts."""
    from .adapters import AXES

    found = []
    for feature in document["features"]:
        record = feature["record"]
        family = feature["family"]
        if family in ("holes", "bosses") and record.get("diameter"):
            axis = tuple(float(v) for v in record["axis"])
            anchor = tuple(float(v) for v in record["location"])
            found.append((axis, anchor, float(record["diameter"]), family[:-1]))
    for profile in (document.get("derived") or {}).get("turned_profiles") or []:
        axis = AXES.get(profile.get("axis", ""))
        if axis is None:
            continue
        for step in profile.get("steps") or []:
            anchor = tuple(
                float(v) for v in (step.get("profile") or {}).get("axis_origin") or (0, 0, 0)
            )
            found.append((axis, anchor, float(step["diameter"]), "shaft"))
    return found


#: Finished rebuilds this close in IoU are a tie, and the simpler billet takes it.
STOCK_TIE = 0.005

#: Seconds to spend carrying further billets through once the first has finished.
TRIAL_BUDGET = 600.0

#: A rebuild this close to the part is good enough: no further billets are tried.
GOOD_ENOUGH = 0.98


def _best_stock(
    plan: BuildPlan, stocks, ctx: Context, good_enough: float | None = None
) -> BuildPlan:
    """Carry each candidate billet through to a finished rebuild and keep the best.

    The features and trims do not depend on the billet, so they are proposed once;
    checking them does, so each billet gets its own copy of the plan to check. The
    sequence check already applies every op in turn, so what it ends holding is the
    rebuild, and that is measured against the part. Rebuilds within STOCK_TIE of the
    best are a tie, and the billet that is simpler to draw takes it, since that is the
    reading closer to how the part was designed. Every billet tried, and how it ended,
    is kept in the plan.
    """
    import copy

    from .geom import mesh_iou

    good_enough = GOOD_ENOUGH if good_enough is None else good_enough
    from .stock import _complexity

    trials = []
    started = time.monotonic()
    for label, code in stocks:
        # The first billet is always carried through; the rest only while there is time,
        # and not at all once one has ended close enough to the part to leave little to
        # find. On sixteen parts, stopping at GOOD_ENOUGH gave up 0.003 IoU in all.
        if trials and time.monotonic() - started > TRIAL_BUDGET:
            break
        if trials and max(entry[0] for entry in trials) >= good_enough:
            break
        began = time.monotonic()
        trial = copy.deepcopy(plan)
        trial.stock_label, trial.stock_code = label, list(code)
        finished = _verify(trial, ctx)
        if len(stocks) == 1:
            trial.stock_trials = [
                {"stock": label, "iou": None, "chosen": True,
                 "seconds": round(time.monotonic() - began, 1)}
            ]
            return trial
        try:
            score = mesh_iou(ctx.part, finished) if finished is not None else 0.0
        except Exception:  # noqa: BLE001 - a rebuild that will not measure scores nothing
            score = 0.0
        simplicity = 1 if "turned" in label else _complexity(code)
        trials.append((score, simplicity, trial, round(time.monotonic() - began, 1)))
    best = max(entry[0] for entry in trials)
    tied = [entry for entry in trials if entry[0] >= best - STOCK_TIE]
    chosen = min(tied, key=lambda entry: entry[1])[2]
    chosen.stock_trials = [
        {"stock": trial.stock_label, "iou": round(score, 4), "chosen": trial is chosen,
         "seconds": seconds}
        for score, _simplicity, trial, seconds in trials
    ]
    return chosen


def build_plan(
    document: dict,
    local_part: Part,
    source: str,
    *,
    verify: bool = True,
    trim_faces: bool = True,
    good_enough: float | None = None,
) -> BuildPlan:
    began = time.monotonic()
    ctx = Context(local_part)
    ctx.note_cylinders(_cylinder_catalogue(document))
    stocks = stock_candidates(ctx, document)
    stock_seconds = time.monotonic() - began
    stock_label, stock_code = stocks[0]
    plan = BuildPlan(
        source=source,
        frame=document["frame"],
        stock_label=stock_label,
        stock_code=stock_code,
        association=document.get("association", {}),
    )

    refused: set[int] = set()
    modelled: set[int] = set()
    for feature in document["features"]:
        family = feature["family"]
        if feature["record_type"] == "SectionRecessRefusal":
            plan.refusals.append(
                f"section recess refused: {feature['record'].get('reason', 'unknown')}"
            )
            continue
        adapter = ADAPTERS.get(family)
        if adapter is None:
            if family not in EVIDENCE_ONLY:
                plan.skipped[family] = plan.skipped.get(family, 0) + 1
            # A record with nothing to build still marks its faces as claimed, and the
            # trimmer only reads faces nobody claimed. A riser is exactly the wall of a
            # slot or a step that no other record describes, so its faces are handed on
            # to the trimmer rather than left out of both.
            refused.update(feature.get("constituent_faces") or ())
            continue
        op = adapter(feature, ctx)
        if op is None:
            plan.skipped[family] = plan.skipped.get(family, 0) + 1
            refused.update(feature.get("constituent_faces") or ())
            continue
        modelled.update(feature.get("constituent_faces") or ())
        plan.ops.append(op)

    if trim_faces:
        trims, passed_over = propose_face_trims(document, ctx, refused - modelled)
        plan.ops.extend(trims)
        if passed_over:
            plan.skipped["unclaimed_faces"] = passed_over

    proposed = time.monotonic()
    if verify:
        plan = _best_stock(plan, stocks, ctx, GOOD_ENOUGH if good_enough is None else good_enough)
    else:
        _accept_unverified(plan)
    trial_seconds = [trial.get("seconds") or 0.0 for trial in plan.stock_trials]
    plan.timings = {
        "stock_search": round(stock_seconds, 1),
        "features_and_trims": round(proposed - began - stock_seconds, 1),
        "first_trial": round(trial_seconds[0], 1) if trial_seconds else None,
        "extra_trials": round(sum(trial_seconds[1:]), 1),
        "trials": len(trial_seconds),
        "total": round(time.monotonic() - began, 1),
    }

    # Cuts first, coarse to fine, then the additive ops that put material back.
    # Quiddity reports evidence, not history, so there is no recorded order to
    # recover; this one is what a bounding-box stock needs.
    plan.ops.sort(key=lambda op: (op.kind == "fuse", -(op.volume or 0.0)))
    return plan
