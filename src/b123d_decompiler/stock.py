"""The solid a part is cut from, inferred from the part itself.

Quiddity publishes no base-body record, so the billet has to be read off the part.
The candidates are the ones a designer would plausibly start from: bar turned about
an axis, the part's own outline extruded, two outlines intersected, a few extruded
steps, and the bounding box when nothing else holds the whole part.
"""

from __future__ import annotations

import itertools
import math
import time

from build123d import Part

from .geom import Context, dot, is_sound, run_source
from .model import fmt, fmt_tuple

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _envelope_stock(ctx: Context) -> tuple[str, list[str]]:
    size = tuple(ctx.bb_max[k] - ctx.bb_min[k] for k in range(3))
    centre = tuple((ctx.bb_max[k] + ctx.bb_min[k]) / 2 for k in range(3))
    label = "stock: bounding box " + " x ".join(fmt(s, 3) for s in size)
    return label, [f"part = Pos{fmt_tuple(centre)} * Box({', '.join(fmt(s) for s in size)})"]


def _reach_by_band(ctx: Context, axis, anchor, along: float, bands):
    """The furthest the part reaches from an axis over each band.

    Every point is placed in the band it actually falls in, rather than a whole face
    being credited to every band its bounding box touches. A hole through a flange
    would otherwise set the diameter for the plain shaft either side of it.

    Only outward-facing points count. A bolt hole's wall stands at its own radius with
    solid all around it, and counting that would turn the bolt circle into the diameter
    the bar has to be turned from.
    """
    reach = [0.0] * len(bands)
    edges = [start for start, _finish, _radius in bands]

    def place(position: float) -> int | None:
        for index, (start, finish, _radius) in enumerate(bands):
            if start - 1e-9 <= position <= finish + 1e-9:
                return index
        return None

    del edges
    for face in ctx.part.faces():
        points = [vertex.to_tuple() for vertex in face.vertices()]
        for edge in face.edges():
            points.extend(tuple(edge @ (step / 8)) for step in range(9))
        grid = 12
        try:
            surface = [
                tuple(face.position_at(i / grid, j / grid))
                for i in range(grid + 1)
                for j in range(grid + 1)
            ]
        except Exception:  # noqa: BLE001 - a face with no parametrisation
            surface = []  # its edges and vertices still describe it
        points.extend(surface)
        for point in points:
            offset = tuple(point[k] - anchor[k] for k in range(3))
            forward = dot(offset, axis)
            radius = math.sqrt(max(sum(v * v for v in offset) - forward * forward, 0.0))
            if radius <= 1e-9:
                continue
            index = place(forward + along)
            if index is None or radius <= reach[index]:
                continue
            outward = tuple(
                (point[k] - anchor[k] - axis[k] * forward) / radius for k in range(3)
            )
            beyond = tuple(
                point[k] + outward[k] * max(radius * 0.02, 1e-3) for k in range(3)
            )
            if ctx.inside_solid(beyond):
                continue
            reach[index] = radius

    # A band nothing landed in is unknown, not empty. Carry the nearest measurement
    # into it, which keeps the billet whole where sampling was too sparse.
    for index, value in enumerate(reach):
        if value > 0.0:
            continue
        before = next((reach[i] for i in range(index - 1, -1, -1) if reach[i] > 0.0), 0.0)
        after = next(
            (reach[i] for i in range(index + 1, len(reach)) if reach[i] > 0.0), 0.0
        )
        reach[index] = max(before, after)
    return reach


def _measured_profile(ctx: Context, axis, anchor, along: float, low: float, high: float,
                      ceiling: float, slices: int = 160):
    """The part's outermost radius along an axis, as a stepped profile.

    A published turned step reports what its own cylindrical faces turn at, which is
    one number for a length that may hold a flange, a hub and a barrel. Slicing finely
    and taking the furthest any face reaches in each slice follows the real shape, and
    revolving that is the tightest billet a lathe could start from.
    """
    edges = [low + (high - low) * i / slices for i in range(slices + 1)]
    bands = list(itertools.pairwise(edges))
    reach = _reach_by_band(ctx, axis, anchor, along, [(a, b, 0.0) for a, b in bands])
    reach = [min(max(r, ceiling * 1e-3), ceiling) for r in reach]

    merged = []
    for (start, finish), radius in zip(bands, reach, strict=True):
        # Round up to a tenth so a profile that is really one diameter does not come
        # out as a hundred almost-identical rings in the generated source.
        stepped = math.ceil(radius * 10 - 1e-9) / 10
        if merged and abs(merged[-1][2] - stepped) < 1e-9:
            merged[-1] = (merged[-1][0], finish, stepped)
        else:
            merged.append((start, finish, stepped))
    return merged


def _bands_to_code(bands, axis, anchor, along: float) -> list[str]:
    """Source that builds a stepped bar from a list of bands."""
    code = []
    for index, (low, high, radius) in enumerate(bands):
        base = tuple(anchor[k] + axis[k] * (low - along) for k in range(3))
        cylinder = (
            f"Plane(origin={fmt_tuple(base)}, z_dir={fmt_tuple(axis)}) * "
            f"Cylinder({fmt(radius)}, {fmt(high - low)}, "
            f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
        )
        code.append(f"{'part =' if index == 0 else 'part +='} {cylinder}")
    return code


def _turned_stock(profile: dict, ctx: Context) -> tuple[str, list[str]] | None:
    """Bar stock turned to a stepped diameter, if it holds the whole part.

    A bounding box round a shaft is mostly air, and for a flanged spool it is many
    times the part. A turned profile gives the outside diameter over each length of
    the axis, which is the billet the part was actually made from. It is used only
    when it proves it contains every bit of the reference: a hexagon head or a lug
    lies outside any diameter, and then the envelope is the honest starting point.
    """
    steps = profile.get("steps") or []
    if not steps:
        return None
    axis = AXES.get(profile.get("axis", ""))
    if axis is None:
        return None
    anchor = tuple(float(v) for v in (steps[0].get("profile") or {}).get("axis_origin") or (0, 0, 0))
    along = dot(anchor, axis)

    # The published steps rarely reach both ends of the body. Whatever length they do
    # not cover is filled at the largest radius the body has, which keeps the billet
    # round without assuming anything about what is there.
    bands = sorted(
        (float(step["lo"]), float(step["hi"]), float(step["diameter"]) / 2)
        for step in steps
        if float(step["hi"]) - float(step["lo"]) > 1e-4  # a zero-length band is noise
    )
    if not bands:
        return None
    body_low, body_high = ctx.extent_along(axis)
    widest = max(
        ctx.bb_max[k] - ctx.bb_min[k] for k in range(3) if abs(axis[k]) < 0.5
    ) / 2
    filled, edge = [], body_low
    for low, high, radius in bands:
        if low - edge > 1e-6:
            filled.append((edge, low, widest))
        filled.append((low, high, radius))
        edge = max(edge, high)
    if body_high - edge > 1e-6:
        filled.append((edge, body_high, widest))
    filled = [band for band in filled if band[1] - band[0] > (body_high - body_low) * 1e-4]

    # A published step reports the diameter its own cylindrical faces turn at, which
    # can be narrower than the widest thing over that length: a flange sitting inside
    # a plain band is not a step of its own. Measuring the part directly is the only
    # reading that does not depend on exactly where a vertex was placed, and a vertex
    # on a band boundary belongs to the step change rather than to either side of it.
    measured = _reach_by_band(ctx, axis, anchor, along, filled)
    widened = [
        (low, high, min(max(radius, measured[index]), widest))
        for index, (low, high, radius) in enumerate(filled)
    ]

    # Three readings of the same bar: the steps as published, those steps opened out
    # to whatever the part reaches inside them, and a fine profile measured slice by
    # slice. Each is built and asked whether it holds the part, and the smallest one
    # that does is the billet. Comparing the fine profile against the widened steps
    # alone was how a band straddling a step change came to set the diameter for the
    # plain shaft beyond it.
    candidates = [widened, filled]
    fine = _measured_profile(ctx, axis, anchor, along, body_low, body_high, widest)
    if fine:
        candidates.insert(0, fine)

    best = None
    for bands in candidates:
        try:
            trial = _bands_to_code(bands, axis, anchor, along)
            solid = run_source(trial, "part")
        except Exception:  # noqa: BLE001 - a reading that will not build is not used
            continue
        if solid.volume <= 0 or ctx.held_by(solid) < HOLDS:
            continue
        if best is None or solid.volume < best[0]:
            best = (solid.volume, bands)
    if best is None:
        return None
    filled = best[1]

    code = _bands_to_code(filled, axis, anchor, along)
    biggest = max(2 * radius for _low, _high, radius in filled)
    label = (
        f"stock: bar turned to {len(filled)} diameters about {profile['axis']}, "
        f"up to \u00d8{fmt(biggest, 3)}"
    )
    return label, code


def _revolution_axis(ctx: Context):
    """The principal axis most of the part's round faces turn about, if there is one.

    Quiddity publishes a turned profile only for parts it reads as shafts, and a part
    made mostly of cones, tori and revolved faces can come through without one. The
    faces themselves carry their axes, so they are grouped by axis line and weighted
    by area. An axis counts only if it lies along x, y or z and carries at least
    AXIS_SHARE of the part's surface: a single cross hole must not turn a block round.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import (
        GeomAbs_Cone,
        GeomAbs_Cylinder,
        GeomAbs_Sphere,
        GeomAbs_SurfaceOfRevolution,
        GeomAbs_Torus,
    )

    groups: dict = {}
    total = 0.0
    for face in ctx.part.faces():
        area = float(face.area)
        total += area
        try:
            surface = BRepAdaptor_Surface(face.wrapped)
            kind = surface.GetType()
            if kind == GeomAbs_Cylinder:
                line = surface.Cylinder().Axis()
            elif kind == GeomAbs_Cone:
                line = surface.Cone().Axis()
            elif kind == GeomAbs_Torus:
                line = surface.Torus().Axis()
            elif kind == GeomAbs_SurfaceOfRevolution:
                line = surface.AxeOfRevolution()
            elif kind == GeomAbs_Sphere:
                continue  # a sphere turns about any axis through its centre
            else:
                continue
        except Exception:  # noqa: BLE001 - a face without a readable axis is skipped
            continue
        direction = line.Direction()
        vector = (direction.X(), direction.Y(), direction.Z())
        index = max(range(3), key=lambda k: abs(vector[k]))
        if abs(vector[index]) < 0.9999:
            continue  # this tool turns bars about x, y or z only
        location = line.Location()
        point = (location.X(), location.Y(), location.Z())
        across = tuple(round(point[k] / (ctx.diagonal * 1e-3)) for k in range(3) if k != index)
        key = (index, across)
        weight, _ = groups.get(key, (0.0, point))
        groups[key] = (weight + area, point)
    if not groups or total <= 0:
        return None
    (index, _across), (weight, point) = max(groups.items(), key=lambda item: item[1][0])
    if weight < AXIS_SHARE * total:
        return None
    axis = tuple(1.0 if k == index else 0.0 for k in range(3))
    anchor = tuple(0.0 if k == index else point[k] for k in range(3))
    return axis, anchor


#: Share of the surface a common axis must carry before the part is treated as turned.
AXIS_SHARE = 0.4


def _self_turned_stock(ctx: Context):
    """A bar turned about the part's own axis of revolution, measured from the part."""
    found = _revolution_axis(ctx)
    if found is None:
        return None
    axis, anchor = found
    along = dot(anchor, axis)
    body_low, body_high = ctx.extent_along(axis)
    widest = math.dist(ctx.bb_min, ctx.bb_max) / 2
    bands = _measured_profile(ctx, axis, anchor, along, body_low, body_high, widest)
    if not bands:
        return None
    code = _bands_to_code(bands, axis, anchor, along)
    solid = run_source(code, "part")
    if solid.volume <= 0 or ctx.held_by(solid) < HOLDS:
        return None
    name = "xyz"[max(range(3), key=lambda k: axis[k])]
    biggest = max(2 * radius for _low, _high, radius in bands)
    label = (
        f"stock: bar turned to {len(bands)} diameters about {name}, measured from the "
        f"part's own round faces, up to \u00d8{fmt(biggest, 3)}"
    )
    return (label, code), float(solid.volume)


#: Stock must hold at least this share of the part, or it is not stock.
HOLDS = 0.995

#: Most places along an axis a stepped billet may choose its steps from.
SLICE_LIMIT = 120

#: A second silhouette earns its place in the script only by this much.
SHADOW_GAIN = 0.98

#: Seconds to spend building outline billets before settling for what else there is.
STOCK_BUDGET = 240.0

def _slice_gaps(part: Part, index: int, ctx: Context):
    """Where the part's geometry changes along an axis: its vertex positions.

    A step in a stepped billet can only break at one of these, which is where a
    designer's steps are. Returns the slices between them and whether there are few
    enough to be steps at all.
    """
    low, high = ctx.bb_min[index], ctx.bb_max[index]
    marks = sorted({round(v.to_tuple()[index], 4) for v in part.vertices()})
    cuts = [low] + [x for x in marks if low < x < high] + [high]
    gaps = [(a, b) for a, b in itertools.pairwise(cuts) if b - a > (high - low) * 1e-6]
    return gaps, len(gaps) <= SLICE_LIMIT


def _outline_source(outline, ctx: Context):
    """Draw one chosen outline from the part's mesh; a function writing its source.

    The outline is worked out in 2D, from the mesh, with shapely: the shadow of every
    surface, or of each step's stretch, with holes filled, grown by the mesh tolerance
    and simplified. The kernel's own 2D union split outlines into dozens of slivers and
    gluing their prisms back together could stall for minutes; none of that is needed
    now, and the kernel only runs the source that comes out, once.
    """
    from .geom import _mesh_for
    from .inside import DEFLECTION_SHARE
    from .outlines import outline_polygons, polygon_source

    triangles = _mesh_for(ctx.part).triangles
    tolerance = max(ctx.diagonal * DEFLECTION_SHARE, 1e-4)
    index = outline.index
    envelope = (ctx.bb_min, ctx.bb_max)
    if outline.kind == "shadow":
        low, high = ctx.bb_min[index], ctx.bb_max[index]
        pieces = [
            (low, high - low, outline_polygons(triangles, index, tolerance, envelope=envelope))
        ]
    else:
        pieces = [
            (
                low,
                high - low,
                outline_polygons(triangles, index, tolerance, low, high, envelope=envelope),
            )
            for low, high in outline.marks
        ]
    drawn = sum(len(polygons) for _start, _length, polygons in pieces)
    if drawn == 0 or drawn > MOST_EXTRUSIONS:
        return None
    return lambda target: polygon_source(pieces, index, target)


#: Most extrusions an outline billet may take before it stops being the billet a
#: designer would draw and becomes a tracing of the part.
MOST_EXTRUSIONS = 8


#: How many of the most plausible combinations are built and measured.
BUILT_COMBINATIONS = 6


def _silhouette_stock(ctx: Context) -> tuple[str, list[str], float] | None:
    """The billet the part's own outlines cut out: ranked by counting, chosen by measuring.

    Every outline, a shadow and a stepped billet along each axis, is scored against
    rasters read off the part's mesh, and so is every combination of them. A plate is
    its own outline; a plate with something formed into it needs two outlines
    intersected, since each shadow smears the formed region along its whole length; a
    part drawn as a few extrusions is a stepped billet. Anything over MOST_EXTRUSIONS
    extrusions is not a billet anyone would draw, so it is not offered.

    The rasters rank close combinations wrongly often enough to matter: on one part the
    plainest combination, two shadows, was estimated the largest and built the
    smallest. Drawing an outline takes a second or two, so the most plausible few are
    built, measured, and the one with the least volume per extrusion that holds the
    whole part is kept.
    """
    from .geom import _mesh_for
    from .outlines import EXTRUSION_COST, OutlineEstimator, ranked

    marks = {}
    for index in range(3):
        gaps, tiled = _slice_gaps(ctx.part, index, ctx)
        marks[index] = ([g[0] for g in gaps] + [gaps[-1][1]]) if (gaps and tiled) else []
    estimator = OutlineEstimator(_mesh_for(ctx.part), ctx.bb_min, ctx.bb_max, marks)
    combinations = ranked(
        estimator.candidates(), ctx.bb_min, ctx.bb_max, SHADOW_GAIN, MOST_EXTRUSIONS
    )

    stop_at = time.monotonic() + STOCK_BUDGET
    sources: dict = {}
    best, tried = None, 0
    for chosen in combinations:
        if tried >= BUILT_COMBINATIONS or time.monotonic() > stop_at:
            break
        writers = []
        for outline in chosen:
            if id(outline) not in sources:
                try:
                    source = _outline_source(outline, ctx)
                    sources[id(outline)] = source if source and source("_shadow") else None
                except Exception:  # noqa: BLE001 - an outline that will not draw is not used
                    sources[id(outline)] = None
            writers.append(sources[id(outline)])
        if any(writer is None for writer in writers):
            continue
        tried += 1
        code = writers[0]("part")
        for writer in writers[1:]:
            code.extend(writer("_shadow"))
            code.append("part = part & _shadow")
        # Judge the source, not the estimate: curves are written out as chords, and a
        # chord falls inside the arc it stands for.
        try:
            drawn = run_source(code, "part")
        except Exception:  # noqa: BLE001 - a stock that will not run is not stock
            continue
        if drawn is None or drawn.volume <= 0 or not is_sound(drawn):
            continue
        if ctx.held_by(drawn) < HOLDS:
            continue
        extrusions = sum(line.count("extrude(") for line in code)
        cost = float(drawn.volume) * (1 + EXTRUSION_COST * extrusions)
        if best is None or cost < best[0]:
            names = ", ".join(outline.name for outline in chosen)
            best = (cost, f"stock: the part's own outline, {names}", code, float(drawn.volume))
    return None if best is None else best[1:]


def _isolated_silhouette_stock(ctx: Context):
    """The silhouette billet, searched for in a child process with a hard time limit.

    The search is the one place where this tool hands the kernel shapes it built itself
    and lets it grind on them: a single fuse of two slice columns once ran for three
    hours on a sheet-metal part, and another took a process down outright. Neither
    can be interrupted from inside, because the time goes in one call. In a child the
    clock is enforced from outside, and whatever happens the part still gets a billet
    from the candidates that remain.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from OCP.BRepTools import BRepTools

    with tempfile.TemporaryDirectory() as folder:
        shape_path, answer_path = Path(folder) / "part.brep", Path(folder) / "stock.json"
        BRepTools.Write_s(ctx.part.wrapped, str(shape_path))
        command = [
            sys.executable, "-m", "b123d_decompiler.cli", "_stock",
            str(shape_path), str(answer_path),
        ]
        try:
            subprocess.run(
                command, capture_output=True, check=False, timeout=STOCK_BUDGET + 60.0
            )
        except subprocess.TimeoutExpired:
            return None
        if not answer_path.exists():
            return None
        answer = json.loads(answer_path.read_text())
    if answer is None:
        return None
    return answer["label"], answer["code"], answer["volume"]


def silhouette_stock_from_file(shape_path: str, answer_path: str) -> None:
    """The child's half of the above: read the part, search, write what it found."""
    import json
    from pathlib import Path

    from build123d import Compound, Solid
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS, TopoDS_Shape

    shape = TopoDS_Shape()
    BRepTools.Read_s(shape, shape_path, BRep_Builder())
    # Downcast before wrapping: a raw shape handed to Part measures as nothing.
    if shape.ShapeType() == TopAbs_SOLID:
        part = Solid(TopoDS.Solid_s(shape))
    else:
        part = Compound(TopoDS.Compound_s(shape))
    try:
        found = _silhouette_stock(Context(part))
    except Exception:  # noqa: BLE001 - no billet is an answer, the parent falls back
        found = None
    answer = None if found is None else {
        "label": found[0], "code": list(found[1]), "volume": float(found[2]),
    }
    Path(answer_path).write_text(json.dumps(answer))


def stock_source(ctx: Context, document: dict) -> tuple[str, list[str]]:
    """The solid the part is cut from.

    Quiddity publishes no base-body record. A turned profile is the one family that
    describes a whole billet rather than a region of the body, so it is tried first and
    kept only if it contains the part. Otherwise the envelope, which is never too small,
    so everything the rebuild misses shows up as material left behind rather than as a
    hole in the part.
    """
    candidates = []
    for profile in (document.get("derived") or {}).get("turned_profiles") or []:
        try:
            turned = _turned_stock(profile, ctx)
        except Exception:  # noqa: BLE001 - an unusable profile just means the envelope
            continue
        if turned is not None:
            candidates.append((run_source(turned[1], "part").volume, turned))
            break
    if not candidates:
        try:
            own = _self_turned_stock(ctx)
        except Exception:  # noqa: BLE001 - no turned bar is an answer too
            own = None
        if own is not None:
            candidates.append((own[1], own[0]))
    outline = _isolated_silhouette_stock(ctx)
    if outline is not None:
        candidates.append((outline[2], (outline[0], outline[1])))
    if candidates:
        return min(candidates, key=lambda entry: _drawing_cost(*entry))[1]
    return _envelope_stock(ctx)


def _drawing_cost(volume: float, stock) -> float:
    """Volume, charged for every extrusion it takes to draw, as the outline chooser is.

    A turned bar is one profile revolved, however many diameters it steps through, and
    it says the part was turned, which the turned treatments after it depend on. An
    outline that is barely smaller but takes several extrusions is the worse reading:
    on a flanged spool the two were within 0.6 % and the outline took 209 lines to the
    bar's 38, and lost the part four points of IoU.
    """
    from .outlines import EXTRUSION_COST

    _label, code = stock
    pieces = 1 if "turned" in _label else max(1, sum(line.count("extrude(") for line in code))
    return volume * (1 + EXTRUSION_COST * pieces)
