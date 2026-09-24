"""Stage 2: turn a recognition document into a validated build plan.

Every op is validated by executing its own generated source and measuring the
result against the reference solid. A cut whose tool eats material the reference
keeps is wrong, and saying so here is what makes a bad number later traceable to
one adapter rather than to "the pipeline".
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import math
import time

from build123d import Part

from . import model
from .adapters import ADAPTERS, EVIDENCE_ONLY, propose_face_trims
from .geom import (
    Context,
    common_volume,
    dot,
    is_sound,
    overlap_after_restore,
    run_source,
    shared_material,
)
from .model import BuildPlan, Op, fmt, fmt_tuple

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

#: Slices to aim for along a silhouette axis, and the most that are worth taking.
SLICE_TARGET = 48
SLICE_LIMIT = 60

#: A second silhouette earns its place in the script only by this much.
SHADOW_GAIN = 0.98

#: Seconds to spend looking for a silhouette billet along one axis, and in total.
#: A part whose shadow is nothing like itself, a bent tube say, can spend minutes on
#: a billet that is still twenty times its own size. Giving up on the clock costs
#: nothing that was going to be worth having.
AXIS_BUDGET = 90.0
STOCK_BUDGET = 240.0

_AXIS_NAMES = ("x", "y", "z")


def _fuse_columns(columns: list, deadline: float | None = None, size=None):
    """Fuse the slice columns into one billet, in pairs and cleaned.

    A chain of fuses re-sews everything built so far at every step, so the last few
    columns each cost as much as all the rest; pairing keeps the shapes small until
    there are only a handful left. Every fuse is cleaned, because the columns overlap
    heavily and the split faces a fuse leaves behind are carried into every boolean
    after it, which is how a tidy billet turns into one that takes minutes to
    intersect with anything. A pair the kernel refuses is left for the next round to
    try against a different partner rather than thrown away, since dropping a column
    puts a hole in the billet.
    """

    measure = size or (lambda shape: shape.volume)

    def join(left, right):
        for attempt in (lambda: left.fuse(right).clean(), lambda: left.fuse(right)):
            try:
                made = attempt()
                if made is not None and measure(made) > 0:
                    return made
            except Exception:  # noqa: BLE001 - a refused fuse is retried or deferred
                continue
        return None

    rounds = 2 * len(columns) + 4
    while len(columns) > 1 and rounds > 0:
        rounds -= 1
        if deadline is not None and time.monotonic() > deadline:
            return None
        merged, deferred = [], []
        for left, right in itertools.zip_longest(columns[::2], columns[1::2]):
            if right is None:
                merged.append(left)
                continue
            made = join(left, right)
            if made is None:
                deferred.extend((left, right))
            else:
                merged.append(made)
        if not merged:
            return None  # nothing in this set will join to anything
        # Rotate the leftovers so a pair that would not join is not offered again.
        columns = merged + deferred[1:] + deferred[:1]
    return columns[0] if len(columns) == 1 else None


def _glue(pieces: list):
    """Join prisms that share faces but never overlap, in one call.

    A flat union comes back as many faces sharing edges, so its prisms touch along
    whole faces. That is the slowest input there is for an ordinary fuse, and one such
    fuse ran for over ten minutes on a part with ten faces. The kernel's glue mode is
    made for exactly this arrangement and treats it as bookkeeping.
    """
    try:
        joined = pieces[0].fuse(*pieces[1:], glue=True)
        with contextlib.suppress(Exception):  # an uncleaned billet is still a billet
            joined = joined.clean()
        # The prisms never overlap, so a sound result holds exactly their total volume.
        # Glue can come back broken without saying so; this is how that shows.
        expected = sum(float(piece.volume) for piece in pieces)
        if not joined.solids() or abs(float(joined.volume) - expected) > 0.01 * expected:
            return None
        return joined
    except Exception:  # noqa: BLE001 - fall back to the ordinary fuse
        return None


def _silhouette_solid(part: Part, index: int, ctx: Context, deadline: float | None = None):
    """The part's own shadow along one axis, run back out to a solid billet.

    Plate and sheet parts are cut from their own outline, so their silhouette is the
    billet rather than a clever approximation of one. It is built by slicing where the
    geometry changes, taking both ends of every slice, and running each cross section
    the full length of the axis. A slab's section cannot change between two consecutive
    vertex positions on a prismatic part, so that is exact there and checked everywhere.
    """
    from build123d import Axis, Box, Pos, extrude, make_face

    axis = (Axis.X, Axis.Y, Axis.Z)[index]
    direction = tuple(1.0 if k == index else 0.0 for k in range(3))
    low, high = ctx.bb_min[index], ctx.bb_max[index]
    reach = (high - low) + 2 * ctx.margin
    oversize = [ctx.bb_max[k] - ctx.bb_min[k] + 2 * ctx.margin for k in range(3)]

    marks = sorted({round(v.to_tuple()[index], 4) for v in part.vertices()})
    cuts = [low] + [x for x in marks if low < x < high] + [high]
    gaps = [(a, b) for a, b in itertools.pairwise(cuts) if b - a > (high - low) * 1e-6]
    if len(gaps) > SLICE_LIMIT:
        gaps = sorted(sorted(gaps, key=lambda gap: gap[0] - gap[1])[:SLICE_LIMIT])
    # Sections are sampled inside each slice, so on a taper or a curve every slice
    # loses the material between its sample and its ends. Splitting the long ones
    # until there are enough of them shrinks that loss without moving the cut planes
    # any closer to the part's own faces.
    while len(gaps) < SLICE_TARGET:
        widest = max(range(len(gaps)), key=lambda i: gaps[i][1] - gaps[i][0])
        begin, stop = gaps[widest]
        if stop - begin <= (high - low) * 1e-4:
            break
        middle = (begin + stop) / 2
        gaps[widest : widest + 1] = [(begin, middle), (middle, stop)]

    # Slice a copy. These booleans widen the tolerances of what they are given, and the
    # part is the reference every later measurement is taken against.
    scratch = copy.deepcopy(part)
    floor = low - ctx.margin
    columns = []
    for start, finish in gaps:
        if deadline is not None and time.monotonic() > deadline:
            return None
        centre = [(ctx.bb_max[k] + ctx.bb_min[k]) / 2 for k in range(3)]
        centre[index] = (start + finish) / 2
        thickness = list(oversize)
        # Half the slice, so the cut planes stay well clear of the part's own faces
        # at either end. Coincident faces make the intersection drop pieces, and a
        # slab that has lost a piece quietly hands back a stock full of holes.
        thickness[index] = (finish - start) * 0.5
        piece = scratch & (Pos(*centre) * Box(*thickness))
        if piece is None or not piece.solids():
            continue
        faces = piece.faces().filter_by(axis)
        if not faces:
            continue
        where = [face.center().to_tuple()[index] for face in faces]
        top, bottom = max(where), min(where)
        for face, position in zip(faces, where, strict=False):
            if min(abs(position - top), abs(position - bottom)) > (finish - start) * 0.51:
                continue
            try:
                # A hole in the section is still solid billet, so only the outer wire.
                filled = make_face(face.outer_wire())
                # Every column runs the same length in the same direction, so the union
                # of the columns is one column of the union of their sections. Laying
                # the sections on one plane and joining them flat is a 2D problem the
                # kernel handles quickly and reliably; joining sixty overlapping solids
                # once took three hours in a single call.
                offset = [0.0, 0.0, 0.0]
                offset[index] = floor - position
                flat = Pos(*offset) * filled
            except Exception:  # noqa: BLE001 - a section that will not build is skipped
                continue
            if flat is not None and flat.area > 0:
                columns.append(flat)
    if not columns:
        return None
    size = tuple(ctx.bb_max[k] - ctx.bb_min[k] for k in range(3))
    centre = tuple((ctx.bb_max[k] + ctx.bb_min[k]) / 2 for k in range(3))
    envelope = Pos(*centre) * Box(*size)

    def extruded(faces):
        try:
            return [extrude(face, amount=reach, dir=direction) for face in faces]
        except Exception:  # noqa: BLE001 - a face that will not extrude spoils the set
            return None

    def flat_first():
        # Join the sections flat, then extrude once and glue the prisms.
        shadow = _fuse_columns(list(columns), deadline, size=lambda shape: shape.area)
        pieces = extruded(shadow.faces()) if shadow is not None else None
        if not pieces:
            return None
        return pieces[0] if len(pieces) == 1 else _glue(pieces)

    def solid_first():
        # The slower way that does not depend on the flat union: a prism per section,
        # joined in 3D. Kept for the parts where the flat union comes back broken.
        pieces = extruded(columns)
        return _fuse_columns(pieces, deadline) if pieces else None

    for build in (flat_first, solid_first):
        try:
            stock = build()
            if stock is None:
                continue
            clipped = stock & envelope
            if clipped.solids() and clipped.volume > 0:
                return clipped
        except Exception:  # noqa: BLE001 - try the other way before giving up
            continue
    return None


def _best_silhouette_axis(ctx: Context) -> list[int]:
    """Rank the axes by how small their shadow is, from the part's mesh.

    Building the exact billet costs a boolean per slice, so on a part with hundreds of
    faces doing it three times over is minutes of work to throw two of them away.
    Rasterising the mesh answers which axis is worth the effort in well under a second.
    """
    from .fingerprint.analyze import decoded_mesh, mesh_shape

    grid = 96
    try:
        vertices, triangles = decoded_mesh(mesh_shape(ctx.part))
    except Exception:  # noqa: BLE001 - without a mesh, try the axes in order
        return [0, 1, 2]
    size = [ctx.bb_max[k] - ctx.bb_min[k] for k in range(3)]
    scored = []
    for index in range(3):
        first, second = (k for k in range(3) if k != index)
        if size[first] <= 0 or size[second] <= 0:
            continue
        cells = set()
        for triangle in triangles:
            points = [vertices[i] for i in triangle]
            us = [(p[first] - ctx.bb_min[first]) / size[first] * grid for p in points]
            vs = [(p[second] - ctx.bb_min[second]) / size[second] * grid for p in points]
            for i in range(max(0, int(min(us))), min(grid, int(max(us)) + 1)):
                for j in range(max(0, int(min(vs))), min(grid, int(max(vs)) + 1)):
                    cells.add((i, j))
        area = len(cells) / (grid * grid) * size[first] * size[second]
        scored.append((area * size[index], index))
    return [index for _, index in sorted(scored)] or [0, 1, 2]


def _prism_source(solid, index: int, ctx: Context, target: str) -> list[str] | None:
    """Source that redraws one silhouette billet, assigned to ``target``."""
    from build123d import Axis

    from .adapters import face_profile_source

    axis = (Axis.X, Axis.Y, Axis.Z)[index]
    outward = tuple(1.0 if k == index else 0.0 for k in range(3))
    caps = solid.faces().filter_by(axis)
    if not caps:
        return None
    top = max(face.center().to_tuple()[index] for face in caps)
    depth = ctx.bb_max[index] - ctx.bb_min[index]

    code, pieces = [], 0
    for face in caps:
        if abs(face.center().to_tuple()[index] - top) > depth * 1e-3:
            continue
        drawn = face_profile_source(face, tuple(-v for v in outward))
        if drawn is None:
            return None
        pieces += 1
        code.extend(drawn[0])
        assign = f"{target} =" if pieces == 1 else f"{target} +="
        code.append(f"{assign} extrude(_plane * make_face(_prof), amount={fmt(depth)})")
    return code if pieces else None


def _silhouette_stock(ctx: Context) -> tuple[str, list[str], float] | None:
    """The billet the part's own shadows cut out, from as many axes as pay their way.

    One shadow is the billet for a plate or a sheet, but a part that is a plate with
    something formed into it has a shadow far larger than itself along every axis: the
    formed region smears the whole length of the sweep. Intersecting the shadows fixes
    that, because a point only survives if every direction saw material there. Each
    shadow contains the part, so their intersection does too, and it is never larger
    than the best single shadow. The price is a boolean and a second outline in the
    script, so an axis has to earn its place by actually taking material away.
    """
    built = []
    stop_at = time.monotonic() + STOCK_BUDGET
    for index in _best_silhouette_axis(ctx):
        if time.monotonic() > stop_at:
            break
        try:
            solid = _silhouette_solid(
                ctx.part, index, ctx, min(stop_at, time.monotonic() + AXIS_BUDGET)
            )
        except Exception:  # noqa: BLE001 - an axis that will not build is simply not used
            continue
        # One axis failing must not cost the others: a shadow with a degenerate face
        # once made a flatness test throw, and the whole search went with it.
        try:
            if solid is None or solid.volume <= 0 or ctx.held_by(solid) < HOLDS:
                continue
            if _prism_source(solid, index, ctx, "_shadow") is None:
                continue  # a shadow this tool cannot draw is a shadow it cannot emit
        except Exception:  # noqa: BLE001 - an axis that will not measure is not used
            continue
        built.append((index, solid))
    if not built:
        return None

    built.sort(key=lambda entry: entry[1].volume)
    kept, current = [built[0]], built[0][1]
    for index, solid in built[1:]:
        try:
            merged = current & solid
        except Exception:  # noqa: BLE001 - a refused intersection just leaves the shadow
            continue
        if merged is None or not merged.solids():
            continue
        if merged.volume > current.volume * SHADOW_GAIN:
            continue  # not enough of a saving to be worth a second outline
        if ctx.held_by(merged) < HOLDS:
            continue
        current, kept = merged, [*kept, (index, solid)]

    code = _prism_source(kept[0][1], kept[0][0], ctx, "part")
    if code is None:
        return None
    for index, solid in kept[1:]:
        more = _prism_source(solid, index, ctx, "_shadow")
        if more is None:
            continue
        code.extend(more)
        code.append("part = part & _shadow")
    # Judge the source, not the solid it was traced from. An outline is written out
    # with curves sampled into chords and coordinates rounded, and a chord always
    # falls inside the arc it stands for, so a billet that held the part before it
    # was drawn can fail to hold it afterwards. That would show up as material
    # missing from the rebuild with nothing in the plan to explain it.
    try:
        drawn = run_source(code, "part")
    except Exception:  # noqa: BLE001 - a stock that will not run is not stock
        return None
    if drawn is None or drawn.volume <= 0 or not is_sound(drawn):
        return None
    if ctx.held_by(drawn) < HOLDS:
        return None

    names = ", ".join(_AXIS_NAMES[index] for index, _ in kept)
    label = (
        f"stock: the part's own outline along {names}"
        if len(kept) > 1
        else f"stock: the part's own outline along {names}, "
        f"{fmt(ctx.bb_max[kept[0][0]] - ctx.bb_min[kept[0][0]], 3)} thick"
    )
    return label, code, drawn.volume


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
        return min(candidates, key=lambda entry: entry[0])[1]
    return _envelope_stock(ctx)


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


def _reject_destructive(plan: BuildPlan, tools: dict[int, Part], stock: Part) -> None:
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


def _verify(plan: BuildPlan, ctx: Context) -> None:
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
        op.overlap = round(overlap_after_restore(tool, ctx.part, restorers), 6)
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
    _reject_destructive(plan, tools, stock)


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


def build_plan(
    document: dict,
    local_part: Part,
    source: str,
    *,
    verify: bool = True,
    trim_faces: bool = True,
) -> BuildPlan:
    ctx = Context(local_part)
    ctx.note_cylinders(_cylinder_catalogue(document))
    stock_label, stock_code = stock_source(ctx, document)
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

    if verify:
        _verify(plan, ctx)

    # Cuts first, coarse to fine, then the additive ops that put material back.
    # Quiddity reports evidence, not history, so there is no recorded order to
    # recover; this one is what a bounding-box stock needs.
    plan.ops.sort(key=lambda op: (op.kind == "fuse", -(op.volume or 0.0)))
    return plan
