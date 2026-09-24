"""One adapter per quiddity family: a record in, a build123d operation out.

Records are deliberately family-specific — some carry a 3D point, others an axis
letter and spans a point has to be rebuilt from — so each family gets explicit
code rather than a clever generic path. A family with no adapter here is reported
as skipped, never silently dropped.
"""

from __future__ import annotations

import itertools
import math

from . import model
from .geom import Context, common_volume, cross, dot, run_source, unit
from .model import Op, fmt, fmt_tuple

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _negate(v):
    return tuple(-float(x) for x in v)


def _segments(boundary, closure, reach: float = 0.0, material_side: str | None = None):
    """Profile points and bulges as (start, end, bulge) segments.

    A bulge at index i applies to the segment i -> i+1, DXF style.

    An open profile has to be closed to make a face. Closing it with a straight chord
    across the opening assumes the surface it was cut into is flat between the two
    ends, and where it is not, the chord cuts through material the part keeps. Running
    the two ends outward instead, away from the profile and past the envelope, only
    ever adds space that is already outside the part.
    """
    points = [tuple(float(v) for v in item["point"]) for item in boundary]
    bulges = [float(item["bulge"]) for item in boundary]
    count = len(points)
    span = count if closure == "closed" else count - 1
    segments = [(points[i], points[(i + 1) % count], bulges[i]) for i in range(span)]
    if closure != "closed":
        segments.extend(_open_closure(points, reach, material_side))
    return [(a, b, g) for a, b, g in segments if math.dist(a, b) > 1e-9]


def _open_closure(points, reach: float, material_side: str | None = None):
    """Segments closing an open profile by running its ends away from the material.

    The record states which side of the directed wall chain stays solid, so the outward
    direction is read rather than guessed. Continuing the traversal from the last point
    to the first keeps material on the same hand, so the empty side is the other one.
    Without that field the profile's own body is used as a stand-in for the solid side.
    """
    first, last = points[0], points[-1]
    chord = (first[0] - last[0], first[1] - last[1])
    length = math.hypot(*chord)
    if length < 1e-9 or reach <= 0.0:
        return [(last, first, 0.0)]  # nothing better available than the chord
    # `normal` is to the left of the closing chord, which runs from the chain's last
    # point back to its first. That is the opposite sense to the chain itself, so the
    # left of the chord is the right of the chain, which is the empty side when the
    # record says material lies to the left.
    normal = (-chord[1] / length, chord[0] / length)
    if material_side in ("left", "right"):
        if material_side == "right":
            normal = (-normal[0], -normal[1])
    else:
        middle = ((first[0] + last[0]) / 2, (first[1] + last[1]) / 2)
        on_side = sum(
            (point[0] - middle[0]) * normal[0] + (point[1] - middle[1]) * normal[1]
            for point in points
        )
        if on_side > 0:
            normal = (-normal[0], -normal[1])
    beyond_last = (last[0] + normal[0] * reach, last[1] + normal[1] * reach)
    beyond_first = (first[0] + normal[0] * reach, first[1] + normal[1] * reach)
    return [
        (last, beyond_last, 0.0),
        (beyond_last, beyond_first, 0.0),
        (beyond_first, first, 0.0),
    ]


def _crosses(segments) -> bool:
    """Does a closed 2D polygon cross itself?

    An open profile is closed here with a straight chord between its two ends. When
    that chord cuts back through the rest of the boundary the face is nonsense, and
    extruding it produces a solid that can delete the part it is subtracted from.
    """

    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lines = [(a, b) for a, b, _ in segments]
    for i, (p1, p2) in enumerate(lines):
        for j in range(i + 2, len(lines)):
            if i == 0 and j == len(lines) - 1:
                continue  # first and last share an endpoint
            p3, p4 = lines[j]
            d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
            d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                return True
    return False


def _profile_source(boundary, closure, reach: float = 0.0,
                    material_side: str | None = None) -> list[str]:
    parts = []
    for start, end, bulge in _segments(boundary, closure, reach, material_side):
        if abs(bulge) > 1e-9:
            # Quiddity's bulge is positive-clockwise; build123d's sagitta is positive
            # to the left of the chord, so the sign inverts. Magnitude is the same:
            # sagitta = |bulge| * chord / 2, both being tan(theta/4) * chord / 2.
            sagitta = -bulge * math.dist(start, end) / 2.0
            parts.append(f"SagittaArc({fmt_tuple(start)}, {fmt_tuple(end)}, {fmt(sagitta)})")
        else:
            parts.append(f"Line({fmt_tuple(start)}, {fmt_tuple(end)})")
    joined = "\n    + ".join(parts)
    return [f"_prof = (\n    {joined}\n)"]


def _axis_name(vector) -> str:
    for name, axis in AXES.items():
        if abs(abs(dot(vector, axis)) - 1.0) < 1e-6:
            return name
    return "oblique"


# ── subtractive ──────────────────────────────────────────────────────


def _recess_code(boundary, closure, reach, base, u_dir, run, length, side=None) -> list[str]:
    return _profile_source(boundary, closure, reach, side) + [
        (
            f"_plane = Plane(origin={fmt_tuple(base)}, x_dir={fmt_tuple(u_dir)}, "
            f"z_dir={fmt_tuple(run)})"
        ),
        f"tool = extrude(_plane * make_face(_prof), amount={fmt(length)})",
    ]


def _best_closure(boundary, closure, geometry, origin, run, u_dir, low, high, ctx,
                  side=None) -> float:
    """How far to run an open profile's ends out before closing it.

    Zero closes it with a straight chord across the opening, which is right when the
    surface it was cut into is flat there and cuts through the part when it is not.
    Running the ends outward instead is right when there is nothing beyond them. The
    record settles which way is outward; how far to go is still worth measuring, so
    both are built and the one that takes less of the part is kept.
    """
    if closure == "closed":
        return 0.0
    base = tuple(origin[k] + run[k] * low for k in range(3))
    best, choice = None, 0.0
    for reach in (0.0, ctx.diagonal):
        segments = _segments(boundary, closure, reach, side)
        if not segments or _crosses(segments):
            continue
        try:
            tool = run_source(
                _recess_code(boundary, closure, reach, base, u_dir, run, high - low, side)
            )
            taken = common_volume(tool, ctx.part)
        except Exception:  # noqa: BLE001 - an unbuildable closure is not chosen
            continue
        if best is None or taken < best:
            best, choice = taken, reach
    return choice


def _empty_reach(piece_code, most: float, ctx: Context) -> float:
    """How far a tool can be carried on past the end of its feature through empty space.

    `piece_code(length)` is the source for just the extension, that many units long.
    The whole of it is tried first, since that is the common case, and failing that
    the farthest length that stays empty in the reference is found by halving.
    """
    from .geom import overlap_after_restore

    def clean(length: float) -> bool:
        try:
            tool = run_source(piece_code(length))
            volume = float(tool.volume)
            if volume <= 0:
                return False
            return overlap_after_restore(tool, ctx.part, []) <= _OPEN_END_TOLERANCE * volume
        except Exception:  # noqa: BLE001 - an extension that will not build is not taken
            return False

    if most <= 0:
        return 0.0
    if clean(most):
        return most
    near, far = 0.0, most
    for _ in range(8):
        middle = (near + far) / 2
        if clean(middle):
            near = middle
        else:
            far = middle
    return near


def _open_end_reach(boundary, closure, side, origin, run, u_dir, at, sign, most, ctx) -> float:
    """How far past an open end the recess can run before it meets material again.

    "Open" says there is space just beyond the end, not that the space goes on to the
    outside of the part. A short channel can open into a pocket or a step, and running
    it on to the envelope carries it straight through whatever lies past that.
    """

    def piece(length: float) -> list[str]:
        start = at if sign > 0 else at - length
        base = tuple(origin[k] + run[k] * start for k in range(3))
        return _recess_code(boundary, closure, 0.0, base, u_dir, run, length, side)

    return _empty_reach(piece, most, ctx)


#: An open end's extension may touch this share of its own volume in material and
#: still count as running through space: the recess wall and the part's face coincide.
_OPEN_END_TOLERANCE = 0.02


def section_recess(feature: dict, ctx: Context) -> Op | None:
    record = feature["record"]
    geometry = record["geometry"]
    frame = geometry["frame"]
    origin = tuple(float(v) for v in frame["origin"])
    run, u_dir = unit(frame["run"]), unit(frame["u"])
    low, high = (float(v) for v in geometry["run_interval"])
    notes = []

    box_low, box_high = ctx.extent_along(run)
    boundary = geometry["profile"]["boundary"]
    closure = geometry["profile"]["closure"]
    side = geometry["profile"].get("material_side")
    recorded = (low, high)
    for which in ("low", "high"):
        end = geometry["ends"][which]
        if end["condition"] == "open":
            if which == "low":
                reach = _open_end_reach(
                    boundary, closure, side, origin, run, u_dir, recorded[0],
                    -1.0, recorded[0] - (box_low - ctx.margin), ctx,
                )
                low = recorded[0] - reach
            else:
                reach = _open_end_reach(
                    boundary, closure, side, origin, run, u_dir, recorded[1],
                    1.0, (box_high + ctx.margin) - recorded[1], ctx,
                )
                high = recorded[1] + reach
            if reach < (recorded[0] - box_low if which == "low" else box_high - recorded[1]):
                notes.append(f"{which} end open onto a space inside the part, run stops there")
            continue
        surface = end["surface"]
        gradient = surface.get("gradient") or (0.0, 0.0)
        if surface.get("type") != "plane":
            notes.append(f"{side} cap is a {surface.get('type')}, modelled flat")
        elif max(abs(float(g)) for g in gradient) > 1e-9:
            notes.append(f"{side} cap is sloped, modelled flat")

    # dot(origin, run) is zero by construction, so the run interval is absolute.
    reach = _best_closure(
        boundary, closure, geometry, origin, run, u_dir, low, high, ctx, side
    )
    segments = _segments(boundary, closure, reach, side)
    if _crosses(segments):
        return Op(
            "cut", "section_recesses", feature["index"],
            "open section that closes onto itself", [], status=model.FAILED,
            note="the straight chord across the open ends crosses the profile",
        )

    base = tuple(origin[k] + run[k] * low for k in range(3))
    classification = record.get("classification", {})
    kind = classification.get("feature_kind", "recess")
    shape = classification.get("section_shape", "?")
    label = (
        f"{kind}: {shape} section, run {_axis_name(run)} "
        f"{fmt(low, 3)}..{fmt(high, 3)}"
    )
    code = _profile_source(boundary, closure, reach, side) + [
        (
            f"_plane = Plane(origin={fmt_tuple(base)}, x_dir={fmt_tuple(u_dir)}, "
            f"z_dir={fmt_tuple(run)})"
        ),
        f"tool = extrude(_plane * make_face(_prof), amount={fmt(high - low)})",
    ]
    return Op("cut", "section_recesses", feature["index"], label, code, note="; ".join(notes))


def hole(feature: dict, ctx: Context) -> Op | None:
    """A drilled hole, with its counterbore, spotface or countersink if it has one.

    Quiddity's hole axis points from the opening into the part, and `location` is the
    opening centre. Everything is anchored to that opening, because a counterbore sits
    there whatever the bore does at the far end.
    """
    record = feature["record"]
    axis = unit(record["axis"])
    location = tuple(float(v) for v in record["location"])
    diameter, depth = float(record["diameter"]), float(record["depth"])
    bottom = record["bottom"]
    notes = []

    if bottom == "through":
        # "Through" is through the wall the hole is drilled in, which the depth gives,
        # not through the whole part: a bolt hole in a flange runs straight on into
        # whatever lies behind the flange. Carry the bore on past each end only as far
        # as the space stays empty, which on a plain plate is the envelope anyway.
        low_extent, high_extent = ctx.extent_along(axis)
        start = dot(location, axis)
        radius = diameter / 2

        def piece(origin, direction):
            def code(length: float) -> list[str]:
                return [
                    f"_plane = Plane(origin={fmt_tuple(origin)}, z_dir={fmt_tuple(direction)})",
                    (
                        f"tool = _plane * Cylinder({fmt(radius)}, {fmt(length)}, "
                        f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
                    ),
                ]

            return code

        exit_point = tuple(location[k] + axis[k] * depth for k in range(3))
        before = _empty_reach(
            piece(location, _negate(axis)), start - low_extent + ctx.margin, ctx
        )
        after = _empty_reach(
            piece(exit_point, axis), high_extent - (start + depth) + ctx.margin, ctx
        )
        offset = -before
        length = before + depth + after
    else:
        forward = ctx.void_score(location, axis, depth, diameter / 2)
        backward = ctx.void_score(location, _negate(axis), depth, diameter / 2)
        if backward > forward:
            axis = _negate(axis)
        elif forward == backward:
            # Nothing to choose between the two: keep the published axis and say so.
            notes.append(
                "no void found either way, kept the published axis"
                if forward == 0
                else "both directions look like a void, kept the published axis"
            )
        if bottom != "flat":
            notes.append(f"bottom '{bottom}' modelled flat")
        offset, length = 0.0, depth

    label = f"hole \u00d8{fmt(diameter, 3)} x {fmt(depth, 3)} {bottom}, axis {_axis_name(axis)}"
    shift = "" if abs(offset) < 1e-9 else f"Pos(0, 0, {fmt(offset)}) * "
    code = [
        f"_mouth = Plane(origin={fmt_tuple(location)}, z_dir={fmt_tuple(axis)})",
        (
            f"tool = _mouth * {shift}Cylinder({fmt(diameter / 2)}, {fmt(length)}, "
            f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
        ),
    ]

    for name in ("cbore", "spotface"):
        step = record.get(name)
        if step:
            code.append(
                f"tool += _mouth * Cylinder({fmt(float(step['diameter']) / 2)}, "
                f"{fmt(float(step['depth']))}, align=(Align.CENTER, Align.CENTER, Align.MIN))"
            )
            label += f" + {name} \u00d8{fmt(float(step['diameter']), 3)}"
    csink = record.get("csink")
    if csink:
        code.append(
            f"tool += _mouth * Cone({fmt(float(csink['major_diameter']) / 2)}, "
            f"{fmt(float(csink['drill_diameter']) / 2)}, {fmt(float(csink['depth']))}, "
            f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
        )
        label += f" + csink \u00d8{fmt(float(csink['major_diameter']), 3)}"

    return Op("cut", "holes", feature["index"], label, code, note="; ".join(notes))


def _section_axes(axis: str) -> tuple[int, int]:
    """The two coordinate indices a section is expressed in, ascending XYZ order.

    Quiddity writes a transverse section in the pair left over once the run axis is
    removed: `yz` for a run along x, `xz` for y, `xy` for z.
    """
    return tuple(k for k in range(3) if k != AXIS_INDEX[axis])  # type: ignore[return-value]


def _reach_envelope(span: tuple[float, float], index: int, ctx: Context) -> tuple[float, float]:
    """Run a tool past the envelope wherever its span already ends there.

    A cut that stops exactly on the outside face leaves a sliver behind once the
    tolerances of two kernels are involved.
    """
    low, high = span
    if abs(low - ctx.bb_min[index]) < 1e-3:
        low -= ctx.margin
    if abs(high - ctx.bb_max[index]) < 1e-3:
        high += ctx.margin
    return low, high


def _half_space_code(axis_index: int, cut_at: float, remove_above: bool, ctx: Context,
                     variable: str) -> str:
    """A box covering everything to one side of a plane, for trimming by subtraction.

    Two of these clip a cylinder to a quadrant. Subtracting boxes is steadier than
    intersecting with one: an intersection can hand back a solid with a tangent face
    attached, which build123d then refuses to subtract from the part.
    """
    reach = 4 * ctx.diagonal
    spans = {k: (-reach, reach) for k in range(3)}
    spans[axis_index] = (cut_at, cut_at + reach) if remove_above else (cut_at - reach, cut_at)
    return _box_code(spans, variable)


def _box_code(spans: dict[int, tuple[float, float]], variable: str = "tool") -> str:
    size = [spans[k][1] - spans[k][0] for k in range(3)]
    centre = [(spans[k][0] + spans[k][1]) / 2 for k in range(3)]
    return f"{variable} = Pos{fmt_tuple(centre)} * Box({', '.join(fmt(v) for v in size)})"


def through_step(feature: dict, ctx: Context) -> Op | None:
    """A rectangular corner run clear through the part.

    The record gives both legs and their concave corner as a transverse section, and
    the midpoint of the prism that was removed. The rectangle spanned by the three
    section points is that prism's cross-section, and the midpoint says where along
    the run it sits, so nothing has to be guessed about which corner went away.
    """
    record = feature["record"]
    axis = record["axis"]
    run = AXIS_INDEX[axis]
    first, second = _section_axes(axis)
    points = record["section"]
    length = float(record["length"])
    middle = float(record["at"][run])

    spans = {
        first: (min(p[0] for p in points), max(p[0] for p in points)),
        second: (min(p[1] for p in points), max(p[1] for p in points)),
        run: (middle - length / 2, middle + length / 2),
    }
    for index in (first, second, run):
        spans[index] = _reach_envelope(spans[index], index, ctx)

    label = (
        f"through step {fmt(spans[first][1] - spans[first][0], 3)} x "
        f"{fmt(spans[second][1] - spans[second][0], 3)}, run {axis} {fmt(length, 3)}"
    )
    return Op("cut", "through_steps", feature["index"], label, [_box_code(spans)])


def circular_blind_step(feature: dict, ctx: Context) -> Op | None:
    """A quarter-cylinder taken out of a stock corner, closed at one end.

    Built by trimming a cylinder down to the corner quadrant rather than by drawing a
    pie slice: the record fixes the centre and both arc ends but not the sense of the
    sweep, and trimming cannot get that backwards.
    """
    record = feature["record"]
    axis = record["axis"]
    radius = float(record["radius"])
    points = [tuple(float(v) for v in point) for point in record["section"]]
    first, second = _section_axes(axis)

    centre = None
    for index, candidate in enumerate(points):
        others = [p for k, p in enumerate(points) if k != index]
        if all(abs(math.dist(candidate, other) - radius) < radius * 1e-3 for other in others):
            centre = candidate
            others_of_centre = others
            break
    if centre is None:
        return None  # the section does not describe a corner of this radius

    terminal, opening = (tuple(float(v) for v in end) for end in record["centreline"])
    direction = unit(tuple(opening[k] - terminal[k] for k in range(3)))
    length = math.dist(terminal, opening) + ctx.margin  # the open end runs past the envelope

    label = f"circular blind step R{fmt(radius, 3)}, run {axis} {fmt(record['length'], 3)}"
    code = [
        f"_axis = Plane(origin={fmt_tuple(terminal)}, z_dir={fmt_tuple(direction)})",
        (
            f"tool = _axis * Cylinder({fmt(radius)}, {fmt(length)}, "
            f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
        ),
    ]
    for coordinate, index in enumerate((first, second)):
        far = max(point[coordinate] for point in others_of_centre)
        near = min(point[coordinate] for point in others_of_centre)
        keep_above = abs(far - centre[coordinate]) > abs(near - centre[coordinate])
        code.append(_half_space_code(index, centre[coordinate], not keep_above, ctx, "_beyond"))
        code.append("tool -= _beyond")
    return Op("cut", "circular_blind_steps", feature["index"], label, code)


def _section_frame(axis: str) -> tuple[tuple, tuple, int]:
    """Plane axes for a transverse section, plus the sign its second coordinate needs.

    A build123d plane derives y_dir as z_dir x x_dir, which is not always the
    ascending-order second coordinate: for a run along y it comes out negated.
    """
    run = AXES[axis]
    first, second = _section_axes(axis)
    x_dir = tuple(1.0 if k == first else 0.0 for k in range(3))
    y_dir = cross(run, x_dir)
    return x_dir, run, (1 if y_dir[second] > 0 else -1)


def _section_extrusion(axis: str, run_span: tuple[float, float], points, label_var="_prof"):
    """Source for a closed section profile extruded along a principal run axis."""
    x_dir, run, sign = _section_frame(axis)
    run_index = AXIS_INDEX[axis]
    origin = tuple(run_span[0] if k == run_index else 0.0 for k in range(3))
    flat = [(u, sign * v) for u, v in points]
    segments = [
        f"Line({fmt_tuple(flat[i])}, {fmt_tuple(flat[(i + 1) % len(flat)])})"
        for i in range(len(flat))
    ]
    joined = "\n    + ".join(segments)
    return [
        f"{label_var} = (\n    {joined}\n)",
        (
            f"_plane = Plane(origin={fmt_tuple(origin)}, x_dir={fmt_tuple(x_dir)}, "
            f"z_dir={fmt_tuple(run)})"
        ),
        (
            f"tool = extrude(_plane * make_face({label_var}), "
            f"amount={fmt(run_span[1] - run_span[0])})"
        ),
    ]


def angled_step(feature: dict, ctx: Context) -> Op | None:
    """A wedge taken off a corner by one oblique wall, blind at one end.

    The record publishes the point on the virtual sharp edge, and the vector from it to
    the slant centre points into the wedge, which settles both leg signs and which leg
    runs along which axis. Older records carry only the legs and the slant centre, and
    for those the corner is matched against the envelope and then probed.
    """
    record = feature["record"]
    axis = record["axis"]
    run = AXIS_INDEX[axis]
    first, second = _section_axes(axis)
    legs = (float(record["leg1"]), float(record["leg2"]))
    at = [float(v) for v in record["at"]]
    length = float(record["length"])
    note = ""

    published = _published_wedge(record, at, axis, legs, ctx)
    if published is not None:
        triangle = list(published)
    else:
        best = None
        for leg_a, leg_b in (legs, legs[::-1]):
            for corner_a, way_a in ((ctx.bb_min[first], 1.0), (ctx.bb_max[first], -1.0)):
                for corner_b, way_b in ((ctx.bb_min[second], 1.0), (ctx.bb_max[second], -1.0)):
                    error = max(
                        abs(corner_a + way_a * leg_a / 2 - at[first]),
                        abs(corner_b + way_b * leg_b / 2 - at[second]),
                    )
                    if best is None or error < best[0]:
                        best = (error, corner_a, way_a, leg_a, corner_b, way_b, leg_b)
        error, corner_a, way_a, leg_a, corner_b, way_b, leg_b = best
        if error <= max(0.05 * max(legs), 0.2):
            triangle = [
                (corner_a, corner_b),
                (corner_a + way_a * leg_a, corner_b),
                (corner_a, corner_b + way_b * leg_b),
            ]
            note = "" if error < 1e-6 else f"corner placed to within {fmt(error, 3)} mm"
        else:
            # Not every angled step sits on a corner of the envelope; some are cut into
            # a face an earlier feature already stepped back. Ask the reference instead.
            probed = _corner_wedge(at, legs, axis, ctx)
            if probed is None:
                return None
            triangle = list(probed)

    span = _reach_envelope((at[run] - length / 2, at[run] + length / 2), run, ctx)
    label = (
        f"angled step {fmt(legs[0], 3)} x {fmt(legs[1], 3)} at "
        f"{fmt(float(record['angle']), 1)}\u00b0, run {axis} {fmt(length, 3)}"
    )
    return Op(
        "cut", "angled_steps", feature["index"], label,
        _section_extrusion(axis, span, triangle), note=note,
    )


def _section_frame(axis: str) -> tuple[tuple, tuple, int]:
    """Plane axes for a transverse section, plus the sign its second coordinate needs.

    A build123d plane derives y_dir as z_dir x x_dir, which is not always the
    ascending-order second coordinate: for a run along y it comes out negated.
    """
    run = AXES[axis]
    first, second = _section_axes(axis)
    x_dir = tuple(1.0 if k == first else 0.0 for k in range(3))
    y_dir = cross(run, x_dir)
    return x_dir, run, (1 if y_dir[second] > 0 else -1)


def _section_extrusion(axis: str, run_span: tuple[float, float], points, label_var="_prof"):
    """Source for a closed section profile extruded along a principal run axis."""
    x_dir, run, sign = _section_frame(axis)
    run_index = AXIS_INDEX[axis]
    origin = tuple(run_span[0] if k == run_index else 0.0 for k in range(3))
    flat = [(u, sign * v) for u, v in points]
    segments = [
        f"Line({fmt_tuple(flat[i])}, {fmt_tuple(flat[(i + 1) % len(flat)])})"
        for i in range(len(flat))
    ]
    joined = "\n    + ".join(segments)
    return [
        f"{label_var} = (\n    {joined}\n)",
        (
            f"_plane = Plane(origin={fmt_tuple(origin)}, x_dir={fmt_tuple(x_dir)}, "
            f"z_dir={fmt_tuple(run)})"
        ),
        (
            f"tool = extrude(_plane * make_face({label_var}), "
            f"amount={fmt(run_span[1] - run_span[0])})"
        ),
    ]


def paired_ramp_step(feature: dict, ctx: Context) -> Op | None:
    """A mirror-symmetric V cut into a stock side, blind at one end.

    The record states `half_width`, the distance from the ridge to either ramp's outer
    edge, which fixes the section without deriving it from the half-angle. It also states
    `opening_direction`, which runs along the axis from the blind terminal towards the
    exterior opening and so says which end of the run to carry past the envelope. Which
    way the V itself opens is still not stated, and is settled by the ridge lying between
    material on one side and space on the other.
    """
    record = feature["record"]
    axis = record["axis"]
    run = AXIS_INDEX[axis]
    first, second = _section_axes(axis)
    at = [float(v) for v in record["at"]]
    angle = math.radians(float(record["angle"]))
    length = float(record["length"])
    if not 0.0 < angle < math.pi / 2:
        return None

    opening = None
    step = ctx.diagonal * 0.01
    for index in (first, second):
        for way in (1.0, -1.0):
            outward = [0.0, 0.0, 0.0]
            outward[index] = way
            ahead = tuple(at[k] + outward[k] * step for k in range(3))
            behind = tuple(at[k] - outward[k] * step for k in range(3))
            if not ctx.inside_solid(ahead) and ctx.inside_solid(behind):
                edge = ctx.bb_max[index] if way > 0 else ctx.bb_min[index]
                reach = abs(edge - at[index])
                if opening is None or reach > opening[2]:
                    opening = (index, way, reach)
    if opening is None:
        return None
    height_axis, way, depth = opening

    # `half_width` is measured at the ramps' exterior edge, not at the envelope, and the
    # two are not the same place. Taking it as the width of a tool that runs to the
    # envelope makes the V too narrow all the way down. The tool keeps the section's
    # slope and carries it out to the envelope, which the published width confirms:
    # half_width / tan(angle) is the distance from the ridge to that exterior edge.
    half_width = depth * math.tan(angle)
    lateral = first if height_axis == second else second
    apex_lateral, apex_height = at[lateral], at[height_axis]
    rim = apex_height + way * depth

    by_index = {
        lateral: (apex_lateral, apex_lateral - half_width, apex_lateral + half_width),
        height_axis: (apex_height, rim, rim),
    }
    triangle = [(by_index[first][k], by_index[second][k]) for k in range(3)]

    low, high = at[run] - length / 2, at[run] + length / 2
    towards = record.get("opening_direction")
    if towards is not None and abs(float(towards[run])) > 0.5:
        # Carry the open end clear of the envelope and leave the blind end where it is.
        if float(towards[run]) > 0:
            high += ctx.margin
        else:
            low -= ctx.margin
        span = (low, high)
    else:
        span = _reach_envelope((low, high), run, ctx)

    label = (
        f"paired ramp step {fmt(2 * float(record['angle']), 1)}\u00b0 included, "
        f"depth {fmt(depth, 3)}, run {axis} {fmt(length, 3)}"
    )
    return Op(
        "cut", "paired_ramp_steps", feature["index"], label,
        _section_extrusion(axis, span, triangle),
    )


def _section_point(axis: str, section: tuple[float, float], run_at: float) -> tuple:
    """Lift a transverse section coordinate pair back to a 3D point."""
    first, second = _section_axes(axis)
    point = [0.0, 0.0, 0.0]
    point[first], point[second] = section
    point[AXIS_INDEX[axis]] = run_at
    return tuple(point)


def _void_fraction(axis: str, corners, run_at: float, ctx: Context) -> float:
    """Share of a section triangle that is empty space in the reference."""
    weights = ((1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6))
    empty = 0
    for weight in weights:
        section = tuple(
            sum(weight[k] * corners[k][coordinate] for k in range(3)) for coordinate in (0, 1)
        )
        if not ctx.inside_solid(_section_point(axis, section, run_at)):
            empty += 1
    return empty / len(weights)


def _wedge_fit(axis: str, triangle, run_at: float, ctx: Context) -> float:
    """How much material lies immediately behind a candidate wedge's slant face.

    Emptiness alone does not always identify the wedge. Swapping which leg runs along
    which axis gives a different triangle inside the same pocket of empty space, and
    both read as empty; only the right one has its slant lying on a face of the part.
    This separates that tie, and is not asked to do more: a chamfer next to a pocket
    has empty space behind part of its slant and is still the right wedge.
    """
    corner, first, second = triangle
    midpoint = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
    outward = (midpoint[0] - corner[0], midpoint[1] - corner[1])
    reach = math.hypot(*outward)
    if reach <= 0.0:
        return 0.0
    step = reach * 0.08
    behind = 0
    places = (0.25, 0.5, 0.75)
    for fraction in places:
        point = (
            first[0] + (second[0] - first[0]) * fraction + outward[0] / reach * step,
            first[1] + (second[1] - first[1]) * fraction + outward[1] / reach * step,
        )
        if ctx.inside_solid(_section_point(axis, point, run_at)):
            behind += 1
    return behind / len(places)


def _published_wedge(record: dict, at, axis: str, legs=None, ctx: Context | None = None):
    """The removed wedge, read straight from the record's own corner point.

    Quiddity publishes the point on the virtual sharp edge, and says the vector from it
    to the slant centre points into the wedge. That fixes the leg signs and which leg
    runs along which axis in one step, so there is nothing left to search for. Absent on
    a turned bevel, whose virtual edge is a circle rather than a point.

    The corner is used only when it agrees with the legs the same record publishes and
    the wedge it describes is empty in the reference. On one corpus part the two fields
    disagree outright, and on another they agree while the wedge still lands in solid
    material. A published fact is a strong hint and a cheap one to check, and checking
    it costs four point classifications against the alternative of a confident wrong cut.
    """
    corner = record.get("corner")
    if corner is None:
        return None
    first, second = _section_axes(axis)
    towards = (float(at[first]) - float(corner[first]), float(at[second]) - float(corner[second]))
    if min(abs(towards[0]), abs(towards[1])) < 1e-9:
        return None  # a degenerate wedge has no two legs to speak of
    if legs:
        implied = sorted((abs(2 * towards[0]), abs(2 * towards[1])))
        stated = sorted(float(v) for v in legs)
        if any(
            abs(a - b) > 0.05 * max(b, 1e-9) for a, b in zip(implied, stated, strict=True)
        ):
            return None
    base = (float(corner[first]), float(corner[second]))
    triangle = (
        base,
        (base[0] + 2 * towards[0], base[1]),
        (base[0], base[1] + 2 * towards[1]),
    )
    if ctx is not None and _void_fraction(axis, triangle, at[AXIS_INDEX[axis]], ctx) < 1.0:
        return None
    return triangle


def _corner_wedge(at, legs, axis: str, ctx: Context):
    """Find the wedge behind a slant face centre.

    A chamfer and an angled step are the same triangle: a right angle on a convex
    corner of the material, with the reported point at the centre of the hypotenuse.
    Four sign choices and two leg assignments cover every candidate, and the reference
    picks between them. Probing beats matching against the envelope, because not every
    such corner is a corner of the stock.
    """
    first, second = _section_axes(axis)
    run_at = at[AXIS_INDEX[axis]]
    ranked = []
    for leg_a, leg_b in {legs, legs[::-1]}:
        for way_a in (1.0, -1.0):
            for way_b in (1.0, -1.0):
                corner = (at[first] + way_a * leg_a / 2, at[second] + way_b * leg_b / 2)
                triangle = (
                    corner,
                    (corner[0] - way_a * leg_a, corner[1]),
                    (corner[0], corner[1] - way_b * leg_b),
                )
                ranked.append(
                    (
                        _void_fraction(axis, triangle, run_at, ctx),
                        _wedge_fit(axis, triangle, run_at, ctx),
                        triangle,
                    )
                )
    ranked.sort(key=lambda entry: (-entry[0], -entry[1]))
    best, runner_up = ranked[0], ranked[1]
    if best[0] < 1.0:
        return None  # no candidate is wholly empty
    if runner_up[0] < 1.0:
        return best[2]  # emptiness alone settles it
    if best[1] - runner_up[1] < 0.5:
        return None  # equally empty, and the slant behind them cannot tell them apart
    return best[2]


def _turned_treatment(record: dict, ctx: Context, kind: str) -> Op | None:
    """A chamfer or round swept about an axis: a cone or a ring, not a wedge.

    The reported point lies on the swept face itself, part way out from the axis, so it
    gives the radius once the axis is known but not the axis. The part's own round
    features supply candidate axes, and the diameter the treatment sits on follows from
    the reported radius plus or minus half the cut, depending on whether the material
    is inside the ring or outside it. Every reading is built and the reference keeps
    whichever one is empty.
    """
    axis_name = record["axis"]
    direction = AXES.get(axis_name)
    if direction is None:
        return None
    at = [float(v) for v in record["at"]]
    if kind == "chamfer":
        legs = [(float(record["leg1"]), float(record["leg2"])),
                (float(record["leg2"]), float(record["leg1"]))]
        section = float(record["leg1"]) * float(record["leg2"]) / 2
    else:
        size = float(record["radius"])
        legs = [(size, size)]
        section = size * size * (1 - math.pi / 4)

    best = None
    for axis, anchor in ctx.sweep_axes(direction, at):
        reported = ctx.radius_about(at, axis, anchor)
        if reported <= 1e-6:
            continue
        for depth, across in legs:
            # The reported point sits mid-way along the swept face, so the diameter it
            # belongs to is half a cut either side of it: outside for a bore, inside
            # for a shaft. Which tool shape goes with which is not fixed, so all four
            # pairings are offered and the reference picks.
            for radius in (reported + across / 2, reported - across / 2):
                if radius <= across * 0.51:
                    continue
                # Which way the cone narrows cannot be settled by how much the tool
                # eats: the two orientations have identical volume and both sit in
                # empty space. The part says it directly. Just inside the full
                # diameter, one end of the treatment is still solid and the other has
                # been cut away, and the cone narrows towards the empty one.
                centre_point = ctx.on_axis(at, axis, anchor)
                probe = ctx._perpendicular(axis)
                ends = []
                for side in (1.0, -1.0):
                    spot = tuple(
                        centre_point[k]
                        + axis[k] * side * depth / 2
                        + probe[k] * (radius - across * 0.1)
                        for k in range(3)
                    )
                    ends.append((side, ctx.inside_solid(spot)))
                solid_ends = [side for side, filled in ends if filled]
                empty_ends = [side for side, filled in ends if not filled]
                if len(solid_ends) == 1 and len(empty_ends) == 1:
                    # The mouth belongs at the empty end so the cone grows back into
                    # the solid. `mouth` is half a depth back along the push, so the
                    # push runs away from the empty end, not towards it.
                    ways = [-empty_ends[0]]
                else:
                    ways = [1.0, -1.0]  # nothing to choose between them; try both
                for outward, way in [(sense, w) for sense in (1.0, -1.0) for w in ways]:
                    push = tuple(axis[k] * way for k in range(3))
                    centre = ctx.on_axis(at, axis, anchor)
                    mouth = tuple(centre[k] - push[k] * depth / 2 for k in range(3))
                    plane = f"Plane(origin={fmt_tuple(mouth)}, z_dir={fmt_tuple(push)})"
                    inner = max(radius - across, 0.001)
                    if outward > 0:
                        # Material outside the ring: the cone opening out of a bore.
                        code = [
                            f"_swept = {plane}",
                            (
                                f"tool = _swept * Cone({fmt(radius)}, {fmt(inner)}, "
                                f"{fmt(depth)}, align=(Align.CENTER, Align.CENTER, Align.MIN))"
                            ),
                        ]
                    else:
                        # Material inside: the cut is the ring the cone leaves behind.
                        code = [
                            f"_swept = {plane}",
                            (
                                f"tool = _swept * Cylinder({fmt(radius)}, {fmt(depth)}, "
                                f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
                            ),
                            (
                                f"tool -= _swept * Cone({fmt(inner)}, {fmt(radius)}, "
                                f"{fmt(depth)}, align=(Align.CENTER, Align.CENTER, Align.MIN))"
                            ),
                        ]
                    try:
                        tool = run_source(code)
                        eaten = common_volume(tool, ctx.part)
                    except Exception:  # noqa: BLE001 - a reading that will not build
                        continue
                    if tool.volume <= 1e-9 or eaten > 0.05 * tool.volume:
                        continue
                    # How big the ring has to be is not in doubt: a section of known
                    # area swept once around gives its volume. Ranking by emptiness
                    # instead lets a degenerate reading win, because a solid cone that
                    # swallows a bored-out core also removes nothing it should not.
                    wanted = 2 * math.pi * reported * section
                    error = abs(tool.volume - wanted) / max(wanted, 1e-9)
                    if best is None or error < best[0]:
                        best = (error, code, radius)
    if best is None or best[0] > 0.35:
        return None  # nothing the right size for the legs this record reports

    _error, code, radius = best
    what = "chamfer" if kind == "chamfer" else f"round R{fmt(legs[0][0], 3)}"
    label = f"turned {what} at \u00d8{fmt(2 * radius, 3)}, swept about {axis_name}"
    family = "chamfers" if kind == "chamfer" else "fillets"
    return Op("cut", family, -1, label, code)


def chamfer(feature: dict, ctx: Context) -> Op | None:
    """A chamfer, cut as the wedge it removes and run the length of its edge.

    Modelled as a subtraction rather than build123d's `chamfer()` because selecting
    the right edge needs the earlier operations to have reproduced it exactly, and a
    wedge needs only the record.
    """
    record = feature["record"]
    if record.get("turned"):
        swept = _turned_treatment(record, ctx, "chamfer")
        if swept is not None:
            swept.index = feature["index"]
        return swept
    axis = record["axis"]
    at = [float(v) for v in record["at"]]
    legs = (float(record["leg1"]), float(record["leg2"]))
    triangle = _published_wedge(record, at, axis, legs, ctx) or _corner_wedge(
        at, legs, axis, ctx
    )
    if triangle is None:
        return None

    run = AXIS_INDEX[axis]
    centroid = tuple(sum(point[k] for point in triangle) / 3 for k in range(2))
    span = _run_extent(axis, centroid, at[run], ctx)
    if span is None:
        return None
    label = (
        f"chamfer {fmt(legs[0], 3)} x {fmt(legs[1], 3)} at "
        f"{fmt(float(record['angle']), 1)}\u00b0, along {axis}"
    )
    return Op(
        "cut", "chamfers", feature["index"], label,
        _section_extrusion(axis, span, list(triangle)),
    )


def _round_evidence(axis, corner, centre, radius, run_at, ctx: Context):
    """(share of the sliver that is empty, share of the disc that is empty).

    Both halves are needed. A sliver reading empty only says material is absent there,
    which is equally true when the whole square sits outside the part; the disc has to
    come back solid for there to be a rounded edge at all.
    """
    grid = (0.1, 0.3, 0.5, 0.7, 0.9)
    counts = {"sliver": [0, 0], "disc": [0, 0]}
    for a in grid:
        for b in grid:
            point = (
                corner[0] + (centre[0] - corner[0]) * a,
                corner[1] + (centre[1] - corner[1]) * b,
            )
            distance = math.dist(point, centre)
            if radius * 0.98 < distance < radius * 1.02:
                continue  # on the arc itself, where the answer is not meaningful
            where = "disc" if distance <= radius else "sliver"
            counts[where][1] += 1
            if not ctx.inside_solid(_section_point(axis, point, run_at)):
                counts[where][0] += 1
    return tuple(
        counts[where][0] / counts[where][1] if counts[where][1] else 0.0
        for where in ("sliver", "disc")
    )


def _run_extent(axis: str, section: tuple[float, float], start: float, ctx: Context):
    """How far a constant-section cut runs, found by walking a point along it.

    A blend record names an edge but not its length, and running the tool the whole
    way through the part removes material wherever that edge stopped short. Walking a
    point that sits inside the cut until it meets material finds the real extent.
    """
    index = AXIS_INDEX[axis]
    if ctx.inside_solid(_section_point(axis, section, start)):
        return None  # the section does not describe empty space even where it should
    step = ctx.diagonal / 200
    limits = []
    for direction in (1.0, -1.0):
        position = last = start
        while ctx.bb_min[index] - step <= position <= ctx.bb_max[index] + step:
            position += direction * step
            if ctx.inside_solid(_section_point(axis, section, position)):
                break
            last = position
        limits.append(last)
    low, high = min(limits), max(limits)
    if low <= ctx.bb_min[index] + step:
        low -= ctx.margin
    if high >= ctx.bb_max[index] - step:
        high += ctx.margin
    return low, high


def fillet(feature: dict, ctx: Context) -> Op | None:
    """A rounded edge, cut in whichever of its two senses the reference shows.

    A radius and the centre of the round's face describe two quite different cuts, and
    which one it is cannot be read off the record. Either the arc is the outside of the
    material and what went is the sliver between it and the sharp corner, or the arc is
    the inside of a cove and what went is the quarter cylinder itself. The reference
    tells them apart: exactly one of the sliver and the disc is empty, and which one it
    is names the cut. A record where neither stands out is left unmodelled.
    """
    record = feature["record"]
    if record.get("turned"):
        swept = _turned_treatment(record, ctx, "round")
        if swept is not None:
            swept.index = feature["index"]
        return swept
    axis = record["axis"]
    radius = float(record["radius"])
    at = [float(v) for v in record["at"]]
    first, second = _section_axes(axis)
    run = AXIS_INDEX[axis]
    offset = radius * (1 - 1 / math.sqrt(2))  # the face centre stands this far off the corner

    ranked = []
    for way_a in (1.0, -1.0):
        for way_b in (1.0, -1.0):
            corner = (at[first] + way_a * offset, at[second] + way_b * offset)
            centre = (corner[0] - way_a * radius, corner[1] - way_b * radius)
            sliver, disc = _round_evidence(axis, corner, centre, radius, at[run], ctx)
            ranked.append((abs(sliver - disc), sliver - disc, corner, centre))
    ranked.sort(key=lambda entry: -entry[0])
    if ranked[0][0] < 0.9 or ranked[1][0] > 0.75:
        return None

    _, sense, corner, centre = ranked[0]
    toward = 0.4 if sense < 0 else 0.9  # a point inside the disc, or inside the sliver
    inside = tuple(
        centre[k] + (corner[k] - centre[k]) * toward for k in range(2)
    )
    span = _run_extent(axis, inside, at[run], ctx)
    if span is None:
        return None
    spans = {
        first: tuple(sorted((corner[0], centre[0]))),
        second: tuple(sorted((corner[1], centre[1]))),
        run: span,
    }
    axis_origin = _section_point(axis, centre, span[0])
    cylinder = (
        f"_arc * Cylinder({fmt(radius)}, {fmt(span[1] - span[0])}, "
        f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
    )
    plane = f"_arc = Plane(origin={fmt_tuple(axis_origin)}, z_dir={fmt_tuple(AXES[axis])})"

    if sense > 0:
        # The arc is the outside of the material: the sliver behind it went away.
        label = f"rounded edge R{fmt(radius, 3)}, along {axis}"
        code = [_box_code(spans, "_square"), plane, f"tool = _square - {cylinder}"]
    else:
        # The arc is the inside of a cove: the quarter cylinder itself went away.
        label = f"rounded corner cut R{fmt(radius, 3)}, along {axis}"
        code = [plane, f"tool = {cylinder}"]
        for coordinate, index in enumerate((first, second)):
            code.append(
                _half_space_code(
                    index, centre[coordinate], corner[coordinate] < centre[coordinate],
                    ctx, "_beyond",
                )
            )
            code.append("tool -= _beyond")
    return Op("cut", "fillets", feature["index"], label, code)


def blend(feature: dict, ctx: Context) -> Op | None:
    """A rolling-ball round, which states its own sense and its own axis.

    Where a Fillet record has to be read back from a face centre, a Blend gives the
    rolling path outright, so the cylinder axis needs no reconstruction, and `side`
    says whether the round is external or internal instead of leaving it to be probed.
    An external round removes the sliver behind it. An internal one is material a
    neighbouring cut takes away and has to put back.
    """
    record = feature["record"]
    path = record["path"]
    if "direction" not in path:
        return None  # a circular rolling path sweeps a torus, not a prism
    direction = unit(path["direction"])
    axis = _axis_name(direction)
    if axis == "oblique":
        return None

    radius = float(record["radius"])
    external = record["side"] == "convex"
    first, second = _section_axes(axis)
    run = AXIS_INDEX[axis]
    anchor = [float(v) for v in path["at"]]
    centre = (anchor[first], anchor[second])

    ranked = []
    for way_a in (1.0, -1.0):
        for way_b in (1.0, -1.0):
            corner = (centre[0] + way_a * radius, centre[1] + way_b * radius)
            sliver, disc = _round_evidence(axis, corner, centre, radius, anchor[run], ctx)
            # An external round leaves the sliver empty and the disc solid; an internal
            # one leaves the disc empty and the sliver solid. The declared side says
            # which to look for, so only the corner is still in question.
            score = (sliver - disc) if external else (disc - sliver)
            ranked.append((score, corner))
    ranked.sort(key=lambda entry: -entry[0])
    if ranked[0][0] < 0.9 or ranked[1][0] > 0.75:
        return None

    corner = ranked[0][1]
    toward = 0.9 if external else 0.4  # a point inside the sliver, or inside the disc
    inside = tuple(centre[k] + (corner[k] - centre[k]) * toward for k in range(2))
    span = _run_extent(axis, inside, anchor[run], ctx)
    if span is None:
        return None

    spans = {
        first: tuple(sorted((corner[0], centre[0]))),
        second: tuple(sorted((corner[1], centre[1]))),
        run: span,
    }
    axis_origin = _section_point(axis, centre, span[0])
    sense = "external" if external else "internal"
    label = f"{sense} blend R{fmt(radius, 3)}, along {axis}"
    code = [
        _box_code(spans, "_square"),
        f"_arc = Plane(origin={fmt_tuple(axis_origin)}, z_dir={fmt_tuple(AXES[axis])})",
        (
            f"tool = _square - _arc * Cylinder({fmt(radius)}, {fmt(span[1] - span[0])}, "
            f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
        ),
    ]
    return Op("cut" if external else "fuse", "blends", feature["index"], label, code)


def step_level(feature: dict, ctx: Context) -> Op | None:
    """A horizontal face, proposed as the cut that exposed it.

    A face level is evidence, not a feature: it says a flat face sits at this height
    over this patch, and says nothing about what was removed to leave it there. The
    obvious reading is that everything past it over that patch is gone, which is right
    for a plain step and wrong wherever something stands on the face. So the op is
    marked speculative, and is kept only if it proves it removes empty space alone.
    """
    record = feature["record"]
    x_span, y_span = record.get("x_span"), record.get("y_span")
    if not x_span or not y_span:
        return None  # a level with no measured patch describes no region
    level = float(record["z"])
    spans = {
        0: (float(x_span[0]), float(x_span[1])),
        1: (float(y_span[0]), float(y_span[1])),
    }
    middle = tuple((spans[k][0] + spans[k][1]) / 2 for k in range(2))
    step = ctx.diagonal * 0.005
    above = ctx.inside_solid((middle[0], middle[1], level + step))
    below = ctx.inside_solid((middle[0], middle[1], level - step))
    if above == below:
        return None  # not a face between material and space, so not a step to cut

    spans[2] = (
        (level, ctx.bb_max[2] + ctx.margin) if below else (ctx.bb_min[2] - ctx.margin, level)
    )
    side = "above" if below else "below"
    label = (
        f"clear {side} the face at z {fmt(level, 3)}, "
        f"{fmt(spans[0][1] - spans[0][0], 3)} x {fmt(spans[1][1] - spans[1][0], 3)}"
    )
    return Op(
        "cut", "step_levels", feature["index"], label, [_box_code(spans)], speculative=True
    )


#: Height of a quarter circle's arc above its chord, per unit radius: 1 - cos 45°.
QUARTER_SAGITTA = 1 - math.sqrt(0.5)


def slot(feature: dict, ctx: Context) -> Op | None:
    """A slot, with its rounded ends or rounded corners when it has them.

    Cutting a rounded slot square was the single reason slots ate material on real
    parts: the square corners sit outside the true slot, and every slot that overlapped
    had a published end radius the adapter was throwing away.
    """
    record = feature["record"]
    width_axis, long_axis = record["width_axis"], record["long_axis"]
    depth_axis = next(a for a in "xyz" if a not in (width_axis, long_axis))
    width, centre = float(record["width"]), float(record["w_center"])
    low, high = float(record["lo"]), float(record["hi"])
    depth_index = AXIS_INDEX[depth_axis]
    spans = {
        AXIS_INDEX[width_axis]: (centre - width / 2, centre + width / 2),
        AXIS_INDEX[long_axis]: (low, high),
        depth_index: (float(record["d_lo"]), float(record["d_hi"])),
    }
    # A slot open to a face reaches the envelope there; run the tool past it.
    spans[depth_index] = _reach_envelope(spans[depth_index], depth_index, ctx)

    end_radius = record.get("end_radius")
    corner_radius = record.get("corner_radius")
    label = (
        f"slot {fmt(width, 3)} wide x {fmt(float(record['length']), 3)} long, "
        f"depth along {depth_axis}"
    )
    if end_radius is None and corner_radius is None:
        return Op("cut", "slots", feature["index"], label, [_box_code(spans)])

    # Draw the section in the plane the slot runs in, then extrude it through the depth.
    radius = float(end_radius if end_radius is not None else corner_radius)
    half = width / 2
    lo_w, hi_w = centre - half, centre + half
    if end_radius is not None:
        radius = min(radius, half)
        outline = [
            ("line", (low + radius, lo_w), (high - radius, lo_w)),
            ("arc", (high - radius, lo_w), (high - radius, hi_w), -radius),
            ("line", (high - radius, hi_w), (low + radius, hi_w)),
            ("arc", (low + radius, hi_w), (low + radius, lo_w), -radius),
        ]
        label += f", {fmt(radius, 3)} radius ends"
    else:
        radius = min(radius, half, (high - low) / 2)
        outline = [
            ("line", (low + radius, lo_w), (high - radius, lo_w)),
            ("arc", (high - radius, lo_w), (high, lo_w + radius), -radius * QUARTER_SAGITTA),
            ("line", (high, lo_w + radius), (high, hi_w - radius)),
            ("arc", (high, hi_w - radius), (high - radius, hi_w), -radius * QUARTER_SAGITTA),
            ("line", (high - radius, hi_w), (low + radius, hi_w)),
            ("arc", (low + radius, hi_w), (low, hi_w - radius), -radius * QUARTER_SAGITTA),
            ("line", (low, hi_w - radius), (low, lo_w + radius)),
            ("arc", (low, lo_w + radius), (low + radius, lo_w), -radius * QUARTER_SAGITTA),
        ]
        label += f", {fmt(radius, 3)} radius corners"

    # The section lives in (long, width); place it on the plane the depth axis normals.
    order = ("xyz".index(long_axis), "xyz".index(width_axis))
    swap = order[0] > order[1]
    # The section plane's own second axis can point against the world axis it stands
    # for; the shared section helper corrects for that with this sign, and a profile
    # drawn here has to as well or the slot comes out mirrored across the part.
    _x_dir, _run, sign = _section_frame(depth_axis)
    drawn = []
    for piece in outline:
        start, finish = piece[1], piece[2]
        if swap:
            start, finish = (start[1], start[0]), (finish[1], finish[0])
        start, finish = (start[0], sign * start[1]), (finish[0], sign * finish[1])
        if piece[0] == "line":
            drawn.append(f"Line({fmt_tuple(start)}, {fmt_tuple(finish)})")
        else:
            # Each reflection, the swap and the sign, turns an arc to the other side.
            sagitta = (-piece[3] if swap else piece[3]) * sign
            drawn.append(f"SagittaArc({fmt_tuple(start)}, {fmt_tuple(finish)}, {fmt(sagitta)})")
    joined = "\n    + ".join(drawn)
    code = [f"_prof = (\n    {joined}\n)"] + _section_extrusion(
        depth_axis, spans[depth_index], [], label_var="_prof"
    )[1:]
    return Op("cut", "slots", feature["index"], label, code)


# ── additive ─────────────────────────────────────────────────────────


def boss(feature: dict, ctx: Context) -> Op | None:
    record = feature["record"]
    axis = unit(record["axis"])
    location = tuple(float(v) for v in record["location"])
    diameter, height = float(record["diameter"]), float(record["height"])
    notes = []
    forward = ctx.material_score(location, axis, height, diameter / 2)
    backward = ctx.material_score(location, _negate(axis), height, diameter / 2)
    if backward > forward:
        axis = _negate(axis)
    elif forward == backward:
        notes.append("material found on neither side, kept the published axis")
    label = f"boss Ø{fmt(diameter, 3)} x {fmt(height, 3)}, axis {_axis_name(axis)}"
    code = [
        f"_plane = Plane(origin={fmt_tuple(location)}, z_dir={fmt_tuple(axis)})",
        (
            f"tool = _plane * Cylinder({fmt(diameter / 2)}, {fmt(height)}, "
            f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
        ),
    ]
    return Op("fuse", "bosses", feature["index"], label, code, note="; ".join(notes))


def pad(feature: dict, ctx: Context) -> Op | None:
    record = feature["record"]
    low = tuple(float(record[k]) for k in ("x0", "y0", "z0"))
    high = tuple(float(record[k]) for k in ("x1", "y1", "z1"))
    size = tuple(high[k] - low[k] for k in range(3))
    centre = tuple((high[k] + low[k]) / 2 for k in range(3))
    label = f"raised pad {' x '.join(fmt(s, 3) for s in size)}, axis {record.get('axis', '?')}"
    code = [f"tool = Pos{fmt_tuple(centre)} * Box({', '.join(fmt(s) for s in size)})"]
    return Op("fuse", "pads", feature["index"], label, code)


# ── trimming back to faces no feature claimed ────────────────────────


def _edge_kind(edge) -> str:
    return str(edge.geom_type).rsplit(".", 1)[-1]


def _ordered_edges(face):
    """The outer wire's edges, chained end to end, or None if they do not form a loop.

    A wire that came out of a fuse is not always sewn to the last decimal: two edges
    that meet on the drawing can be a few microns apart on the face, and a chain that
    insists on an exact meeting throws the whole outline away over it. So the next
    edge is the nearest one within a tolerance set by the size of the face, and its
    start is snapped to the previous edge's end, which closes the gap in the profile
    rather than passing it on to a make_face that will refuse it.
    """
    edges = list(face.outer_wire().edges())
    if not edges:
        return None
    box = face.bounding_box()
    span = math.dist(box.min.to_tuple(), box.max.to_tuple())
    tolerance = max(1e-6, span * 1e-4)
    edges = [edge for edge in edges if edge.length > tolerance] or edges

    chain = [(edges[0], tuple(edges[0] @ 0.0), tuple(edges[0] @ 1.0))]
    remaining = edges[1:]
    while remaining:
        tail = chain[-1][2]
        best = None
        for position, edge in enumerate(remaining):
            start, end = tuple(edge @ 0.0), tuple(edge @ 1.0)
            for gap, far in ((math.dist(start, tail), end), (math.dist(end, tail), start)):
                if gap < tolerance and (best is None or gap < best[0]):
                    best = (gap, position, edge, far)
        if best is None:
            return None
        _gap, position, edge, far = best
        chain.append((edge, tail, far))
        remaining.pop(position)
    return chain


def _chord_error(edge, first: float, second: float) -> float:
    """How far the curve strays from the chord between two points along an edge."""
    start, end = tuple(edge @ first), tuple(edge @ second)
    middle = tuple(edge @ ((first + second) / 2))
    span = [end[k] - start[k] for k in range(3)]
    length = math.dist(start, end)
    if length < 1e-12:
        return math.dist(middle, start)
    offset = [middle[k] - start[k] for k in range(3)]
    along = sum(offset[k] * span[k] for k in range(3)) / (length * length)
    foot = [start[k] + along * span[k] for k in range(3)]
    return math.dist(middle, foot)


def _outline_segments(face, project):
    """The outer wire as 2D segments a build123d profile can be written from.

    Straight edges stay straight and circles become one arc, so the emitted profile
    reads like the shape it is. Anything else, an ellipse or a spline, is sampled into
    a chain of short lines: the tool it builds only has to remove the right space, and
    an approximation that removes too much is caught before it is kept.
    """
    chain = _ordered_edges(face)
    if chain is None:
        return None
    box = face.bounding_box()
    tolerance = max(1e-4, math.dist(box.min.to_tuple(), box.max.to_tuple()) * 1e-3)
    segments, approximated = [], False
    for edge, start, end in chain:
        kind = _edge_kind(edge)
        flat_start, flat_end = project(start), project(end)
        if kind == "LINE":
            segments.append(("line", flat_start, flat_end))
            continue
        if kind == "CIRCLE" and math.dist(start, end) > 1e-6:
            middle = project(tuple(edge @ 0.5))
            chord = (flat_end[0] - flat_start[0], flat_end[1] - flat_start[1])
            length = math.hypot(*chord)
            if length > 1e-9:
                # Signed height of the arc over its chord, left of the chord positive,
                # which is exactly what build123d's SagittaArc takes.
                sagitta = (
                    chord[0] * (middle[1] - flat_start[1])
                    - chord[1] * (middle[0] - flat_start[0])
                ) / length
                segments.append(("arc", flat_start, flat_end, sagitta))
                continue
        # Sample as finely as the curve needs and no finer. A fixed count writes the
        # same wall of coordinates for an edge that is almost straight as for one that
        # doubles back, and on a silhouette outline drawn from a dozen spline edges
        # that is the difference between a script that runs and one that does not.
        count = 4
        while count < 24:
            worst = max(
                _chord_error(edge, index / count, (index + 1) / count)
                for index in range(count)
            )
            if worst <= tolerance:
                break
            count *= 2
        # Sample along the edge the way the chain walks it, not the way it was built:
        # chaining reverses an edge whenever that is how it meets its neighbour, and
        # sampling the other way round would draw the outline back on itself. The two
        # ends are then taken from the chain so the samples join what came before.
        backwards = math.dist(end, tuple(edge @ 1.0)) > math.dist(end, tuple(edge @ 0.0))
        points = []
        for index in range(count + 1):
            fraction = index / count
            points.append(project(tuple(edge @ (1.0 - fraction if backwards else fraction))))
        points[0], points[-1] = flat_start, flat_end
        for first, second in itertools.pairwise(points):
            if math.dist(first, second) > 1e-9:
                segments.append(("line", first, second))
        approximated = True
    segments = [seg for seg in segments if math.dist(seg[1], seg[2]) > 1e-9]
    if len(segments) < 3:
        return None
    # Close the loop on the drawing. Chaining snapped every join to the edge before
    # it, but the last edge still ends wherever the wire ended, and build123d will
    # not make a face from a profile that does not come back to where it started.
    last = segments[-1]
    segments[-1] = (last[0], last[1], segments[0][1], *last[3:])
    if math.dist(segments[-1][1], segments[-1][2]) <= 1e-9:
        segments.pop()
    return (segments, approximated) if len(segments) >= 3 else None


#: A trim may clip this fraction of its own volume out of material the part keeps
#: before it counts as running into something.
_TRIM_TOLERANCE = 0.02

#: How many unclaimed faces are worth trying on one part, largest first. This covers all
#: but the most involved parts outright, and beyond it the returns are small while each
#: candidate still costs several booleans to place.
TRIM_BUDGET = 150


def _deepest_empty(build, reach: float, ctx: Context) -> float | None:
    """How far in front of a face the space is empty, by halving the distance.

    Running a trim all the way to the envelope fails on any part that folds back on
    itself, because the prism eventually meets material again. Stopping where the
    emptiness stops turns most of those refusals into shorter cuts that are still true.
    """

    def clear(depth: float) -> bool:
        try:
            tool = run_source(build(depth))
        except Exception:  # noqa: BLE001 - an unbuildable depth is simply not usable
            return False
        volume = float(tool.volume)
        return volume > 0.0 and common_volume(tool, ctx.part) / volume <= _TRIM_TOLERANCE

    if clear(reach):
        return reach
    low, high, best = 0.0, reach, None
    for _ in range(4):
        middle = (low + high) / 2
        if clear(middle):
            best, low = middle, middle
        else:
            high = middle
    return best


def _bore_trim(face, ctx: Context, index: int) -> Op | None:
    """An unrecognised cylindrical face with space inside it is a bore nobody named.

    A plain prism cannot express one, so it would otherwise be passed over. The space in
    front of a bore wall is the bore, and cutting it needs only the surface's own axis
    and radius. A cylinder with material on the outside is a boss, where the space in
    front is an annulus and nothing simple to cut, so it is left alone.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    surface = BRepAdaptor_Surface(face.wrapped)
    cylinder = surface.Cylinder()
    radius = float(cylinder.Radius())
    axis_position = cylinder.Axis().Location()
    axis_direction = cylinder.Axis().Direction()
    anchor = (axis_position.X(), axis_position.Y(), axis_position.Z())
    along = unit((axis_direction.X(), axis_direction.Y(), axis_direction.Z()))

    # Material outside and space inside is a bore; the other way round is a boss.
    centre = face.center()
    towards = unit(
        tuple(anchor[k] - (centre.X, centre.Y, centre.Z)[k] for k in range(3))
    )
    inside = tuple((centre.X, centre.Y, centre.Z)[k] + towards[k] * radius * 0.5 for k in range(3))
    if ctx.inside_solid(inside):
        return None

    # Cut only the space the wall actually faces. A curved wall is often a slice of a
    # large cylinder rather than a whole bore, and cutting the whole cylinder for it
    # runs straight through the part on the far side of the axis. The face's own
    # parameter range gives the arc it spans and the length it runs along the axis,
    # so the tool is the pie slice between that arc and the axis, and nothing more.
    from OCP.BRepTools import BRepTools

    u_low, u_high, v_low, v_high = BRepTools.UVBounds_s(face.wrapped)
    position = cylinder.Position()
    x_axis = position.XDirection()
    across = (x_axis.X(), x_axis.Y(), x_axis.Z())
    base = tuple(anchor[k] + along[k] * (v_low - 1e-3) for k in range(3))
    length = (v_high - v_low) + 2e-3
    label = f"clear an unrecognised bore \u00d8{fmt(2 * radius, 3)}"

    if u_high - u_low >= 2 * math.pi - 1e-6:
        code = [
            (
                f"_bore = Plane(origin={fmt_tuple(base)}, x_dir={fmt_tuple(across)}, "
                f"z_dir={fmt_tuple(along)})"
            ),
            (
                f"tool = _bore * Cylinder({fmt(radius)}, {fmt(length)}, "
                f"align=(Align.CENTER, Align.CENTER, Align.MIN))"
            ),
        ]
    else:

        def rim(angle):
            return (radius * math.cos(angle), radius * math.sin(angle))

        first, middle, last = rim(u_low), rim((u_low + u_high) / 2), rim(u_high)
        label += f", {fmt(math.degrees(u_high - u_low), 3)}\u00b0 of it"
        code = [
            (
                f"_bore = Plane(origin={fmt_tuple(base)}, x_dir={fmt_tuple(across)}, "
                f"z_dir={fmt_tuple(along)})"
            ),
            (
                f"_prof = Line((0, 0), {fmt_tuple(first)}) + ThreePointArc("
                f"{fmt_tuple(first)}, {fmt_tuple(middle)}, {fmt_tuple(last)}) + "
                f"Line({fmt_tuple(last)}, (0, 0))"
            ),
            f"tool = extrude(_bore * make_face(_prof), amount={fmt(length)})",
        ]
    return Op("cut", "unclaimed_faces", index, label, code, speculative=True)


def face_profile_source(face, outward, variable: str = "_prof") -> tuple[list[str], tuple] | None:
    """Source for a planar face's outer boundary, drawn in its own plane.

    Shared by stock and by the trims, because both need the same thing: a flat face
    written out as a profile that build123d can make a face from and extrude.
    """
    chain = _ordered_edges(face)
    if chain is None:
        return None
    centre = face.center()
    origin = (centre.X, centre.Y, centre.Z)
    first = unit(tuple(chain[0][2][k] - chain[0][1][k] for k in range(3)))
    second = cross(outward, first)

    def project(point, _origin=origin, _first=first, _second=second):
        return (
            sum((point[k] - _origin[k]) * _first[k] for k in range(3)),
            sum((point[k] - _origin[k]) * _second[k] for k in range(3)),
        )

    outline = _outline_segments(face, project)
    if outline is None:
        return None
    segments, _approximated = outline
    drawn = []
    for segment in segments:
        if segment[0] == "line":
            drawn.append(f"Line({fmt_tuple(segment[1])}, {fmt_tuple(segment[2])})")
        else:
            drawn.append(
                f"SagittaArc({fmt_tuple(segment[1])}, {fmt_tuple(segment[2])}, "
                f"{fmt(segment[3])})"
            )
    joined = "\n    + ".join(drawn)
    return (
        [
            f"{variable} = (\n    {joined}\n)",
            (
                f"_plane = Plane(origin={fmt_tuple(origin)}, x_dir={fmt_tuple(first)}, "
                f"z_dir={fmt_tuple(outward)})"
            ),
        ],
        origin,
    )


def propose_face_trims(
    document: dict, ctx: Context, orphaned: set[int] | None = None
) -> tuple[list[Op], int]:
    """Cut the stock back to the flat faces that no recognised feature accounts for.

    This is a different kind of evidence from the rest of the file. Everything else
    rebuilds a feature the recogniser named; this reads the faces it could not name and
    clears the space in front of them. It is what closes the gap on a part whose bulk
    outer form is published as nothing at all, and it is speculative for the same
    reason: a face says where material ends, not what was removed to end it there. Each
    trim is kept only if it proves it takes away empty space alone.
    """
    faces = ctx.part.faces()
    unclaimed = list((document.get("association") or {}).get("unassociated_faces") or [])
    # A record an adapter refused leaves its faces in limbo: the feature path did not
    # model them and the association calls them claimed, so tracing never sees them
    # either. Hand them to the trimmer, which is the only thing left that can clear
    # the space they bound.
    if orphaned:
        unclaimed.extend(sorted(set(orphaned) - set(unclaimed)))
    # Biggest faces first, and only so many: a real part can leave hundreds of faces
    # unclaimed, each costing several booleans to place, and the large ones are where
    # the material is. The rest are reported rather than chased.
    ordered = sorted(
        (index for index in unclaimed if index < len(faces)),
        key=lambda index: -faces[index].area,
    )
    proposed, passed_over = [], max(len(ordered) - TRIM_BUDGET, 0)
    for index in ordered[:TRIM_BUDGET]:
        face = faces[index]
        centre = face.center()
        origin = (centre.X, centre.Y, centre.Z)
        normal = face.normal_at(centre)
        outward = unit((normal.X, normal.Y, normal.Z))
        step = ctx.diagonal * 0.002
        if ctx.inside_solid(tuple(origin[k] + outward[k] * step for k in range(3))):
            outward = _negate(outward)  # the face was oriented into the material

        reach = max(dot(corner, outward) for corner in ctx.corners()) - dot(origin, outward)
        if reach <= step:
            # The face is already on the envelope. There is nothing in front of it to
            # clear, so this is not a face the tool failed to use.
            continue

        kind = str(face.geom_type).rsplit(".", 1)[-1]
        if kind == "CYLINDER":
            bore = _bore_trim(face, ctx, index)
            if bore is None:
                passed_over += 1
            else:
                proposed.append(bore)
            continue
        if kind != "PLANE":
            passed_over += 1  # room in front, but this tool only writes flat profiles
            continue

        chain = _ordered_edges(face)
        if chain is None:
            passed_over += 1
            continue
        first = unit(tuple(chain[0][2][k] - chain[0][1][k] for k in range(3)))
        second = cross(outward, first)

        def project(point, _origin=origin, _first=first, _second=second):
            return (
                sum((point[k] - _origin[k]) * _first[k] for k in range(3)),
                sum((point[k] - _origin[k]) * _second[k] for k in range(3)),
            )

        outline = _outline_segments(face, project)
        if outline is None:
            passed_over += 1
            continue
        segments, approximated = outline

        drawn = []
        for segment in segments:
            if segment[0] == "line":
                drawn.append(f"Line({fmt_tuple(segment[1])}, {fmt_tuple(segment[2])})")
            else:
                drawn.append(
                    f"SagittaArc({fmt_tuple(segment[1])}, {fmt_tuple(segment[2])}, "
                    f"{fmt(segment[3])})"
                )
        joined = "\n    + ".join(drawn)
        def build(depth, _joined=joined, _origin=origin, _first=first, _outward=outward):
            return [
                f"_prof = (\n    {_joined}\n)",
                (
                    f"_face = Plane(origin={fmt_tuple(_origin)}, x_dir={fmt_tuple(_first)}, "
                    f"z_dir={fmt_tuple(_outward)})"
                ),
                f"tool = extrude(_face * make_face(_prof), amount={fmt(depth)})",
            ]

        depth = _deepest_empty(build, reach + ctx.margin, ctx)
        if depth is None:
            passed_over += 1  # nothing in front of this face is provably empty
            continue
        code = build(depth)
        shape = "curve approximated by lines, " if approximated else ""
        label = (
            f"clear the space in front of an unrecognised flat face, "
            f"{shape}{len(segments)} sides"
        )
        proposed.append(
            Op("cut", "unclaimed_faces", index, label, code, speculative=True)
        )
    return proposed, passed_over


#: Families this prototype models. Everything else is reported as skipped.
ADAPTERS = {
    "section_recesses": section_recess,
    "holes": hole,
    "slots": slot,
    "through_steps": through_step,
    "circular_blind_steps": circular_blind_step,
    "angled_steps": angled_step,
    "paired_ramp_steps": paired_ramp_step,
    "chamfers": chamfer,
    "fillets": fillet,
    "blends": blend,
    "step_levels": step_level,
    "bosses": boss,
    "pads": pad,
}

#: Families that are evidence for other stages rather than operations of their own.
EVIDENCE_ONLY = {
    "risers",
    "turned_steps",  # consumed by stock inference rather than cut as features
    "plates",
    "countersinks",
    "flats_evidence",
    "diameters",
}
