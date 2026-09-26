"""A thin-walled panel with free-form skins, rebuilt as a body less its cavity.

Some thin-walled parts are panels: a plan outline, bounded top and bottom (and
perhaps along a sloped side) by free-form skins, and hollowed to a constant wall.
Drawing their walls one by one fails at the free-form skins, where the kernel's offset
of a whole shell drops material and face-by-face pieces will not join. They are drawn
the way such a part is designed instead: the plan outline pulled through the part and
cut back by each outer skin, less the same thing done with the inner skins and the
plan inset by the wall. Quiddity's thin-wall record says which faces are outside and
inside; its free-form records say which skins are offsets of which.

A skin is written exactly, as its B-spline surface, extended past its edges so that
the cut it makes runs cleanly through the plan prism. The script defines one helper
for that and calls it once per skin.
"""

from __future__ import annotations

from .geom import Context
from .model import fmt, fmt_tuple

#: A face counts as a wall of the plan when its normal is within this of square to the
#: plan direction; any other outer or inner face bounds the panel and cuts it.
WALL_SLOPE = 0.1

#: How far each skin is extended past its edges, as a share of the part's diagonal,
#: so the cut it makes crosses the whole plan prism.
EXTENSION_SHARE = 0.2

#: How far a skin is swept away from the material, as a share of the diagonal: far
#: enough to clear the part, no further, since an extended skin curls and a long
#: sweep of it runs back through the body.
SWEEP_SHARE = 1.0

#: A panel is drawn this way only when this much of the outer area is free-form.
FREEFORM_SHARE = 0.3

BSPLINE_HELPER = [
    "def _bspline_face(u_degree, v_degree, poles, weights, u_knots, v_knots, u_mults, v_mults):",
    '    """A face on an exact B-spline surface, from its control net and knots."""',
    "    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace",
    "    from OCP.Geom import Geom_BSplineSurface",
    "    from OCP.TColgp import TColgp_Array2OfPnt",
    "    from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal",
    "    from OCP.gp import gp_Pnt",
    "",
    "    net = TColgp_Array2OfPnt(1, len(poles), 1, len(poles[0]))",
    "    mass = TColStd_Array2OfReal(1, len(poles), 1, len(poles[0]))",
    "    for i, row in enumerate(poles, 1):",
    "        for j, (x, y, z) in enumerate(row, 1):",
    "            net.SetValue(i, j, gp_Pnt(x, y, z))",
    "            mass.SetValue(i, j, weights[i - 1][j - 1])",
    "",
    "    def reals(values):",
    "        array = TColStd_Array1OfReal(1, len(values))",
    "        for k, value in enumerate(values, 1):",
    "            array.SetValue(k, value)",
    "        return array",
    "",
    "    def counts(values):",
    "        array = TColStd_Array1OfInteger(1, len(values))",
    "        for k, value in enumerate(values, 1):",
    "            array.SetValue(k, value)",
    "        return array",
    "",
    "    surface = Geom_BSplineSurface(",
    "        net, mass, reals(u_knots), reals(v_knots), counts(u_mults), counts(v_mults),",
    "        u_degree, v_degree, False, False,",
    "    )",
    "    return Face(BRepBuilderAPI_MakeFace(surface, 1e-6).Face())",
]


def thin_wall_record(document: dict) -> dict | None:
    for feature in document.get("features") or ():
        if feature.get("family") == "thin_wall_bodies":
            return feature["record"]
    return None


def panel_stock(document: dict, ctx: Context) -> tuple[str, list[str]] | None:
    """(label, code) building a free-form thin-walled panel, or None when it is not one."""
    record = thin_wall_record(document)
    if record is None:
        return None
    hint = record.get("history_hint") or {}
    outer = tuple(hint.get("outer_faces") or ())
    inner = tuple(hint.get("inner_faces") or ())
    if not outer or not inner:
        return None
    faces = ctx.part.faces()
    free_area = sum(faces[i].area for i in outer if faces[i].geom_type.name == "BSPLINE")
    if free_area < FREEFORM_SHARE * sum(faces[i].area for i in outer):
        return None
    thickness = float(record["thickness"])
    axis = _plan_axis(faces, outer)
    extension = EXTENSION_SHARE * ctx.diagonal

    plan = _plan_polygon(ctx, axis)
    if plan is None:
        return None
    inset = plan.buffer(-thickness, join_style="mitre")
    inset = _largest(inset)
    if inset is None:
        return None

    low = ctx.bb_min[axis] - ctx.margin
    length = ctx.bb_max[axis] - ctx.bb_min[axis] + 2 * ctx.margin
    code = list(BSPLINE_HELPER) + ["", f"THICKNESS = {fmt(thickness)}"]
    code += _plan_prism(plan, axis, low, length, "body")
    for index in outer:
        cut = _skin_cut(faces[index], axis, extension, ctx, target="body")
        if cut is None and faces[index].geom_type.name == "BSPLINE":
            return None
        code += cut or []
    for rim in record.get("rim_regions") or ():
        for index in rim:
            code += _rim_cut(faces[index], ctx) or []
    # Cutting a prism back to curved skins leaves slivers beyond them; the panel is
    # the one body they bound.
    code.append("body = max(body.solids(), key=lambda s: s.volume)")
    code += _plan_prism(inset, axis, low, length, "cavity")
    for index in inner:
        cut = _skin_cut(faces[index], axis, extension, ctx, target="cavity")
        if cut is None and faces[index].geom_type.name == "BSPLINE":
            return None
        code += cut or []
    code += [
        "part = body - cavity",
        "part = max(part.solids(), key=lambda s: s.volume)",
    ]
    code = _checked(code)
    if code is None:
        return None
    label = f"stock: thin panel {fmt(thickness, 3)} thick, body less cavity"
    return label, code


def _plan_axis(faces, outer) -> int:
    """The direction the panel's walls stand along: the most wall area square to it."""
    best, chosen = -1.0, 2
    for axis in range(3):
        area = 0.0
        for index in outer:
            face = faces[index]
            if face.geom_type.name == "BSPLINE":
                continue
            normal = face.normal_at(face.center())
            if abs((normal.X, normal.Y, normal.Z)[axis]) < WALL_SLOPE:
                area += face.area
        if area > best:
            best, chosen = area, axis
    return chosen


def _largest(geometry):
    import shapely

    parts = [g for g in shapely.get_parts(geometry) if isinstance(g, shapely.Polygon)]
    return max(parts, key=lambda g: g.area) if parts else None


def _plan_polygon(ctx: Context, axis: int):
    """The part's shadow along the plan axis, with the holes that go right through."""
    import shapely

    from .geom import _mesh_for
    from .outlines import _projected

    mesh = _mesh_for(ctx.part)
    shadow = _projected(mesh.triangles, axis)
    if shadow is None:
        return None
    tolerance = ctx.diagonal * 2e-4
    polygon = _largest(shadow.simplify(tolerance))
    if polygon is None:
        return None
    holes = [ring for ring in polygon.interiors if shapely.Polygon(ring).area > 25 * tolerance**2]
    return shapely.Polygon(polygon.exterior, holes)


def _ring_source(ring, variable: str, tolerance: float) -> str:
    from .outlines import _arcs_and_lines

    corners = [(float(u), float(v)) for u, v in list(ring.coords)[:-1]]
    drawn = []
    for segment in _arcs_and_lines(corners, tolerance):
        if segment[0] == "line":
            drawn.append(f"Line({fmt_tuple(segment[1])}, {fmt_tuple(segment[2])})")
        else:
            drawn.append(
                f"ThreePointArc({fmt_tuple(segment[1])}, {fmt_tuple(segment[2])}, "
                f"{fmt_tuple(segment[3])})"
            )
    joined = "\n    + ".join(drawn)
    return f"{variable} = (\n    {joined}\n)"


def _plan_prism(polygon, axis: int, low: float, length: float, target: str) -> list[str]:
    """Source extruding a plan outline, holes and all, right through the part."""
    x_dir = tuple(1.0 if k == next(j for j in range(3) if j != axis) else 0.0 for k in range(3))
    z_dir = tuple(1.0 if k == axis else 0.0 for k in range(3))
    origin = tuple(low if k == axis else 0.0 for k in range(3))
    code = [
        _ring_source(polygon.exterior, "_prof", 1e-3),
        "_sheet = make_face(_prof)",
    ]
    for ring in polygon.interiors:
        code += [_ring_source(ring, "_hole", 1e-3), "_sheet -= make_face(_hole)"]
    code += [
        (
            f"_plane = Plane(origin={fmt_tuple(origin)}, x_dir={fmt_tuple(x_dir)}, "
            f"z_dir={fmt_tuple(z_dir)})"
        ),
        f"{target} = extrude(_plane * _sheet, amount={fmt(length)})",
    ]
    return code


def _skin_cut(face, axis: int, extension: float, ctx: Context, target: str) -> list[str] | None:
    """Source cutting away everything on the far side of a skin, or None for a wall.

    The skin's surface is extended past its edges and swept away from the material
    it bounds, far enough to clear the part.
    """
    normal = face.normal_at(face.center())
    outward = (normal.X, normal.Y, normal.Z)
    if abs(outward[axis]) < WALL_SLOPE and face.geom_type.name != "BSPLINE":
        return []  # a wall of the plan: the prism already has it
    if face.geom_type.name != "BSPLINE":
        return None
    net = _extended_net(face, extension)
    if net is None:
        return None
    sweep = SWEEP_SHARE * ctx.diagonal
    # A face's own normal points out of the material it bounds; on the cavity's side
    # that is into the cavity, so the sweep goes the other way.
    direction = outward if target == "body" else tuple(-v for v in outward)
    return [
        f"_skin = _bspline_face({net})",
        f"{target} -= extrude(_skin, amount={fmt(sweep)}, dir={fmt_tuple(direction)})",
    ]


def _extended_net(face, extension: float) -> str | None:
    """A face's whole B-spline support surface, as helper arguments."""
    from OCP.BRep import BRep_Tool
    from OCP.Geom import Geom_BSplineSurface

    surface = BRep_Tool.Surface_s(face.wrapped)
    if not isinstance(surface, Geom_BSplineSurface):
        try:
            surface = Geom_BSplineSurface.DownCast(surface)
        except Exception:  # noqa: BLE001
            return None
    if surface is None or surface.IsUPeriodic() or surface.IsVPeriodic():
        return None
    # The support surface as the file defines it runs well past the trimmed skin,
    # which is what the cut needs. (OCCT's ExtendSurfByLength cannot help from here:
    # it replaces the surface behind a handle Python never sees.)
    rows, cols = surface.NbUPoles(), surface.NbVPoles()
    poles = [
        [tuple(round(c, 6) for c in surface.Pole(i, j).Coord()) for j in range(1, cols + 1)]
        for i in range(1, rows + 1)
    ]
    weights = [[round(surface.Weight(i, j), 9) for j in range(1, cols + 1)] for i in range(1, rows + 1)]
    u_knots = [round(surface.UKnot(k), 9) for k in range(1, surface.NbUKnots() + 1)]
    v_knots = [round(surface.VKnot(k), 9) for k in range(1, surface.NbVKnots() + 1)]
    u_mults = [surface.UMultiplicity(k) for k in range(1, surface.NbUKnots() + 1)]
    v_mults = [surface.VMultiplicity(k) for k in range(1, surface.NbVKnots() + 1)]
    return (
        f"{surface.UDegree()}, {surface.VDegree()}, {poles!r}, {weights!r}, "
        f"{u_knots!r}, {v_knots!r}, {u_mults!r}, {v_mults!r}"
    )


def _rim_cut(face, ctx: Context) -> list[str] | None:
    """Source cutting the body back to a flat opening rim, the wall's cut end."""
    if face.geom_type.name != "PLANE":
        return None
    centre = face.center()
    normal = face.normal_at(centre)
    size = 4 * ctx.diagonal
    return [
        (
            f"_rim = Plane(origin={fmt_tuple((centre.X, centre.Y, centre.Z))}, "
            f"z_dir={fmt_tuple((normal.X, normal.Y, normal.Z))})"
        ),
        (
            f"body -= _rim * Box({fmt(size)}, {fmt(size)}, {fmt(size)}, "
            "align=(Align.CENTER, Align.CENTER, Align.MIN))"
        ),
    ]


def _checked(code: list[str]) -> list[str] | None:
    """The source, less any cut the kernel gets wrong.

    Each statement is run as the script will run it. A cut that removes more than its
    tool overlaps, or leaves an invalid solid, is the boolean failing rather than the
    skin, and it is dropped so the rest of the body survives.
    """
    helper = len(BSPLINE_HELPER)
    space: dict = {}
    exec("from build123d import *\n" + "\n".join(code[:helper]), space)  # noqa: S102
    kept = list(code[:helper])
    for statement in code[helper:]:
        target = next((name for name in ("body", "cavity") if statement.startswith(f"{name} -= ")), None)
        if target is None:
            exec(statement, space)  # noqa: S102
            kept.append(statement)
            continue
        before = space[target]
        try:
            tool = eval(statement.split("-=", 1)[1].strip(), space)
            common = before & tool
            overlap = common.volume if common is not None else 0.0
            after = before - tool
            removed = before.volume - after.volume
            if not after.is_valid or after.volume <= 0 or removed > overlap * 1.05 + 1e-6:
                continue
        except Exception:  # noqa: BLE001 - a cut that fails is left out
            continue
        space[target] = after
        kept.append(statement)
    part = space.get("part")
    if part is None or part.volume <= 0:
        return None
    return kept
