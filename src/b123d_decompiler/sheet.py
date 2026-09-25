"""Sheet metal: a part folded from one sheet, rebuilt as its flanges and bends.

Quiddity recognises a sheet-metal body and publishes its thickness, its flanges (the
flat stretches, each with a reference face on one side of the sheet) and its bends
(each a pair of coaxial cylinders a thickness apart, with its axis, angle and inner
radius). That is how the part was designed, and it is also exact: a flange is its
reference face pushed through the sheet, and a bend is the ring sector between its
two cylinders. The two meet face to face along each bend line, so the pieces join into
one solid, where thickening faces one by one leaves gaps at every corner.

Holes and cutouts through a flange are not drawn here: each has its own record, and
they are cut afterwards like any other feature.
"""

from __future__ import annotations

import math

from .geom import Context
from .model import fmt, fmt_tuple

#: Pieces that should touch can miss by a rounding error in the emitted outlines, so
#: they are fused with this much tolerance, in millimetres.
JOIN_TOLERANCE = 1e-3


def sheet_metal_record(document: dict) -> dict | None:
    """The sheet-metal body quiddity found, if any."""
    for feature in document.get("features") or ():
        if feature.get("family") == "sheet_metal_bodies":
            return feature["record"]
    return None


def sheet_metal_stock(document: dict, ctx: Context) -> tuple[str, list[str]] | None:
    """(label, code) building the folded sheet, or None when the part is not one."""
    from .adapters import face_profile_source

    record = sheet_metal_record(document)
    if record is None or not record.get("flanges"):
        return None
    faces = ctx.part.faces()
    thickness = float(record["thickness"])
    code = [f"THICKNESS = {fmt(thickness)}", "_pieces = []"]
    for flange in record["flanges"]:
        for index in flange["reference_faces"]:
            face = faces[index]
            normal = face.normal_at(face.center())
            outward = (normal.X, normal.Y, normal.Z)
            drawn = face_profile_source(face, outward)
            if drawn is None:
                return None
            lines, _origin = drawn
            code.append(f"# flange {flange['index']}")
            code += lines
            code.append(
                "_pieces.append(extrude(_plane * make_face(_prof), amount=THICKNESS, "
                f"dir={fmt_tuple(tuple(-v for v in outward))}))"
            )
    for bend in record.get("bends") or ():
        for pair in bend["face_pairs"]:
            piece = _bend_source(faces[pair["first_face"]], faces[pair["second_face"]])
            if piece is None:
                return None
            code.append(
                f"# bend {bend['index']}: {fmt(float(bend['angle_degrees']), 3)}°, "
                f"inner radius {fmt(float(bend['inner_radius']), 3)}"
            )
            code += piece
    code.append(f"part = _pieces[0].fuse(*_pieces[1:], tol={JOIN_TOLERANCE}).clean()")
    bends = len(record.get("bends") or ())
    label = (
        f"stock: sheet metal {fmt(thickness, 3)} thick, "
        f"{len(record['flanges'])} flanges and {bends} bends"
    )
    return label, code


def _bend_source(first, second) -> list[str] | None:
    """The ring sector between a bend's two cylinders, as extrusion source."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Cylinder

    surface = BRepAdaptor_Surface(first.wrapped)
    if surface.GetType() != GeomAbs_Cylinder:
        return None
    cylinder = surface.Cylinder()
    inner = min(first.radius, second.radius)
    outer = max(first.radius, second.radius)
    axis = cylinder.Axis()
    along = axis.Direction().Coord()
    across = cylinder.Position().XDirection().Coord()
    u_low, u_high, v_low, v_high = BRepTools.UVBounds_s(first.wrapped)
    base = tuple(axis.Location().Coord()[k] + along[k] * v_low for k in range(3))

    def rim(radius, angle):
        return (radius * math.cos(angle), radius * math.sin(angle))

    middle = (u_low + u_high) / 2
    outer_start, outer_mid, outer_end = (rim(outer, a) for a in (u_low, middle, u_high))
    inner_end, inner_mid, inner_start = (rim(inner, a) for a in (u_high, middle, u_low))
    return [
        (
            f"_bend = Plane(origin={fmt_tuple(base)}, x_dir={fmt_tuple(across)}, "
            f"z_dir={fmt_tuple(along)})"
        ),
        (
            f"_prof = ThreePointArc({fmt_tuple(outer_start)}, {fmt_tuple(outer_mid)}, "
            f"{fmt_tuple(outer_end)}) + Line({fmt_tuple(outer_end)}, {fmt_tuple(inner_end)}) "
            f"+ ThreePointArc({fmt_tuple(inner_end)}, {fmt_tuple(inner_mid)}, "
            f"{fmt_tuple(inner_start)}) + Line({fmt_tuple(inner_start)}, {fmt_tuple(outer_start)})"
        ),
        f"_pieces.append(extrude(_bend * make_face(_prof), amount={fmt(v_high - v_low)}))",
    ]
