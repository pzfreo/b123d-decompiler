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

#: Places the sheet's coordinates are written to. The flanges and bends must meet
#: face to face; at the usual four places they miss by a few hundred-thousandths of a
#: millimetre, and joining them across that with a looser tolerance leaves a solid
#: whose tolerances then make later cuts go wrong.
PLACES = 7

#: What is left of the gap at PLACES is closed by fusing with this much tolerance, in
#: millimetres.
JOIN_TOLERANCE = 1e-5


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
            drawn = face_profile_source(face, outward, places=PLACES)
            if drawn is None:
                return None
            lines, _origin = drawn
            code.append(f"# flange {flange['index']}")
            code += lines
            code.append(
                "_pieces.append(extrude(_plane * make_face(_prof), amount=THICKNESS, "
                f"dir={fmt_tuple(tuple(-v for v in outward), PLACES)}))"
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
    # A formed feature (a tab bent out of the sheet, say) is made of the same pieces:
    # plane pairs a thickness apart are more flanges and coaxial cylinder pairs a
    # thickness apart are more bends. Anything else in one is left to the features.
    for number, formed in enumerate(record.get("formed_features") or ()):
        found = _formed_pieces(formed, faces, thickness)
        if found:
            code.append(f"# formed feature {number}")
            code += found
    code.append(f"part = _pieces[0].fuse(*_pieces[1:], tol={JOIN_TOLERANCE}).clean()")
    bends = len(record.get("bends") or ())
    label = (
        f"stock: sheet metal {fmt(thickness, 3)} thick, "
        f"{len(record['flanges'])} flanges and {bends} bends"
    )
    return label, code


def _formed_pieces(formed: dict, faces, thickness: float) -> list[str]:
    """Flanges and bends found inside a formed feature, as extrusion source."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    from .adapters import face_profile_source

    tolerance = max(thickness * 0.02, 1e-3)
    planes, cylinders = [], []
    for index in formed.get("faces") or ():
        face = faces[index]
        kind = face.geom_type.name
        if kind == "PLANE":
            planes.append(face)
        elif kind == "CYLINDER":
            axis = BRepAdaptor_Surface(face.wrapped).Cylinder().Axis()
            cylinders.append((face, axis))
    code: list[str] = []
    used: set[int] = set()
    for first_index, first in enumerate(planes):
        if first_index in used:
            continue
        normal = first.normal_at(first.center())
        for second_index in range(first_index + 1, len(planes)):
            second = planes[second_index]
            other = second.normal_at(second.center())
            gap = (second.center() - first.center()).dot(normal)
            if normal.dot(other) < -0.9999 and abs(abs(gap) - thickness) < tolerance:
                outward = (normal.X, normal.Y, normal.Z)
                drawn = face_profile_source(first, outward, places=PLACES)
                if drawn is None:
                    break
                code += drawn[0]
                code.append(
                    "_pieces.append(extrude(_plane * make_face(_prof), amount=THICKNESS, "
                    f"dir={fmt_tuple(tuple(-v for v in outward), PLACES)}))"
                )
                used.update((first_index, second_index))
                break
    paired: set[int] = set()
    for first_index, (first, first_axis) in enumerate(cylinders):
        if first_index in paired:
            continue
        for second_index in range(first_index + 1, len(cylinders)):
            second, second_axis = cylinders[second_index]
            if (
                first_axis.IsCoaxial(second_axis, 1e-6, tolerance)
                and abs(abs(first.radius - second.radius) - thickness) < tolerance
            ):
                piece = _bend_source(first, second)
                if piece:
                    code += piece
                    paired.update((first_index, second_index))
                break
    return code


def _bend_source(first, second) -> list[str] | None:
    """The ring sector between a bend's two cylinders, as extrusion source."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Cylinder

    # The arc comes from whichever cylinder spans more of it: where a bend runs into a
    # rounded corner, one of its two faces can be left as a sliver.
    def span(face):
        u_low, u_high, _v_low, _v_high = BRepTools.UVBounds_s(face.wrapped)
        return u_high - u_low

    if span(second) > span(first):
        first, second = second, first
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
            f"_bend = Plane(origin={fmt_tuple(base, PLACES)}, x_dir={fmt_tuple(across, PLACES)}, "
            f"z_dir={fmt_tuple(along, PLACES)})"
        ),
        (
            f"_prof = ThreePointArc({fmt_tuple(outer_start, PLACES)}, {fmt_tuple(outer_mid, PLACES)}, "
            f"{fmt_tuple(outer_end, PLACES)}) + Line({fmt_tuple(outer_end, PLACES)}, {fmt_tuple(inner_end, PLACES)}) "
            f"+ ThreePointArc({fmt_tuple(inner_end, PLACES)}, {fmt_tuple(inner_mid, PLACES)}, "
            f"{fmt_tuple(inner_start, PLACES)}) + Line({fmt_tuple(inner_start, PLACES)}, {fmt_tuple(outer_start, PLACES)})"
        ),
        f"_pieces.append(extrude(_bend * make_face(_prof), amount={fmt(v_high - v_low, PLACES)}))",
    ]
