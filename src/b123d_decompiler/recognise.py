"""Stage 1: STEP in, quiddity recognition document plus the local working shape out."""

from __future__ import annotations

from build123d import Location, Part, Plane
from quiddity import import_step_geometry
from quiddity.document import build_recognition_document


def frame_plane(frame: dict) -> Plane:
    """The plane whose location maps local coordinates to caller coordinates.

    Quiddity maps a local point back as ``origin + x*u + y*v + z*w``, which is exactly
    a rigid placement with columns x, y, z. ``Plane.y_dir`` is ``z_dir cross x_dir``,
    and quiddity's (x, y, z) is right-handed, so x_dir and z_dir fix it.
    """
    return Plane(origin=tuple(frame["origin"]), x_dir=tuple(frame["x"]), z_dir=tuple(frame["z"]))


def to_local(part: Part, frame: dict) -> Part:
    """Express a caller-coordinate part in the recognition frame."""
    return Location(frame_plane(frame)).inverse() * part


def recognise(path: str) -> tuple[dict, Part, Part]:
    """Return (document, caller part, local part) for a STEP file."""
    caller = import_step_geometry(path)
    document = build_recognition_document(caller)
    return document, caller, to_local(caller, document["frame"])
