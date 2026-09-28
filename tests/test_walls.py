"""Wall pieces drawn from a pair of faces, checked against the wall they came from."""

from __future__ import annotations

import pytest
from build123d import Axis, Circle, Plane, Pos, Rectangle, Rot, revolve

from b123d_decompiler.geom import mesh_iou, run_source
from b123d_decompiler.sheet import _torus_source


def _torus_wall(sweep: float, placement):
    """A 3 mm wall bent round a 20 mm centre circle through `sweep` degrees."""
    section = Pos(20, 0) * (Circle(8) - Circle(5)) & Pos(24, 4) * Rectangle(8, 8)
    return placement * revolve(Plane.XZ * section, Axis.Z, sweep)


@pytest.mark.parametrize("sweep", [90.0, 200.0])
@pytest.mark.parametrize("placement", [Pos(0, 0, 0), Pos(5, -3, 2) * Rot(30, 45, 0)])
def test_a_torus_wall_is_drawn_as_its_ring_sector_revolved(sweep, placement):
    wall = _torus_wall(sweep, placement)
    tori = [face for face in wall.faces() if face.geom_type.name == "TORUS"]
    assert len(tori) == 2
    source = _torus_source(*tori)
    assert source is not None
    piece = run_source(["_pieces = []", *source, "tool = _pieces[0]"])
    assert piece.volume == pytest.approx(wall.volume, rel=1e-3)
    # Counted, not intersected: an exact boolean of two solids that share every face
    # can come back nearly empty even when they are the same shape.
    assert mesh_iou(piece, wall) > 0.99
