"""Is a point inside the part? Asked of a triangle mesh, many points at a time.

The kernel's own classifier answers one point at a time and, on a part with threads
or other free-form faces, takes tens of milliseconds to do it. Measuring how much of a
trim is material takes hundreds of points, and a part can have a couple of hundred
trims, so on one threaded connector that alone was twenty minutes.

A mesh answers the same question by counting how often a ray from the point crosses
the surface, and numpy can do that for thousands of points at once. The answer agrees
with the exact one everywhere except within the mesh tolerance of the surface, which
is set fine enough that the volumes this is used to measure do not notice.
"""

from __future__ import annotations

import copy

import numpy as np

#: Mesh tolerance as a share of the part's diagonal.
DEFLECTION_SHARE = 2e-4

#: Grid cells per side for sorting triangles by where they project.
GRID = 48


def mesh_triangles(part, deflection: float) -> np.ndarray:
    """Every triangle of a fine mesh of the part, as an array of shape (n, 3, 3).

    The mesh is built on a copy, so the part itself is left exactly as it was.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    shape = copy.deepcopy(part).wrapped
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.2, True)
    triangles = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        where = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, where)
        if mesh is not None:
            move = where.Transformation()
            nodes = []
            for i in range(1, mesh.NbNodes() + 1):
                point = mesh.Node(i).Transformed(move)
                nodes.append((point.X(), point.Y(), point.Z()))
            nodes = np.asarray(nodes)
            for i in range(1, mesh.NbTriangles() + 1):
                a, b, c = mesh.Triangle(i).Get()
                triangles.append(nodes[[a - 1, b - 1, c - 1]])
        explorer.Next()
    return np.asarray(triangles, dtype=float)


class MeshInside:
    """Point containment against a mesh, by counting ray crossings along each axis.

    A ray along a principal axis turns the crossing test into a 2D point-in-triangle
    test in the other two coordinates, which a grid can narrow down to a handful of
    candidate triangles per point. Three rays, one per axis, vote, so a ray that grazes
    an edge exactly and miscounts is outvoted by the other two.
    """

    def __init__(self, part):
        box = part.bounding_box()
        self.low = np.array([box.min.X, box.min.Y, box.min.Z])
        self.high = np.array([box.max.X, box.max.Y, box.max.Z])
        diagonal = float(np.linalg.norm(self.high - self.low))
        self.triangles = mesh_triangles(part, max(diagonal * DEFLECTION_SHARE, 1e-4))
        self.axes = [self._index(axis) for axis in range(3)]

    def _index(self, axis: int):
        """Triangles sorted into grid cells by their shadow across the ray axis."""
        first, second = (k for k in range(3) if k != axis)
        low = self.low[[first, second]]
        span = np.maximum(self.high[[first, second]] - low, 1e-12)
        flat = self.triangles[:, :, [first, second]]
        lo_cell = np.floor((flat.min(axis=1) - low) / span * GRID).astype(int)
        hi_cell = np.floor((flat.max(axis=1) - low) / span * GRID).astype(int)
        lo_cell = np.clip(lo_cell, 0, GRID - 1)
        hi_cell = np.clip(hi_cell, 0, GRID - 1)
        cells: dict[tuple[int, int], list[int]] = {}
        for index in range(len(self.triangles)):
            for u in range(lo_cell[index, 0], hi_cell[index, 0] + 1):
                for v in range(lo_cell[index, 1], hi_cell[index, 1] + 1):
                    cells.setdefault((u, v), []).append(index)
        members = {key: np.asarray(value) for key, value in cells.items()}
        return first, second, low, span, members

    def _crossings(self, points: np.ndarray, axis: int) -> np.ndarray:
        """How many times a ray from each point, in the + direction of `axis`, crosses."""
        first, second, low, span, members = self.axes[axis]
        count = np.zeros(len(points), dtype=int)
        cell = np.floor((points[:, [first, second]] - low) / span * GRID).astype(int)
        inside_grid = np.all((cell >= 0) & (cell < GRID), axis=1)
        keys = cell[:, 0] * GRID + cell[:, 1]
        for key in np.unique(keys[inside_grid]):
            chosen = np.nonzero(inside_grid & (keys == key))[0]
            candidates = members.get((int(key) // GRID, int(key) % GRID))
            if candidates is None:
                continue
            tri = self.triangles[candidates]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            p = points[chosen]
            # Barycentric test in the plane across the ray, all points by all triangles.
            pu, pv = p[:, None, first], p[:, None, second]
            au, av = a[None, :, first], a[None, :, second]
            bu, bv = b[None, :, first], b[None, :, second]
            cu, cv = c[None, :, first], c[None, :, second]
            det = (bv - cv) * (au - cu) + (cu - bu) * (av - cv)
            with np.errstate(divide="ignore", invalid="ignore"):
                w1 = ((bv - cv) * (pu - cu) + (cu - bu) * (pv - cv)) / det
                w2 = ((cv - av) * (pu - cu) + (au - cu) * (pv - cv)) / det
                w3 = 1.0 - w1 - w2
                hit = (det != 0) & (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
                # Where the ray meets the triangle's plane, along the ray axis.
                height = (
                    w1 * a[None, :, axis] + w2 * b[None, :, axis] + w3 * c[None, :, axis]
                )
                hit &= height > p[:, None, axis]
            count[chosen] = hit.sum(axis=1)
        return count

    def contains(self, points) -> np.ndarray:
        """Which of the points are inside the part."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        votes = sum((self._crossings(points, axis) % 2 == 1).astype(int) for axis in range(3))
        return votes >= 2

    def first_hit(self, points, axis: int, sign: float) -> np.ndarray:
        """Distance from each point along an axis, one way, to the first surface met.

        Infinity where nothing is met. Used to find how far the space in front of a
        face runs before it reaches the part again, without building anything.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        first, second, low, span, members = self.axes[axis]
        found = np.full(len(points), np.inf)
        cell = np.floor((points[:, [first, second]] - low) / span * GRID).astype(int)
        inside_grid = np.all((cell >= 0) & (cell < GRID), axis=1)
        keys = cell[:, 0] * GRID + cell[:, 1]
        for key in np.unique(keys[inside_grid]):
            chosen = np.nonzero(inside_grid & (keys == key))[0]
            candidates = members.get((int(key) // GRID, int(key) % GRID))
            if candidates is None:
                continue
            tri = self.triangles[candidates]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            p = points[chosen]
            pu, pv = p[:, None, first], p[:, None, second]
            au, av = a[None, :, first], a[None, :, second]
            bu, bv = b[None, :, first], b[None, :, second]
            cu, cv = c[None, :, first], c[None, :, second]
            det = (bv - cv) * (au - cu) + (cu - bu) * (av - cv)
            with np.errstate(divide="ignore", invalid="ignore"):
                w1 = ((bv - cv) * (pu - cu) + (cu - bu) * (pv - cv)) / det
                w2 = ((cv - av) * (pu - cu) + (au - cu) * (pv - cv)) / det
                w3 = 1.0 - w1 - w2
                hit = (det != 0) & (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
                height = (
                    w1 * a[None, :, axis] + w2 * b[None, :, axis] + w3 * c[None, :, axis]
                )
                distance = (height - p[:, None, axis]) * sign
                distance = np.where(hit & (distance > 0), distance, np.inf)
            found[chosen] = distance.min(axis=1)
        return found

    def crossings(self, points, axis: int, sign: float) -> list[np.ndarray]:
        """Every distance at which a ray from each point, one way along an axis, meets
        the surface, nearest first. A ray from outside the part alternates entering and
        leaving it, so these pair up into the stretches of the ray that are material."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        first, second, low, span, members = self.axes[axis]
        found = [np.empty(0) for _ in range(len(points))]
        cell = np.floor((points[:, [first, second]] - low) / span * GRID).astype(int)
        inside_grid = np.all((cell >= 0) & (cell < GRID), axis=1)
        keys = cell[:, 0] * GRID + cell[:, 1]
        for key in np.unique(keys[inside_grid]):
            chosen = np.nonzero(inside_grid & (keys == key))[0]
            candidates = members.get((int(key) // GRID, int(key) % GRID))
            if candidates is None:
                continue
            tri = self.triangles[candidates]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            p = points[chosen]
            pu, pv = p[:, None, first], p[:, None, second]
            au, av = a[None, :, first], a[None, :, second]
            bu, bv = b[None, :, first], b[None, :, second]
            cu, cv = c[None, :, first], c[None, :, second]
            det = (bv - cv) * (au - cu) + (cu - bu) * (av - cv)
            with np.errstate(divide="ignore", invalid="ignore"):
                w1 = ((bv - cv) * (pu - cu) + (cu - bu) * (pv - cv)) / det
                w2 = ((cv - av) * (pu - cu) + (au - cu) * (pv - cv)) / det
                w3 = 1.0 - w1 - w2
                hit = (det != 0) & (w1 >= 0) & (w2 >= 0) & (w3 >= 0)
                height = (
                    w1 * a[None, :, axis] + w2 * b[None, :, axis] + w3 * c[None, :, axis]
                )
                distance = (height - p[:, None, axis]) * sign
                keep = hit & (distance > 0)
            for row, index in enumerate(chosen):
                found[index] = np.unique(np.round(distance[row][keep[row]], 9))
        return found
