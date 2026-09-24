"""Choosing an outline billet by counting, before any solid is built.

Every candidate billet used to be built as a solid just to learn its volume and
whether it held the part, and building them was where the time and the kernel
failures went. Here each candidate is a raster read off a dense sample of the part's
surface, which answers both questions for every candidate and every combination of
candidates in a fraction of a second. Only the one chosen is then built, emitted and
checked exactly.

Two kinds of outline per axis:

- a shadow: every cell across the axis that the part covers anywhere along it, run
  the whole length of the axis, which is how a plate is cut from its own outline;
- a stepped billet: the same, but slice by slice, so each stretch of the axis is only
  as wide as the part is there, which is how a part is drawn as a few extrusions.

Rasters are grown by one cell in every direction across the axis, so a candidate
never looks as if it misses a piece of the part because a thin wall fell between
samples. They are only estimates; the billet that is built is held to the exact test.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

#: Cells per side across the axis.
GRID = 64

#: Step counts offered for a stepped billet: the best billet with at most this many.
STEP_COUNTS = (2, 3, 4, 6, 8)


def _surface_points(triangles: np.ndarray, spacing: float) -> np.ndarray:
    """Points over every triangle, no further apart than `spacing`."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    longest = np.maximum.reduce(
        [np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)]
    )
    steps = np.maximum(1, np.ceil(longest / spacing)).astype(int)
    points = [triangles.reshape(-1, 3)]
    for count in np.unique(steps):
        if count <= 1:
            continue
        chosen = steps == count
        grid = [
            (i / count, j / count)
            for i in range(count + 1)
            for j in range(count + 1 - i)
        ]
        weights = np.asarray(grid)
        u, v = weights[:, 0][None, :, None], weights[:, 1][None, :, None]
        base, first, second = a[chosen][:, None], (b - a)[chosen][:, None], (c - a)[chosen][:, None]
        points.append((base + u * first + v * second).reshape(-1, 3))
    return np.concatenate(points)


class Outline:
    """One candidate: a predicate over points, its estimated volume, and its size."""

    def __init__(self, kind: str, index: int, contains, volume: float, extrusions: int, marks):
        self.kind, self.index = kind, index
        self.contains, self.volume = contains, volume
        self.extrusions, self.marks = extrusions, marks

    @property
    def name(self) -> str:
        steps = f" in {self.extrusions} steps" if self.kind == "stepped" else ""
        return f"{self.kind} along {'xyz'[self.index]}{steps}"


class OutlineEstimator:
    """Rasters of every outline candidate, read off the part's mesh."""

    def __init__(self, mesh, low, high, marks_by_axis):
        self.mesh, self.low, self.high = mesh, np.asarray(low), np.asarray(high)
        self.size = np.maximum(self.high - self.low, 1e-9)
        # Samples no further apart than a cell on the smallest side; the rasters are
        # grown by a cell afterwards, which covers what falls between samples.
        cell = float(np.min(self.size) / GRID)
        self.surface = _surface_points(mesh.triangles, max(cell, 1e-6))
        self.marks_by_axis = marks_by_axis

    def _cells(self, index: int, points: np.ndarray) -> np.ndarray:
        first, second = (k for k in range(3) if k != index)
        across = np.stack([points[:, first], points[:, second]], axis=1)
        low = self.low[[first, second]]
        span = self.size[[first, second]]
        cells = np.floor((across - low) / span * GRID).astype(int)
        return np.clip(cells, 0, GRID - 1)

    @staticmethod
    def _grow(raster: np.ndarray) -> np.ndarray:
        """One cell wider all round, without wrapping past the edge of the grid."""
        padded = np.pad(raster, 1)
        grown = np.zeros_like(raster)
        for du, dv in itertools.product((0, 1, 2), repeat=2):
            grown |= padded[du : du + GRID, dv : dv + GRID]
        return grown

    def shadow(self, index: int) -> Outline:
        raster = np.zeros((GRID, GRID), dtype=bool)
        cells = self._cells(index, self.surface)
        raster[cells[:, 0], cells[:, 1]] = True
        raster = self._grow(raster)
        cell_area = np.prod(self.size[[k for k in range(3) if k != index]]) / GRID**2
        volume = float(raster.sum() * cell_area * self.size[index])

        def contains(points, _raster=raster, _index=index):
            where = self._cells(_index, points)
            return _raster[where[:, 0], where[:, 1]]

        return Outline("shadow", index, contains, volume, 1, None)

    def stepped(self, index: int) -> list[Outline]:
        """The best stepped billets along an axis with at most k steps, for each k offered.

        Every slice gets a raster; a step is a run of slices joined, as wide as their
        union all along it. Choosing where the steps break is a partition of the slices,
        and the partition with least volume for a given number of steps falls out of a
        small dynamic program. That is the billet a designer would draw with k
        extrusions, and nothing finer is offered.
        """
        marks = self.marks_by_axis[index]
        if len(marks) < 3:
            return []
        coordinate = self.surface[:, index]
        cells = self._cells(index, self.surface)
        first, second = (k for k in range(3) if k != index)
        centres_u = self.low[first] + (np.arange(GRID) + 0.5) / GRID * self.size[first]
        centres_v = self.low[second] + (np.arange(GRID) + 0.5) / GRID * self.size[second]
        grid_u, grid_v = np.meshgrid(centres_u, centres_v, indexing="ij")
        rasters = []
        for start, finish in itertools.pairwise(marks):
            raster = np.zeros((GRID, GRID), dtype=bool)
            within = (coordinate >= start) & (coordinate <= finish)
            raster[cells[within, 0], cells[within, 1]] = True
            # Columns solid right through the slice have no surface in it to be seen.
            middle = np.empty((GRID * GRID, 3))
            middle[:, index] = (start + finish) / 2
            middle[:, first], middle[:, second] = grid_u.ravel(), grid_v.ravel()
            raster |= self.mesh.contains(middle).reshape(GRID, GRID)
            rasters.append(self._grow(raster))

        count = len(rasters)
        cell_area = float(np.prod(self.size[[first, second]])) / GRID**2
        # cost[i][j]: volume of one step running from slice i to slice j.
        cost = np.full((count, count), np.inf)
        unions = {}
        for i in range(count):
            union = np.zeros((GRID, GRID), dtype=bool)
            for j in range(i, count):
                union = union | rasters[j]
                cost[i, j] = union.sum() * cell_area * (marks[j + 1] - marks[i])
                unions[i, j] = union

        found = []
        most = min(max(STEP_COUNTS), count)
        best = np.full((most + 1, count + 1), np.inf)
        back = np.zeros((most + 1, count + 1), dtype=int)
        best[0, 0] = 0.0
        for steps in range(1, most + 1):
            for j in range(1, count + 1):
                options = best[steps - 1, :j] + cost[np.arange(j), j - 1]
                back[steps, j] = int(np.argmin(options))
                best[steps, j] = options[back[steps, j]]
        for steps in STEP_COUNTS:
            if steps > most or not np.isfinite(best[steps, count]):
                continue
            edges, j = [count], count
            for level in range(steps, 0, -1):
                j = back[level, j]
                edges.append(j)
            edges = edges[::-1]
            runs = [(edges[k], edges[k + 1] - 1) for k in range(steps) if edges[k + 1] > edges[k]]
            chosen = [unions[i, j] for i, j in runs]
            bounds = [(marks[i], marks[j + 1]) for i, j in runs]
            boundaries = np.asarray([b[0] for b in bounds] + [bounds[-1][1]])
            volume = float(best[steps, count])

            def contains(points, _chosen=chosen, _edges=boundaries, _index=index):
                where = self._cells(_index, points)
                slot = np.searchsorted(_edges, points[:, _index], side="right") - 1
                slot = np.clip(slot, 0, len(_chosen) - 1)
                return np.stack(_chosen)[slot, where[:, 0], where[:, 1]]

            found.append(Outline("stepped", index, contains, volume, len(runs), bounds))
        return found

    def candidates(self) -> list[Outline]:
        found = []
        for index in range(3):
            found.append(self.shadow(index))
            found.extend(self.stepped(index))
        return found


#: What one more extrusion must save, as a share of the billet, to be worth drawing.
EXTRUSION_COST = 0.03


# ── building the chosen outline ─────────────────────────────────────────


def _plane_uv(points: np.ndarray, index: int) -> np.ndarray:
    """Coordinates in the drawing plane of an axis, as the emitted Plane will read them.

    The plane has its x along the first remaining world axis and its normal along the
    axis itself, so its own y runs along the second remaining axis, reversed when the
    axis is y.
    """
    first, second = (k for k in range(3) if k != index)
    sign = -1.0 if index == 1 else 1.0
    return np.stack([points[..., first], sign * points[..., second]], axis=-1)


def _projected(triangles: np.ndarray, index: int):
    """The union of triangles projected across an axis, as a shapely geometry."""
    import shapely

    flat = _plane_uv(triangles, index)
    area = 0.5 * np.abs(
        (flat[:, 1, 0] - flat[:, 0, 0]) * (flat[:, 2, 1] - flat[:, 0, 1])
        - (flat[:, 2, 0] - flat[:, 0, 0]) * (flat[:, 1, 1] - flat[:, 0, 1])
    )
    flat = flat[area > 1e-12]
    if not len(flat):
        return None
    rings = np.concatenate([flat, flat[:, :1]], axis=1)
    return shapely.union_all(shapely.polygons(rings))


def _clipped(triangles: np.ndarray, index: int, low: float, high: float) -> np.ndarray:
    """The parts of the triangles lying between two positions along an axis, as triangles.

    A long face reaches across several steps, and projecting all of it into each one
    makes every step as wide as the whole shadow; only the piece inside the step counts.
    Triangles wholly inside pass through, those wholly outside are dropped, and the
    rest are cut by the two planes and fanned back into triangles.
    """
    height = triangles[:, :, index]
    inside = (height.min(axis=1) >= low) & (height.max(axis=1) <= high)
    outside = (height.max(axis=1) < low) | (height.min(axis=1) > high)
    kept = [triangles[inside]]
    for triangle in triangles[~inside & ~outside]:
        polygon = list(triangle)
        for limit, keep_above in ((low, True), (high, False)):
            clipped = []
            for k in range(len(polygon)):
                a, b = polygon[k], polygon[(k + 1) % len(polygon)]
                a_in = a[index] >= limit if keep_above else a[index] <= limit
                b_in = b[index] >= limit if keep_above else b[index] <= limit
                if a_in:
                    clipped.append(a)
                if a_in != b_in:
                    t = (limit - a[index]) / (b[index] - a[index])
                    clipped.append(a + t * (b - a))
            polygon = clipped
            if len(polygon) < 3:
                break
        for k in range(1, len(polygon) - 1):
            kept.append(np.asarray([[polygon[0], polygon[k], polygon[k + 1]]]))
    return np.concatenate(kept) if kept else np.empty((0, 3, 3))


def _section(triangles: np.ndarray, index: int, at: float, grid: float = 1e-6):
    """The part's cross section at one position along an axis, as a shapely geometry.

    Each triangle the plane crosses gives a segment. Neighbouring segments meet only
    to within rounding, and a loop that does not quite close is no loop at all to
    polygonize, so the ends are snapped to a fine grid first.
    """
    import shapely

    height = triangles[:, :, index] - at
    above = height > 0
    crossing = above.any(axis=1) & ~above.all(axis=1)
    segments = []
    for triangle, h in zip(triangles[crossing], height[crossing], strict=True):
        ends = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (h[a] > 0) != (h[b] > 0):
                t = h[a] / (h[a] - h[b])
                ends.append(triangle[a] + t * (triangle[b] - triangle[a]))
        if len(ends) == 2:
            segments.append(_plane_uv(np.asarray(ends), index))
    if not segments:
        return None
    snapped = np.round(np.asarray(segments) / grid) * grid
    snapped = snapped[np.linalg.norm(snapped[:, 0] - snapped[:, 1], axis=1) > 0]
    if not len(snapped):
        return None
    lines = shapely.multilinestrings(snapped)
    faces = shapely.polygonize(shapely.get_parts(shapely.node(lines)))
    return shapely.union_all(shapely.get_parts(faces)) if not faces.is_empty else None


def outline_polygons(triangles: np.ndarray, index: int, tolerance: float,
                     low: float | None = None, high: float | None = None,
                     envelope=None) -> list:
    """The outline a billet is drawn from, as simple polygons that hold the part.

    Without bounds it is the part's whole shadow across the axis. With bounds it is
    the shadow of the stretch between them: every surface there, projected, and the
    sections just inside each end, which cover a stretch that is solid right through
    and so has no surface inside it. Holes are filled, since a billet has none. The
    outline is grown by twice the mesh tolerance, because a mesh's chords lie inside
    the curves they stand for, and simplified by one, so it has few corners and still
    holds the part.
    """
    import shapely

    if low is None:
        chosen = triangles
        pieces = [_projected(chosen, index)]
    else:
        chosen = _clipped(triangles, index, low, high)
        inset = (high - low) * 1e-3
        pieces = [
            _projected(chosen, index) if len(chosen) else None,
            _section(triangles, index, low + inset, tolerance * 0.1),
            _section(triangles, index, high - inset, tolerance * 0.1),
        ]
    pieces = [p for p in pieces if p is not None and not p.is_empty]
    if not pieces:
        return []
    region = shapely.union_all(pieces)
    filled = [
        shapely.Polygon(part.exterior)
        for part in shapely.get_parts(region)
        if isinstance(part, shapely.Polygon) and part.area > tolerance**2
    ]
    region = shapely.union_all(filled).buffer(2 * tolerance, join_style="mitre", mitre_limit=2.0)
    region = region.simplify(tolerance, preserve_topology=True)
    if envelope is not None:
        # Growing the outline pushes flat sides past the part's box, and the billet
        # has no business being wider than the part anywhere.
        corners = _plane_uv(np.asarray(envelope), index)
        region = region.intersection(
            shapely.box(*corners.min(axis=0), *corners.max(axis=0))
        )
    return [
        shapely.Polygon(part.exterior)
        for part in shapely.get_parts(region)
        if isinstance(part, shapely.Polygon)
    ]


def _circle(a, b, c):
    """Centre and radius of the circle through three points, or None if they are in line."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy), math.dist((ux, uy), a)


def _arcs_and_lines(points: list, tolerance: float) -> list:
    """An outline as the lines and arcs a designer would have drawn it with.

    Simplifying a curve leaves a run of short chords. Where a run of at least four
    vertices lies on one circle to within the tolerance, and turns less than half a
    circle, it is written as one arc through its ends and its middle vertex. That is
    one curved face on the billet instead of a dozen flat ones, which is both closer
    to the drawing and far kinder to every boolean that follows.
    """
    count = len(points)
    segments, start = [], 0
    while start < count:
        best = start + 1
        # A run may end on the starting vertex but never pass it, or the outline
        # would overlap its own beginning instead of closing on it.
        for end in range(start + 3, min(count, start + 63) + 1):
            run = [points[k % count] for k in range(start, end + 1)]
            fit = _circle(run[0], run[len(run) // 2], run[-1])
            if fit is None:
                break
            centre, radius = fit
            if radius > 1e6 * tolerance:
                break
            if any(abs(math.dist(centre, p) - radius) > tolerance for p in run):
                break
            if math.dist(run[0], run[-1]) < 1e-9:
                break
            steps = [
                math.atan2(
                    (p[0] - centre[0]) * (q[1] - centre[1]) - (p[1] - centre[1]) * (q[0] - centre[0]),
                    (p[0] - centre[0]) * (q[0] - centre[0]) + (p[1] - centre[1]) * (q[1] - centre[1]),
                )
                for p, q in itertools.pairwise(run)
            ]
            # Every step has to turn the same way. A thin spike runs out and back, and
            # its vertices can sit near one wide circle while the turning cancels out;
            # drawn as an arc it cuts straight across the part.
            if not (all(step > 0 for step in steps) or all(step < 0 for step in steps)):
                break
            # Under half a circle, where three points pin the arc down.
            if abs(sum(steps)) >= math.pi * 0.95:
                break
            # The vertices sit on the circle, but after simplifying they can be far apart,
            # and between two of them the outline is a straight chord. The arc bows away
            # from each chord by its sagitta, which must stay inside the tolerance too;
            # on a plate eight millimetres of bow took a tenth of the part away.
            if any(radius * (1 - math.cos(abs(step) / 2)) > tolerance for step in steps):
                break
            best = end
        if best - start >= 3:
            run = [points[k % count] for k in range(start, best + 1)]
            segments.append(("arc", run[0], run[len(run) // 2], run[-1]))
        else:
            best = start + 1
            segments.append(("line", points[start % count], points[best % count]))
        start = best
    return segments


def polygon_source(pieces, index: int, target: str, tolerance: float = 1e-3) -> list[str]:
    """Source for a billet drawn as polygons, each extruded across its stretch.

    `pieces` is a list of (start, length, polygons). One statement draws each outline
    and one extrudes it, which is how a designer would sketch and pull it.
    """
    from .model import fmt, fmt_tuple

    x_dir = tuple(1.0 if k == next(j for j in range(3) if j != index) else 0.0 for k in range(3))
    z_dir = tuple(1.0 if k == index else 0.0 for k in range(3))
    code, count = [], 0
    for start, length, polygons in pieces:
        origin = tuple(start if k == index else 0.0 for k in range(3))
        for polygon in polygons:
            corners = [(float(u), float(v)) for u, v in list(polygon.exterior.coords)[:-1]]
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
            code.append(f"_prof = (\n    {joined}\n)")
            code.append(
                f"_plane = Plane(origin={fmt_tuple(origin)}, x_dir={fmt_tuple(x_dir)}, "
                f"z_dir={fmt_tuple(z_dir)})"
            )
            assign = f"{target} =" if count == 0 else f"{target} +="
            code.append(f"{assign} extrude(_plane * make_face(_prof), amount={fmt(length)})")
            count += 1
    return code
