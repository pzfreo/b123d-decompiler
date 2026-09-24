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
        return f"{self.kind} along {'xyz'[self.index]}"


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


def choose(candidates: list[Outline], low, high, gain: float, most_extrusions: int,
           points: int = 40000) -> list[Outline]:
    """The combination of outlines a designer would most plausibly have started from.

    Each candidate is tried as the first outline, and others are intersected with it
    in order of size while each takes away at least `1 - gain` of what is left and the
    whole stays within `most_extrusions`. Of the combinations that gives, the one kept
    has the least volume once every extrusion is charged EXTRUSION_COST of it: a
    billet one extrusion longer has to be that much tighter to be the better reading.
    """
    low, high = np.asarray(low), np.asarray(high)
    places = np.random.default_rng(20260924).uniform(low, high, size=(points, 3))
    usable = [c for c in candidates if c.extrusions <= most_extrusions]
    if not usable:
        return []
    inside = {id(c): c.contains(places) for c in usable}
    usable.sort(key=lambda c: inside[id(c)].sum())

    best, best_score = None, math.inf
    for first in usable:
        chosen, current, extrusions = [first], inside[id(first)].copy(), first.extrusions
        for candidate in usable:
            if candidate is first or extrusions + candidate.extrusions > most_extrusions:
                continue
            merged = current & inside[id(candidate)]
            if merged.sum() > current.sum() * gain:
                continue
            chosen.append(candidate)
            current, extrusions = merged, extrusions + candidate.extrusions
        score = current.sum() * (1 + EXTRUSION_COST * extrusions)
        if score < best_score:
            best, best_score = chosen, score
    return best or []


def estimate_volume(outline_set: list[Outline], low, high, points: int = 40000) -> float:
    low, high = np.asarray(low), np.asarray(high)
    places = np.random.default_rng(20260924).uniform(low, high, size=(points, 3))
    inside = np.ones(points, dtype=bool)
    for outline in outline_set:
        inside &= outline.contains(places)
    return float(inside.sum()) / points * math.prod(high - low)
