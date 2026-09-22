"""Approximate Hausdorff distance between triangle meshes.

The rest of the fingerprint measures *aggregate* quantities (volume, area,
inertia, section areas). Those can all agree while the surface is locally
wrong — a fillet in the wrong place, a boss shifted 1 mm, a chamfer that
became a round. The Hausdorff distance is the complementary measure: the
worst-case point-to-surface deviation anywhere on the part.

Distances are computed between triangle meshes, both directions:

    forward   = max over points sampled on A of distance(point, surface B)
    backward  = max over points sampled on B of distance(point, surface A)
    hausdorff = max(forward, backward)

Sampling is deterministic (area-weighted stratification + a low-discrepancy
sequence, no RNG), so the same mesh always yields the same points and the
same numbers.

Everything below the mesh codec is dependency-free pure Python: the test
generator embeds these functions verbatim (via ``inspect.getsource``) into
the generated pytest file, so there is a single implementation shared by
``compare`` and by generated tests.
"""

from __future__ import annotations

import base64
import bisect
import math
import struct
import zlib

# ── mesh codec ───────────────────────────────────────────────────────
#
# Meshes are stored (in JSON fingerprints and in generated test files) as
# zlib-compressed, base64-encoded binary blobs: 16-bit quantised vertex
# coordinates and 16- or 32-bit triangle indices. A 7000-triangle mesh costs
# ~40 KB of text this way, against well over 1 MB as indented JSON numbers.


def encode_mesh(vertices: list, triangles: list) -> dict:
    """Encode (vertices, triangles) as a compact, JSON-safe dict.

    Coordinates are quantised to 16 bits per axis across the mesh bounding
    box (sub-micron for any part that fits on a desk), indices use the
    narrowest integer type that fits, and both arrays are zlib-compressed
    and base64-encoded.
    """
    if not vertices:
        return {
            "encoding": "q16", "vertex_count": 0, "triangle_count": 0,
            "bbox_min": (0.0, 0.0, 0.0), "bbox_max": (0.0, 0.0, 0.0),
            "index_bits": 16, "vertices": "", "triangles": "",
        }

    lo = tuple(min(v[a] for v in vertices) for a in range(3))
    hi = tuple(max(v[a] for v in vertices) for a in range(3))
    scale = tuple((hi[a] - lo[a]) / 65535.0 if hi[a] > lo[a] else 0.0
                  for a in range(3))

    quantised = []
    for v in vertices:
        for a in range(3):
            q = int(round((v[a] - lo[a]) / scale[a])) if scale[a] else 0
            quantised.append(min(max(q, 0), 65535))
    vbuf = struct.pack(f"<{len(quantised)}H", *quantised)

    index_bits = 16 if len(vertices) <= 65536 else 32
    code = "H" if index_bits == 16 else "I"
    flat = [i for t in triangles for i in t]
    tbuf = struct.pack(f"<{len(flat)}{code}", *flat)

    return {
        "encoding": "q16",
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "bbox_min": tuple(round(c, 6) for c in lo),
        "bbox_max": tuple(round(c, 6) for c in hi),
        "index_bits": index_bits,
        "vertices": base64.b64encode(zlib.compress(vbuf, 9)).decode("ascii"),
        "triangles": base64.b64encode(zlib.compress(tbuf, 9)).decode("ascii"),
    }


def decode_mesh(payload: dict) -> tuple[list, list]:
    """Decode an :func:`encode_mesh` payload back into (vertices, triangles)."""
    if payload and payload.get("truncated"):
        raise ValueError(
            "this fingerprint's surface mesh was truncated for display; "
            "re-run the analysis with --json to write the full mesh"
        )
    if not payload or not payload.get("vertex_count"):
        return [], []
    lo = payload["bbox_min"]
    hi = payload["bbox_max"]
    scale = [(hi[a] - lo[a]) / 65535.0 if hi[a] > lo[a] else 0.0
             for a in range(3)]

    vbuf = zlib.decompress(base64.b64decode(payload["vertices"]))
    raw = struct.unpack(f"<{len(vbuf) // 2}H", vbuf)
    vertices = [
        (lo[0] + raw[i] * scale[0],
         lo[1] + raw[i + 1] * scale[1],
         lo[2] + raw[i + 2] * scale[2])
        for i in range(0, len(raw), 3)
    ]

    tbuf = zlib.decompress(base64.b64decode(payload["triangles"]))
    if payload.get("index_bits", 16) == 16:
        flat = struct.unpack(f"<{len(tbuf) // 2}H", tbuf)
    else:
        flat = struct.unpack(f"<{len(tbuf) // 4}I", tbuf)
    triangles = [flat[i:i + 3] for i in range(0, len(flat), 3)]
    return vertices, triangles



def cluster_decimate(vertices, triangles, cell: float) -> tuple[list, list]:
    """Reduce a mesh by snapping vertices to a grid of size ``cell``.

    Used for STL references that are too dense to embed in a test file: the
    facets cannot be re-meshed (there is no analytical surface behind them),
    so vertices are clustered instead. Every vertex moves by at most
    ``cell * sqrt(3) / 2`` and triangles that collapse to a line or a point
    are dropped, so the decimated surface stays within roughly ``cell`` of
    the original — which is why callers fold ``cell`` into the mesh
    resolution they report.
    """
    if cell <= 0.0 or not triangles:
        return vertices, triangles

    index_of = {}
    new_vertices = []
    remap = []
    for x, y, z in vertices:
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)),
               int(math.floor(z / cell)))
        idx = index_of.get(key)
        if idx is None:
            idx = len(new_vertices)
            index_of[key] = idx
            new_vertices.append((
                (key[0] + 0.5) * cell, (key[1] + 0.5) * cell,
                (key[2] + 0.5) * cell,
            ))
        remap.append(idx)

    new_triangles = []
    seen = set()
    for i, j, k in triangles:
        a, b, c = remap[i], remap[j], remap[k]
        if a == b or b == c or a == c:
            continue  # collapsed into a line or a point
        key = tuple(sorted((a, b, c)))
        if key in seen:
            continue
        seen.add(key)
        new_triangles.append((a, b, c))
    return new_vertices, new_triangles


def estimate_facet_resolution(
    vertices, triangles, feature_angle_limit: float = 0.7854,
) -> float:
    """Estimate how far a triangulation sits from the smooth surface behind it.

    Used for STL, where there is no analytical surface to measure against —
    the chord error has to be read off the facets themselves. Across a smooth
    interior edge the two facets turn by an angle ``theta`` over a span ``h``,
    approximating an arc of radius ``h / theta`` whose chord sits
    ``h * theta / 8`` inside it. ``h`` is the *smaller* of the two facets'
    extents perpendicular to the shared edge, so a narrow chamfer band
    between two broad faces contributes only its own width.

    Folds sharper than ``feature_angle_limit`` are read as real features, not
    as chord error, and ignored. That is a genuine limit rather than a
    conservative choice: a hexagonal prism and a six-facet cylinder are the
    same mesh, and nothing in the file says which was meant. A mesh coarse
    enough for that ambiguity to matter is reported by
    :func:`coarse_fold_fraction`, and the caller can state the export
    tolerance outright instead of estimating it.

    Returns the worst qualifying edge, which is the right pairing for a
    worst-case Hausdorff tolerance.
    """
    worst = 0.0
    for span, theta in _interior_folds(vertices, triangles):
        if theta <= feature_angle_limit:
            worst = max(worst, span * theta / 8.0)
    return worst


def coarse_fold_fraction(
    vertices, triangles, feature_angle_limit: float = 0.7854,
) -> float:
    """Fraction of interior edges folding too sharply to read as chord error.

    High on a coarsely faceted mesh — and on a genuinely faceted part, which
    is the ambiguity — so it is a reason to ask the user for the export
    tolerance, not a measurement in itself.
    """
    folds = list(_interior_folds(vertices, triangles))
    if not folds:
        return 0.0
    sharp = sum(1 for _, theta in folds if theta > feature_angle_limit)
    return sharp / len(folds)


def _interior_folds(vertices, triangles):
    """Yield (span, turn angle) for every manifold interior edge."""
    if not triangles:
        return

    normals = []
    for i, j, k in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        normals.append(
            (nx / length, ny / length, nz / length) if length > 1e-15 else None
        )

    # Keyed on position, not index: STL facets rarely share vertex indices.
    def key(index):
        x, y, z = vertices[index]
        return (round(x, 6), round(y, 6), round(z, 6))

    edges: dict = {}
    for t, (i, j, k) in enumerate(triangles):
        for a, b, opposite in ((i, j, k), (j, k, i), (k, i, j)):
            ka, kb = key(a), key(b)
            edge = (ka, kb) if ka <= kb else (kb, ka)
            edges.setdefault(edge, []).append((t, key(opposite)))

    for (ka, kb), facets in edges.items():
        if len(facets) != 2:
            continue  # boundary, or non-manifold — no dihedral to read
        (t1, p1), (t2, p2) = facets
        n1, n2 = normals[t1], normals[t2]
        if n1 is None or n2 is None:
            continue
        dot = min(max(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2], -1.0), 1.0)
        span = min(
            _distance_to_line(p1, ka, kb), _distance_to_line(p2, ka, kb)
        )
        yield span, math.acos(dot)


def _distance_to_line(point, a, b) -> float:
    """Perpendicular distance from a point to the line through a and b."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    length2 = dx * dx + dy * dy + dz * dz
    if length2 < 1e-24:
        return 0.0
    px, py, pz = point[0] - a[0], point[1] - a[1], point[2] - a[2]
    cx = py * dz - pz * dy
    cy = pz * dx - px * dz
    cz = px * dy - py * dx
    return math.sqrt((cx * cx + cy * cy + cz * cz) / length2)


def mesh_resolution(surface_mesh: dict) -> float:
    """Combined meshing error in mm for a stored reference mesh, in mm.

    Two terms, because two surfaces are involved: how far the stored mesh
    sits from the reference surface, plus how far the part under test will
    sit from its own surface when meshed at the recorded deflection.
    """
    if not surface_mesh:
        return 0.0
    return (
        surface_mesh.get("resolution", 0.0)
        + surface_mesh.get(
            "candidate_resolution", surface_mesh.get("deflection", 0.0)
        )
    )


def tolerance_floor(
    surface_mesh: dict, candidate_deflection: float | None = None,
) -> float:
    """Smallest deviation a mesh of this resolution can meaningfully resolve.

    ``candidate_deflection`` is the deflection the part being compared was
    actually meshed at, which may be coarser than the reference's if it had
    more detail to fit into the same triangle budget. Chord error scales with
    deflection, so the reference's own figures scale to cover it — reading
    the error off the candidate instead would let the part being judged widen
    its own tolerance.

    Twice the combined meshing error. For a STEP reference that error is
    measured rather than estimated, and across spheres, cylinders, cones and
    tori at deflections from 0.02 mm to 5 mm the deviation between two
    triangulations of one surface stayed at ~0.35x the combined figure; the
    margin covers the STL path, where the error can only be estimated from
    the facets and can run about 1.4x under the truth.
    """
    reference = surface_mesh.get("resolution", 0.0)
    candidate = surface_mesh.get(
        "candidate_resolution", surface_mesh.get("deflection", 0.0)
    )
    recorded = surface_mesh.get("deflection", 0.0)
    if candidate_deflection and recorded > 0:
        candidate *= candidate_deflection / recorded
    return 2.0 * (reference + candidate)


# ── point ↔ triangle distance ────────────────────────────────────────


def _point_triangle_distance2(p, a, b, c) -> float:
    """Squared distance from point p to triangle (a, b, c).

    Voronoi-region algorithm (Ericson, *Real-Time Collision Detection*):
    classify p against the triangle's vertex/edge/face regions, then
    project onto whichever feature is closest.
    """
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    apx, apy, apz = p[0] - a[0], p[1] - a[1], p[2] - a[2]

    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz

    bpx, bpy, bpz = p[0] - b[0], p[1] - b[1], p[2] - b[2]
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        v = d1 / denom if denom != 0.0 else 0.0
        dx, dy, dz = apx - v * abx, apy - v * aby, apz - v * abz
        return dx * dx + dy * dy + dz * dz

    cpx, cpy, cpz = p[0] - c[0], p[1] - c[1], p[2] - c[2]
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        w = d2 / denom if denom != 0.0 else 0.0
        dx, dy, dz = apx - w * acx, apy - w * acy, apz - w * acz
        return dx * dx + dy * dy + dz * dz

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if denom != 0.0 else 0.0
        dx = p[0] - (b[0] + w * (c[0] - b[0]))
        dy = p[1] - (b[1] + w * (c[1] - b[1]))
        dz = p[2] - (b[2] + w * (c[2] - b[2]))
        return dx * dx + dy * dy + dz * dz

    denom = va + vb + vc
    if denom == 0.0:  # degenerate triangle — fall back to vertex a
        return apx * apx + apy * apy + apz * apz
    v = vb / denom
    w = vc / denom
    dx = apx - (v * abx + w * acx)
    dy = apy - (v * aby + w * acy)
    dz = apz - (v * abz + w * acz)
    return dx * dx + dy * dy + dz * dz


def _mesh_area(tris) -> float:
    """Total area of a list of (a, b, c) vertex triples."""
    total = 0.0
    for a, b, c in tris:
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        total += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
    return total


class TriangleGrid:
    """Uniform spatial grid over a triangle mesh for nearest-surface queries.

    Each triangle is registered in every cell its bounding box touches.
    A query walks outward in cubic shells around the query point's cell and
    stops once the best distance found is closer than any unvisited shell
    could possibly be.
    """

    def __init__(self, vertices, triangles, target_per_cell: float = 3.0):
        self.tris = [
            (vertices[i], vertices[j], vertices[k]) for i, j, k in triangles
        ]
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        if not xs:
            self.cells = {}
            self.cell = 1.0
            self.dims = (1, 1, 1)
            self.origin = (0.0, 0.0, 0.0)
            return
        pad = 1e-6
        self.origin = (min(xs) - pad, min(ys) - pad, min(zs) - pad)
        span = (
            max(xs) - min(xs) + 2 * pad,
            max(ys) - min(ys) + 2 * pad,
            max(zs) - min(zs) + 2 * pad,
        )
        # Size the cell from surface area, not bounding-box volume: the
        # triangles form a 2D sheet inside the box, and a flat part has no
        # volume to divide by at all. Occupied cells go as area / cell², so
        # cell = sqrt(target_per_cell * area / triangle count) puts roughly
        # target_per_cell triangles in each.
        area = _mesh_area(self.tris)
        count = max(len(self.tris), 1)
        if area > 0.0:
            self.cell = math.sqrt(target_per_cell * area / count)
        else:
            self.cell = max(span) / 8.0
        # Never so fine that one triangle spans a huge number of cells.
        self.cell = max(self.cell, max(span) / 512.0, 1e-9)
        self.dims = tuple(max(int(s / self.cell) + 1, 1) for s in span)
        self.max_ring = max(self.dims)

        cells = {}
        ox, oy, oz = self.origin
        c = self.cell
        for idx, (a, b, cc) in enumerate(self.tris):
            i0 = int((min(a[0], b[0], cc[0]) - ox) / c)
            i1 = int((max(a[0], b[0], cc[0]) - ox) / c)
            j0 = int((min(a[1], b[1], cc[1]) - oy) / c)
            j1 = int((max(a[1], b[1], cc[1]) - oy) / c)
            k0 = int((min(a[2], b[2], cc[2]) - oz) / c)
            k1 = int((max(a[2], b[2], cc[2]) - oz) / c)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    for k in range(k0, k1 + 1):
                        cells.setdefault((i, j, k), []).append(idx)
        self.cells = cells

    def nearest_distance(self, p) -> float:
        """Distance from p to the closest point on the mesh surface."""
        if not self.tris:
            return float("inf")
        c = self.cell
        ox, oy, oz = self.origin
        ci = int((p[0] - ox) / c)
        cj = int((p[1] - oy) / c)
        ck = int((p[2] - oz) / c)
        inside = (
            0 <= ci < self.dims[0]
            and 0 <= cj < self.dims[1]
            and 0 <= ck < self.dims[2]
        )
        if not inside:
            # Outside the mesh's grid: the shell bound below assumes p sits
            # in its own cell, so fall back to an exhaustive scan. Only
            # happens for points well clear of the reference surface.
            best2 = float("inf")
            for tri in self.tris:
                d2 = _point_triangle_distance2(p, tri[0], tri[1], tri[2])
                if d2 < best2:
                    best2 = d2
            return math.sqrt(best2)

        best2 = float("inf")
        checked = set()
        cells = self.cells
        tris = self.tris
        ring = 0
        while ring <= self.max_ring:
            for i in range(ci - ring, ci + ring + 1):
                for j in range(cj - ring, cj + ring + 1):
                    for k in range(ck - ring, ck + ring + 1):
                        # only the shell surface, inner cells already done
                        if ring and (
                            abs(i - ci) != ring
                            and abs(j - cj) != ring
                            and abs(k - ck) != ring
                        ):
                            continue
                        for idx in cells.get((i, j, k), ()):
                            if idx in checked:
                                continue
                            checked.add(idx)
                            tri = tris[idx]
                            d2 = _point_triangle_distance2(
                                p, tri[0], tri[1], tri[2]
                            )
                            if d2 < best2:
                                best2 = d2
            # A triangle first appearing at shell ring+1 is at least
            # ring*cell away from p, so this bound is safe.
            if best2 <= (ring * c) ** 2:
                break
            ring += 1
        return math.sqrt(best2)


# ── deterministic surface sampling ───────────────────────────────────


def _radical_inverse(n: int, base: int) -> float:
    """Van der Corput radical inverse — a deterministic low-discrepancy value."""
    result = 0.0
    inv = 1.0 / base
    f = inv
    while n > 0:
        result += (n % base) * f
        n //= base
        f *= inv
    return result


def sample_mesh_points(vertices, triangles, count: int) -> list:
    """Sample ``count`` points spread over the mesh surface, area-weighted.

    Deterministic: triangles are picked by stratifying the cumulative-area
    axis and barycentric coordinates come from a Halton sequence, so no
    random number generator is involved and results are reproducible across
    machines and Python versions.
    """
    if not triangles or count <= 0:
        return []
    cumulative = []
    total = 0.0
    for i, j, k in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        total += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)
        cumulative.append(total)
    if total <= 0.0:
        return []

    points = []
    for n in range(count):
        target = (n + 0.5) / count * total
        t_idx = min(bisect.bisect_left(cumulative, target), len(triangles) - 1)
        i, j, k = triangles[t_idx]
        a, b, c = vertices[i], vertices[j], vertices[k]
        r1 = _radical_inverse(n + 1, 2)
        r2 = _radical_inverse(n + 1, 3)
        s = math.sqrt(r1)
        w0 = 1.0 - s
        w1 = s * (1.0 - r2)
        w2 = s * r2
        points.append((
            a[0] * w0 + b[0] * w1 + c[0] * w2,
            a[1] * w0 + b[1] * w1 + c[1] * w2,
            a[2] * w0 + b[2] * w1 + c[2] * w2,
        ))
    return points


def _distance_stats(distances: list) -> dict:
    """max / mean / rms / 95th percentile of a distance list."""
    if not distances:
        return {"max": 0.0, "mean": 0.0, "rms": 0.0, "p95": 0.0}
    n = len(distances)
    ordered = sorted(distances)
    return {
        "max": ordered[-1],
        "mean": sum(ordered) / n,
        "rms": math.sqrt(sum(d * d for d in ordered) / n),
        "p95": ordered[min(int(0.95 * n), n - 1)],
    }


def hausdorff_distance(mesh_a, mesh_b, samples: int = 2000) -> dict:
    """Two-sided approximate Hausdorff distance between two meshes.

    Args:
        mesh_a: (vertices, triangles) of the reference mesh.
        mesh_b: (vertices, triangles) of the mesh being compared.
        samples: points sampled per direction; must be positive.

    Returns a dict with ``forward`` (A→B), ``backward`` (B→A) and combined
    ``hausdorff`` / ``mean`` / ``rms`` / ``p95`` values, all in model units.
    """
    if samples < 1:
        # Zero samples would report a flawless match for any pair of shapes.
        raise ValueError(f"samples must be at least 1, got {samples}")

    va, ta = mesh_a
    vb, tb = mesh_b
    grid_a = TriangleGrid(va, ta)
    grid_b = TriangleGrid(vb, tb)

    pts_a = sample_mesh_points(va, ta, samples)
    pts_b = sample_mesh_points(vb, tb, samples)
    fwd = [grid_b.nearest_distance(p) for p in pts_a]
    bwd = [grid_a.nearest_distance(p) for p in pts_b]

    f_stats = _distance_stats(fwd)
    b_stats = _distance_stats(bwd)
    both = _distance_stats(fwd + bwd)
    return {
        "samples": samples,
        "forward": f_stats,
        "backward": b_stats,
        "hausdorff": max(f_stats["max"], b_stats["max"]),
        "mean": both["mean"],
        "rms": both["rms"],
        "p95": both["p95"],
    }
