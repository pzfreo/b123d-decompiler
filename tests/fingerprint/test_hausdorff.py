"""Unit tests for the Hausdorff distance implementation.

Pure Python — no build123d or OCC needed. The module is loaded straight from
its file rather than through the package, whose ``__init__`` pulls in OCC, so
this suite runs anywhere. Geometry is hand-built, so the expected distances
are known exactly.
"""

import importlib.util
import math
import os

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "b123d_decompiler", "fingerprint", "hausdorff.py"
)
_spec = importlib.util.spec_from_file_location("_hausdorff_standalone", _PATH)
hausdorff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hausdorff)

TriangleGrid = hausdorff.TriangleGrid
_distance_stats = hausdorff._distance_stats
_point_triangle_distance2 = hausdorff._point_triangle_distance2
cluster_decimate = hausdorff.cluster_decimate
coarse_fold_fraction = hausdorff.coarse_fold_fraction
decode_mesh = hausdorff.decode_mesh
encode_mesh = hausdorff.encode_mesh
estimate_facet_resolution = hausdorff.estimate_facet_resolution
hausdorff_distance = hausdorff.hausdorff_distance
sample_mesh_points = hausdorff.sample_mesh_points


# ── fixture geometry ─────────────────────────────────────────────────

def cube_mesh(size=10.0, offset=(0.0, 0.0, 0.0)):
    """Axis-aligned cube as (vertices, triangles), 12 triangles."""
    ox, oy, oz = offset
    s = size
    v = [
        (ox, oy, oz), (ox + s, oy, oz), (ox + s, oy + s, oz), (ox, oy + s, oz),
        (ox, oy, oz + s), (ox + s, oy, oz + s), (ox + s, oy + s, oz + s),
        (ox, oy + s, oz + s),
    ]
    t = [
        (0, 2, 1), (0, 3, 2),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # -Y
        (2, 3, 7), (2, 7, 6),  # +Y
        (1, 2, 6), (1, 6, 5),  # +X
        (0, 4, 7), (0, 7, 3),  # -X
    ]
    return v, t


UNIT_TRIANGLE = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _planar_mesh(steps=30, size=100.0):
    """Flat sheet of triangles in the z=0 plane — zero bounding-box volume."""
    vertices, triangles = [], []
    for i in range(steps + 1):
        for j in range(steps + 1):
            vertices.append((i * size / steps, j * size / steps, 0.0))
    row = steps + 1
    for i in range(steps):
        for j in range(steps):
            a = i * row + j
            triangles.append((a, a + 1, a + row))
            triangles.append((a + 1, a + row + 1, a + row))
    return vertices, triangles


# ── mesh codec ───────────────────────────────────────────────────────

class TestMeshCodec:
    def test_round_trip_preserves_geometry(self):
        vertices, triangles = cube_mesh(size=25.0, offset=(-3.0, 7.5, 0.25))
        payload = encode_mesh(vertices, triangles)
        out_v, out_t = decode_mesh(payload)

        assert [tuple(t) for t in out_t] == triangles
        for original, decoded in zip(vertices, out_v):
            for a, b in zip(original, decoded):
                # 16-bit quantisation over a 25 mm box → sub-micron error
                assert abs(a - b) < 1e-3

    def test_payload_is_json_safe(self):
        payload = encode_mesh(*cube_mesh())
        import json
        json.loads(json.dumps(payload))
        assert payload["vertex_count"] == 8
        assert payload["triangle_count"] == 12
        assert payload["index_bits"] == 16

    def test_payload_is_compact(self):
        """Encoding must be far smaller than plain JSON numbers."""
        vertices, triangles = cube_mesh()
        vertices = vertices * 100  # 800 vertices
        triangles = triangles * 100
        payload = encode_mesh(vertices, triangles)
        size = len(payload["vertices"]) + len(payload["triangles"])
        assert size < len(str(vertices)) / 4

    def test_empty_mesh(self):
        payload = encode_mesh([], [])
        assert decode_mesh(payload) == ([], [])
        assert decode_mesh({}) == ([], [])


# ── point ↔ triangle distance ────────────────────────────────────────

class TestPointTriangleDistance:
    def _dist(self, p):
        return math.sqrt(_point_triangle_distance2(p, *UNIT_TRIANGLE))

    def test_point_above_face(self):
        assert abs(self._dist((0.25, 0.25, 2.0)) - 2.0) < 1e-9

    def test_point_on_face(self):
        assert self._dist((0.25, 0.25, 0.0)) < 1e-9

    def test_point_beyond_vertex(self):
        assert abs(self._dist((-3.0, 0.0, 0.0)) - 3.0) < 1e-9
        assert abs(self._dist((0.0, -1.0, 0.0)) - 1.0) < 1e-9

    def test_point_beyond_edge(self):
        # Beyond the hypotenuse, in-plane: closest point is (0.5, 0.5, 0)
        d = self._dist((1.0, 1.0, 0.0))
        assert abs(d - math.sqrt(0.5)) < 1e-9

    def test_point_off_corner(self):
        # Diagonally past the origin corner and lifted off the plane
        d = self._dist((-1.0, -1.0, 1.0))
        assert abs(d - math.sqrt(3.0)) < 1e-9

    def test_degenerate_triangle(self):
        p = (0.0, 0.0, 5.0)
        zero = ((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0))
        d = math.sqrt(_point_triangle_distance2(p, *zero))
        assert abs(d - math.sqrt(1 + 1 + 25)) < 1e-9


# ── spatial grid ─────────────────────────────────────────────────────

class TestTriangleGrid:
    def test_matches_brute_force(self):
        vertices, triangles = cube_mesh(size=10.0)
        grid = TriangleGrid(vertices, triangles)
        tris = [(vertices[i], vertices[j], vertices[k]) for i, j, k in triangles]

        probes = [
            (5.0, 5.0, 5.0),      # centre of the cube
            (5.0, 5.0, 12.0),     # above the top face
            (-4.0, 5.0, 5.0),     # outside the -X face
            (0.1, 0.1, 0.1),      # near a corner
            (10.5, 10.5, 10.5),   # diagonally outside a corner
            (3.0, 7.0, 0.0),      # on a face
        ]
        for p in probes:
            expected = math.sqrt(min(
                _point_triangle_distance2(p, *t) for t in tris
            ))
            assert abs(grid.nearest_distance(p) - expected) < 1e-9, p

    def test_distance_from_inside_cube_centre(self):
        vertices, triangles = cube_mesh(size=10.0)
        grid = TriangleGrid(vertices, triangles)
        # Centre of a 10 mm cube is 5 mm from every wall
        assert abs(grid.nearest_distance((5.0, 5.0, 5.0)) - 5.0) < 1e-9

    def test_empty_grid(self):
        grid = TriangleGrid([], [])
        assert grid.nearest_distance((0.0, 0.0, 0.0)) == float("inf")

    def test_flat_mesh_does_not_explode_the_grid(self):
        """A zero-thickness mesh has no volume to size cells from.

        Sizing cells by bounding-box volume made every triangle land in
        thousands of cells; an open or planar STL would exhaust memory.
        """
        vertices, triangles = _planar_mesh(steps=30, size=100.0)
        grid = TriangleGrid(vertices, triangles)
        entries = sum(len(ids) for ids in grid.cells.values())
        assert entries < 10 * len(triangles), (
            f"{entries} grid entries for {len(triangles)} triangles"
        )

    def test_flat_mesh_still_answers_exactly(self):
        vertices, triangles = _planar_mesh(steps=12, size=100.0)
        grid = TriangleGrid(vertices, triangles)
        tris = [(vertices[i], vertices[j], vertices[k]) for i, j, k in triangles]
        probes = [
            (50.0, 50.0, 7.0),     # above the sheet
            (-30.0, 50.0, 0.0),    # off one edge, in plane
            (120.0, 120.0, 5.0),   # past a corner
            (10.0, 90.0, -3.0),    # below the sheet
        ]
        for p in probes:
            expected = math.sqrt(min(
                _point_triangle_distance2(p, *t) for t in tris
            ))
            assert abs(grid.nearest_distance(p) - expected) < 1e-9, p


# ── surface sampling ─────────────────────────────────────────────────

class TestSampleMeshPoints:
    def test_count_and_determinism(self):
        vertices, triangles = cube_mesh()
        first = sample_mesh_points(vertices, triangles, 500)
        second = sample_mesh_points(vertices, triangles, 500)
        assert len(first) == 500
        assert first == second, "sampling must be reproducible"

    def test_points_lie_on_the_surface(self):
        vertices, triangles = cube_mesh()
        grid = TriangleGrid(vertices, triangles)
        for p in sample_mesh_points(vertices, triangles, 200):
            assert grid.nearest_distance(p) < 1e-9

    def test_points_cover_every_face(self):
        """Area-weighted sampling must reach all six faces of a cube."""
        vertices, triangles = cube_mesh(size=10.0)
        points = sample_mesh_points(vertices, triangles, 600)
        on_face = set()
        for x, y, z in points:
            for axis, value in enumerate((x, y, z)):
                if abs(value) < 1e-6:
                    on_face.add((axis, "min"))
                if abs(value - 10.0) < 1e-6:
                    on_face.add((axis, "max"))
        assert len(on_face) == 6, f"only reached {sorted(on_face)}"

    def test_empty_inputs(self):
        assert sample_mesh_points([], [], 100) == []
        assert sample_mesh_points(*cube_mesh(), 0) == []


# ── distance statistics ──────────────────────────────────────────────

class TestDistanceStats:
    def test_known_values(self):
        stats = _distance_stats([3.0, 1.0, 2.0, 4.0])
        assert stats["max"] == 4.0
        assert abs(stats["mean"] - 2.5) < 1e-12
        assert abs(stats["rms"] - math.sqrt(30 / 4)) < 1e-12

    def test_empty(self):
        assert _distance_stats([]) == {"max": 0.0, "mean": 0.0,
                                       "rms": 0.0, "p95": 0.0}


# ── end-to-end Hausdorff ─────────────────────────────────────────────

class TestHausdorffDistance:
    def test_identical_meshes(self):
        mesh = cube_mesh()
        result = hausdorff_distance(mesh, mesh, samples=400)
        assert result["hausdorff"] < 1e-9
        assert result["mean"] < 1e-9

    def test_translated_cube(self):
        """A rigid 0.5 mm shift shows up as a 0.5 mm deviation."""
        a = cube_mesh(size=10.0)
        b = cube_mesh(size=10.0, offset=(0.5, 0.0, 0.0))
        result = hausdorff_distance(a, b, samples=1000)
        assert abs(result["hausdorff"] - 0.5) < 1e-6
        # Most of the surface is only 0 – 0.5 mm out, so the mean is lower
        assert 0.0 < result["mean"] < 0.5

    def test_uniformly_grown_cube(self):
        """A 1 mm offset shell reads as 1 mm of deviation from the inside out.

        The reverse direction is larger: the outer cube's corners are
        sqrt(3) mm from the inner cube, which is what a Hausdorff distance
        is supposed to report.
        """
        inner = cube_mesh(size=10.0)
        outer = cube_mesh(size=12.0, offset=(-1.0, -1.0, -1.0))
        result = hausdorff_distance(inner, outer, samples=1000)
        assert abs(result["forward"]["max"] - 1.0) < 1e-6
        assert 1.0 < result["backward"]["max"] <= math.sqrt(3.0) + 1e-6
        assert result["hausdorff"] == result["backward"]["max"]

    def test_local_bump_moves_max_but_not_mean(self):
        """A dent in one face dominates the max while barely moving the mean."""
        vertices, triangles = cube_mesh(size=10.0)
        dented = list(vertices)
        dented[6] = (11.0, 11.0, 11.0)  # pull one corner 1 mm out per axis
        result = hausdorff_distance(
            (vertices, triangles), (dented, triangles), samples=1000
        )
        assert result["hausdorff"] > 0.5
        assert result["mean"] < result["hausdorff"] / 4

    def test_survives_encode_decode(self):
        """Quantisation must not change the measured distance meaningfully."""
        a = cube_mesh(size=10.0)
        b = cube_mesh(size=10.0, offset=(0.25, 0.0, 0.0))
        direct = hausdorff_distance(a, b, samples=500)
        round_tripped = hausdorff_distance(
            decode_mesh(encode_mesh(*a)), decode_mesh(encode_mesh(*b)),
            samples=500,
        )
        assert abs(direct["hausdorff"] - round_tripped["hausdorff"]) < 1e-3

    def test_reports_both_directions(self):
        result = hausdorff_distance(cube_mesh(), cube_mesh(), samples=100)
        for key in ("forward", "backward"):
            assert set(result[key]) == {"max", "mean", "rms", "p95"}
        assert result["samples"] == 100


# ── facet resolution estimate ────────────────────────────────────────

class TestFacetResolution:
    def test_flat_mesh_has_no_chord_error(self):
        """A cube's facets are exact; its 90° edges are features, not error."""

        assert estimate_facet_resolution(*cube_mesh(size=10.0)) == 0.0

    def test_faceted_cylinder_error_tracks_the_facet_size(self):
        """Chord error of an n-gon prism approximating a cylinder: r(1-cos(pi/n))."""

        def prism(sides, radius=10.0, height=4.0):
            vertices, triangles = [], []
            for i in range(sides):
                angle = 2 * math.pi * i / sides
                x, y = radius * math.cos(angle), radius * math.sin(angle)
                vertices.append((x, y, 0.0))
                vertices.append((x, y, height))
            for i in range(sides):
                a, b = 2 * i, 2 * i + 1
                c, d = (2 * (i + 1)) % (2 * sides), (2 * (i + 1) + 1) % (2 * sides)
                triangles.append((a, c, b))
                triangles.append((b, c, d))
            return vertices, triangles

        for sides in (12, 24, 48):
            true_error = 10.0 * (1 - math.cos(math.pi / sides))
            estimate = estimate_facet_resolution(*prism(sides))
            assert estimate > 0
            # Within 2x of the true sagitta, and never wildly optimistic
            assert true_error / 2 < estimate < true_error * 3, (
                f"{sides} sides: estimate {estimate:.5f} vs true {true_error:.5f}"
            )

    def test_finer_facets_estimate_a_smaller_error(self):

        def dome(steps, radius=10.0):
            vertices, triangles = [], []
            for i in range(steps + 1):
                theta = math.pi / 2 * i / steps
                for j in range(steps + 1):
                    phi = math.pi / 2 * j / steps
                    vertices.append((
                        radius * math.sin(theta) * math.cos(phi),
                        radius * math.sin(theta) * math.sin(phi),
                        radius * math.cos(theta),
                    ))
            row = steps + 1
            for i in range(steps):
                for j in range(steps):
                    a = i * row + j
                    triangles.append((a, a + 1, a + row))
                    triangles.append((a + 1, a + row + 1, a + row))
            return vertices, triangles

        coarse = estimate_facet_resolution(*dome(6))
        fine = estimate_facet_resolution(*dome(24))
        assert fine < coarse / 3

    def test_empty_mesh(self):

        assert estimate_facet_resolution([], []) == 0.0


class TestClusterDecimate:
    def test_reduces_a_dense_mesh(self):

        vertices, triangles = cube_mesh(size=10.0)
        # Subdivide-free stand-in: many cubes' worth of coincident geometry
        v2, t2 = cluster_decimate(vertices, triangles, cell=20.0)
        assert len(v2) < len(vertices)
        assert len(t2) <= len(triangles)

    def test_moves_vertices_by_less_than_the_cell(self):

        vertices, triangles = cube_mesh(size=10.0)
        cell = 1.0
        v2, t2 = cluster_decimate(vertices, triangles, cell)
        grid = TriangleGrid(v2, t2)
        for p in vertices:
            assert grid.nearest_distance(p) < cell * math.sqrt(3)

    def test_zero_cell_is_a_no_op(self):

        vertices, triangles = cube_mesh()
        assert cluster_decimate(vertices, triangles, 0.0) == (vertices, triangles)


class TestSampleCountValidation:
    def test_zero_samples_is_rejected(self):
        """Zero samples would grade any pair of shapes as a perfect match."""
        import pytest

        with pytest.raises(ValueError, match="at least 1"):
            hausdorff_distance(cube_mesh(10.0), cube_mesh(20.0), samples=0)

    def test_negative_samples_is_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            hausdorff_distance(cube_mesh(), cube_mesh(), samples=-5)


class TestCoarseFoldFraction:
    """Signals when facet folds are too sharp to read as chord error."""

    def test_flat_mesh_has_no_sharp_interior_folds(self):

        assert coarse_fold_fraction(*_planar_mesh(steps=8)) == 0.0

    def test_cube_is_all_sharp_folds(self):

        # Every non-diagonal edge of a cube turns by 90°
        assert coarse_fold_fraction(*cube_mesh()) > 0.5

    def test_empty_mesh(self):

        assert coarse_fold_fraction([], []) == 0.0


class TestFacetResolutionAccuracy:
    """The estimate is checked against chord error that is known exactly."""

    @staticmethod
    def _prism(sides, radius=10.0, height=20.0):
        """n-gon prism: chord error against the cylinder is r(1 - cos(pi/n))."""
        vertices, triangles = [], []
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            x, y = radius * math.cos(angle), radius * math.sin(angle)
            vertices += [(x, y, 0.0), (x, y, height)]
        for i in range(sides):
            a, b = 2 * i, 2 * i + 1
            c = (2 * (i + 1)) % (2 * sides)
            d = (2 * (i + 1) + 1) % (2 * sides)
            triangles += [(a, c, b), (b, c, d)]
        bottom, top = len(vertices), len(vertices) + 1
        vertices += [(0.0, 0.0, 0.0), (0.0, 0.0, height)]
        for i in range(sides):
            triangles.append((bottom, 2 * ((i + 1) % sides), 2 * i))
            triangles.append((top, 2 * i + 1, 2 * ((i + 1) % sides) + 1))
        return vertices, triangles

    def test_matches_analytic_chord_error(self):

        for sides in (8, 12, 24, 48, 96):
            estimate = estimate_facet_resolution(*self._prism(sides))
            exact = 10.0 * (1 - math.cos(math.pi / sides))
            assert abs(estimate - exact) / exact < 0.1, (
                f"{sides} sides: estimate {estimate:.5f} vs exact {exact:.5f}"
            )

    def test_chamfer_band_contributes_only_its_own_width(self):
        """A narrow chamfer between broad faces is a feature, not chord error.

        Measuring the span across the *broad* face would report a chord
        error the size of the whole part.
        """

        # Two 50 mm faces meeting through a 0.5 mm chamfer band at 30°
        vertices = [
            (0.0, 0.0, 0.0), (0.0, 50.0, 0.0),
            (50.0, 0.0, 0.0), (50.0, 50.0, 0.0),
            (50.433, 0.0, 0.25), (50.433, 50.0, 0.25),
            (100.0, 0.0, 0.25), (100.0, 50.0, 0.25),
        ]
        triangles = [
            (0, 2, 1), (1, 2, 3),        # broad face
            (2, 4, 3), (3, 4, 5),        # chamfer band
            (4, 6, 5), (5, 6, 7),        # broad face
        ]
        estimate = estimate_facet_resolution(vertices, triangles)
        assert estimate < 0.1, (
            f"chamfer read as {estimate:.4f} mm of chord error"
        )
