"""Unit tests for pure-Python mesh math in analyze.py.

These tests require no build123d or OCC — they test _compute_mesh_props
and _segments_to_area directly with known geometry.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from b123d_decompiler.fingerprint.analyze import _compute_mesh_props, _segments_to_area


# ── fixture data ─────────────────────────────────────────────────────

# Right-angle tetrahedron: O=(0,0,0), A=(1,0,0), B=(0,1,0), C=(0,0,1)
# Faces wound with outward normals.
TETRAHEDRON = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # face ABC, normal (+,+,+)
    ((0, 0, 0), (1, 0, 0), (0, 0, 1)),  # face OAC, normal (0,-1,0)
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),  # face OCB, normal (-1,0,0)
    ((0, 0, 0), (0, 1, 0), (1, 0, 0)),  # face OBA, normal (0,0,-1)
]

# Unit cube [0,1]³ — 12 triangles, one per half-face.
UNIT_CUBE = [
    # +X (x=1)
    ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((1, 0, 0), (1, 1, 1), (1, 0, 1)),
    # -X (x=0)
    ((0, 0, 0), (0, 0, 1), (0, 1, 1)),
    ((0, 0, 0), (0, 1, 1), (0, 1, 0)),
    # +Y (y=1)
    ((0, 1, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 1, 0), (1, 1, 1), (1, 1, 0)),
    # -Y (y=0)
    ((0, 0, 0), (1, 0, 0), (1, 0, 1)),
    ((0, 0, 0), (1, 0, 1), (0, 0, 1)),
    # +Z (z=1)
    ((0, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 1), (1, 1, 1), (0, 1, 1)),
    # -Z (z=0)
    ((0, 0, 0), (0, 1, 0), (1, 1, 0)),
    ((0, 0, 0), (1, 1, 0), (1, 0, 0)),
]


# ── _compute_mesh_props ──────────────────────────────────────────────

class TestComputeMeshPropsTetrahedron:
    def setup_method(self):
        self.va, self.moi = _compute_mesh_props(TETRAHEDRON)

    def test_volume(self):
        assert abs(self.va["volume"] - 1 / 6) < 1e-10

    def test_surface_area(self):
        # 3 right-triangle faces (area 0.5 each) + equilateral hypotenuse (√3/2)
        expected = 1.5 + math.sqrt(3) / 2
        assert abs(self.va["surface_area"] - expected) < 1e-10

    def test_center_of_mass(self):
        com = self.va["center_of_mass"]
        assert abs(com[0] - 0.25) < 1e-10
        assert abs(com[1] - 0.25) < 1e-10
        assert abs(com[2] - 0.25) < 1e-10


class TestComputeMeshPropsUnitCube:
    def setup_method(self):
        self.va, self.moi = _compute_mesh_props(UNIT_CUBE)

    def test_volume(self):
        assert abs(self.va["volume"] - 1.0) < 1e-10

    def test_surface_area(self):
        assert abs(self.va["surface_area"] - 6.0) < 1e-10

    def test_center_of_mass(self):
        com = self.va["center_of_mass"]
        assert abs(com[0] - 0.5) < 1e-10
        assert abs(com[1] - 0.5) < 1e-10
        assert abs(com[2] - 0.5) < 1e-10

    def test_principal_moments(self):
        # Ixx = Iyy = Izz = 1/6 for unit cube about its CoM
        expected = 1 / 6
        assert abs(self.moi["Ixx"] - expected) < 1e-10
        assert abs(self.moi["Iyy"] - expected) < 1e-10
        assert abs(self.moi["Izz"] - expected) < 1e-10

    def test_products_of_inertia_zero(self):
        # Cube is symmetric — all products of inertia must vanish
        assert abs(self.moi["Ixy"]) < 1e-10
        assert abs(self.moi["Ixz"]) < 1e-10
        assert abs(self.moi["Iyz"]) < 1e-10

    def test_scaled_cube(self):
        # 2×3×4 box: volume=24, Ixx=(M/12)*(b²+c²)=(24/12)*(9+16)=50
        box = []
        for tri in UNIT_CUBE:
            box.append(tuple((x * 2, y * 3, z * 4) for x, y, z in tri))
        va, moi = _compute_mesh_props(box)
        assert abs(va["volume"] - 24.0) < 1e-8
        assert abs(va["surface_area"] - 2 * (2*3 + 2*4 + 3*4)) < 1e-8
        expected_Ixx = (24 / 12) * (3**2 + 4**2)  # (M/12)*(b²+c²) = 2*25 = 50
        assert abs(moi["Ixx"] - expected_Ixx) < 1e-6


# ── _segments_to_area ────────────────────────────────────────────────

class TestSegmentsToArea:
    def test_empty(self):
        area, cx, cy = _segments_to_area([])
        assert area == 0.0
        assert cx == 0.0
        assert cy == 0.0

    def test_unit_square(self):
        segments = [
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (0, 1)),
            ((0, 1), (0, 0)),
        ]
        area, cx, cy = _segments_to_area(segments)
        assert abs(area - 1.0) < 1e-10
        assert abs(cx - 0.5) < 1e-10
        assert abs(cy - 0.5) < 1e-10

    def test_2x3_rectangle(self):
        segments = [
            ((0, 0), (2, 0)),
            ((2, 0), (2, 3)),
            ((2, 3), (0, 3)),
            ((0, 3), (0, 0)),
        ]
        area, cx, cy = _segments_to_area(segments)
        assert abs(area - 6.0) < 1e-10
        assert abs(cx - 1.0) < 1e-10
        assert abs(cy - 1.5) < 1e-10

    def test_reversed_segment_direction(self):
        # Segments in reverse order should still assemble correctly
        segments = [
            ((0, 1), (0, 0)),
            ((1, 1), (0, 1)),
            ((1, 0), (1, 1)),
            ((0, 0), (1, 0)),
        ]
        area, cx, cy = _segments_to_area(segments)
        assert abs(area - 1.0) < 1e-10

    def test_two_separate_squares(self):
        # Two non-adjacent unit squares — both should be counted
        s1 = [((0,0),(1,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((0,1),(0,0))]
        s2 = [((3,0),(4,0)), ((4,0),(4,1)), ((4,1),(3,1)), ((3,1),(3,0))]
        area, cx, cy = _segments_to_area(s1 + s2)
        assert abs(area - 2.0) < 1e-10
        assert abs(cx - 2.0) < 1e-10   # average of 0.5 and 3.5
        assert abs(cy - 0.5) < 1e-10
