"""Outcome-based tests for the bundled examples.

Each test loads an example file via CadFingerprint and checks that the
fingerprint values match the reference constants in the corresponding
generated test file. This validates the full analysis pipeline end-to-end.
"""

import os
import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
PEGHEAD_STEP = os.path.join(EXAMPLES_DIR, "peghead7mm.step")
CRANK_STL = os.path.join(EXAMPLES_DIR, "crank.stl")


# ── peghead STEP ─────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(PEGHEAD_STEP), reason="peghead7mm.step not in examples/")
class TestPegheadStep:
    """Fingerprint of peghead7mm.step matches the values in test_peghead.py."""

    @pytest.fixture(autouse=True, scope="class")
    def fingerprint(self):
        from b123d_decompiler.fingerprint import CadFingerprint
        type(self)._fp = CadFingerprint.from_step(PEGHEAD_STEP)

    def test_volume(self):
        assert abs(self._fp.volume_and_area["volume"] - 375.6419) / 375.6419 < 0.001

    def test_surface_area(self):
        assert abs(self._fp.volume_and_area["surface_area"] - 568.3970) / 568.3970 < 0.001

    def test_bounding_box_size(self):
        size = self._fp.bounding_box["size"]
        for actual, ref in zip(size, (12.5000, 8.5000, 29.4260)):
            assert abs(actual - ref) < 0.01, f"bbox size {actual:.4f} vs ref {ref}"

    def test_inertia_principal(self):
        moi = self._fp.moments_of_inertia
        assert abs(moi["Ixx"] - 20339.6797) / 20339.6797 < 0.005
        assert abs(moi["Iyy"] - 22246.9700) / 22246.9700 < 0.005
        assert abs(moi["Izz"] - 2959.4089)  / 2959.4089  < 0.005

    def test_face_count(self):
        assert abs(len(self._fp.face_inventory) - 27) <= 3

    def test_face_types_present(self):
        types = {f["type"] for f in self._fp.face_inventory}
        for expected in ("Cylinder", "Plane", "Torus", "BSpline"):
            assert expected in types, f"Expected face type '{expected}' not found; got {types}"

    def test_cylinder_diameters(self):
        diams = sorted(
            f["diameter"] for f in self._fp.face_inventory if "diameter" in f
        )
        ref_diams = [0.597, 1.0, 2.1, 3.8, 7.0, 9.8]
        for rd in ref_diams:
            assert any(abs(d - rd) < 0.1 for d in diams), (
                f"Reference cylinder d={rd:.3f} not found in {diams}"
            )

    def test_cross_sections_count(self):
        assert len(self._fp.cross_sections) == 20

    def test_cross_sections_nonzero(self):
        areas = [cs["area"] for cs in self._fp.cross_sections]
        assert any(a > 0 for a in areas)

    def test_radial_profile_shape(self):
        assert len(self._fp.radial_profile) == 15
        for rp in self._fp.radial_profile:
            assert len(rp["radii"]) == 12

    def test_source_format(self):
        assert self._fp.source_format == "step"


# ── crank STL ────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(CRANK_STL), reason="crank.stl not in examples/")
class TestCrankStl:
    """Fingerprint of crank.stl matches the values in test_crank.py."""

    @pytest.fixture(autouse=True, scope="class")
    def fingerprint(self):
        from b123d_decompiler.fingerprint import CadFingerprint
        type(self)._fp = CadFingerprint.from_stl(CRANK_STL)

    def test_volume(self):
        assert abs(self._fp.volume_and_area["volume"] - 8230.2040) / 8230.2040 < 0.001

    def test_surface_area(self):
        assert abs(self._fp.volume_and_area["surface_area"] - 3868.7450) / 3868.7450 < 0.001

    def test_bounding_box_size(self):
        size = self._fp.bounding_box["size"]
        for actual, ref in zip(size, (55.9950, 15.0000, 16.0000)):
            assert abs(actual - ref) < 0.1, f"bbox size {actual:.4f} vs ref {ref}"

    def test_inertia_principal(self):
        moi = self._fp.moments_of_inertia
        assert abs(moi["Ixx"] -  273430.3485) /  273430.3485 < 0.005
        assert abs(moi["Iyy"] - 2142180.7884) / 2142180.7884 < 0.005
        assert abs(moi["Izz"] - 2181015.6360) / 2181015.6360 < 0.005

    def test_face_inventory_is_mesh(self):
        faces = self._fp.face_inventory
        assert len(faces) == 1
        assert faces[0]["type"] == "mesh"
        assert faces[0]["triangle_count"] > 0

    def test_cross_sections_count(self):
        assert len(self._fp.cross_sections) == 20

    def test_cross_sections_nonzero(self):
        areas = [cs["area"] for cs in self._fp.cross_sections]
        assert any(a > 0 for a in areas)

    def test_radial_profile_shape(self):
        assert len(self._fp.radial_profile) == 15
        for rp in self._fp.radial_profile:
            assert len(rp["radii"]) == 12

    def test_source_format(self):
        assert self._fp.source_format == "stl"
