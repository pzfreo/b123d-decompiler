"""Integration tests for analyze_step and analyze_stl.

Requires build123d and OCC. Marked with pytest.mark.integration
but run by default (no special flag needed — they're just slower).
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
PEGHEAD_STEP = os.path.join(EXAMPLES_DIR, "peghead7mm.step")


# ── analyze_step ─────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(PEGHEAD_STEP), reason="peghead7mm.step not found")
class TestAnalyzeStep:
    def setup_method(self):
        from b123d_decompiler.fingerprint.analyze import analyze_step
        self.result = analyze_step(PEGHEAD_STEP)

    def test_returns_all_keys(self):
        expected = {
            "file", "source_format", "bounding_box", "volume_and_area",
            "moments_of_inertia", "topology", "face_inventory",
            "cross_sections", "radial_profile", "build_quality", "description",
        }
        assert expected.issubset(self.result.keys())

    def test_source_format(self):
        assert self.result["source_format"] == "step"

    def test_volume(self):
        # peghead: 375.64 mm³ from reference test file header
        vol = self.result["volume_and_area"]["volume"]
        assert abs(vol - 375.64) / 375.64 < 0.01

    def test_bounding_box_positive(self):
        bb = self.result["bounding_box"]
        for dim in bb["size"]:
            assert dim > 0

    def test_face_inventory_has_cylinders(self):
        faces = self.result["face_inventory"]
        types = {f["type"] for f in faces}
        assert "Cylinder" in types, f"No cylinders found; types = {types}"

    def test_cross_sections_count(self):
        assert len(self.result["cross_sections"]) == 20

    def test_cross_sections_positive_area(self):
        areas = [cs["area"] for cs in self.result["cross_sections"]]
        assert any(a > 0 for a in areas), "All cross-section areas are zero"

    def test_radial_profile_count(self):
        # 15 slices × 12 angles
        assert len(self.result["radial_profile"]) == 15
        for rp in self.result["radial_profile"]:
            assert len(rp["radii"]) == 12

    def test_inertia_diagonal_positive(self):
        moi = self.result["moments_of_inertia"]
        assert moi["Ixx"] > 0
        assert moi["Iyy"] > 0
        assert moi["Izz"] > 0

    def test_build_quality_single_solid(self):
        bq = self.result["build_quality"]
        assert bq["solid_count"] == 1
        assert bq["is_valid"] is True


# ── analyze_stl ──────────────────────────────────────────────────────

class TestAnalyzeStlBox:
    """Analyze a 10×20×30 box exported from build123d as STL.

    A rectangular box has flat faces — the triangulation is exact,
    so volume and surface area should match the analytical values tightly.
    """

    @pytest.fixture(autouse=True)
    def create_and_analyze(self, tmp_path):
        from build123d import Box, export_stl
        from b123d_decompiler.fingerprint.analyze import analyze_stl

        stl_path = str(tmp_path / "box.stl")
        box = Box(10, 20, 30)
        export_stl(box, stl_path, tolerance=1e-4)

        self.result = analyze_stl(stl_path)

    def test_returns_all_keys(self):
        expected = {
            "file", "source_format", "bounding_box", "volume_and_area",
            "moments_of_inertia", "topology", "face_inventory",
            "cross_sections", "radial_profile", "build_quality",
        }
        assert expected.issubset(self.result.keys())

    def test_source_format(self):
        assert self.result["source_format"] == "stl"

    def test_volume(self):
        # 10×20×30 = 6000 mm³; flat faces → exact triangulation
        vol = self.result["volume_and_area"]["volume"]
        assert abs(vol - 6000.0) < 0.1

    def test_surface_area(self):
        # 2*(10*20 + 10*30 + 20*30) = 2200 mm²
        area = self.result["volume_and_area"]["surface_area"]
        assert abs(area - 2200.0) < 0.1

    def test_bounding_box(self):
        bb = self.result["bounding_box"]
        sizes = sorted(bb["size"])
        assert abs(sizes[0] - 10) < 0.01
        assert abs(sizes[1] - 20) < 0.01
        assert abs(sizes[2] - 30) < 0.01

    def test_center_of_mass_at_origin(self):
        com = self.result["volume_and_area"]["center_of_mass"]
        for v in com:
            assert abs(v) < 0.01, f"CoM component {v} not near origin"

    def test_inertia_diagonal(self):
        # Ixx = (M/12)*(b²+c²) about CoM; M=6000, a=10, b=20, c=30
        moi = self.result["moments_of_inertia"]
        expected_Ixx = (6000 / 12) * (20**2 + 30**2)   # 650000
        expected_Iyy = (6000 / 12) * (10**2 + 30**2)   # 500000
        expected_Izz = (6000 / 12) * (10**2 + 20**2)   # 250000
        assert abs(moi["Ixx"] - expected_Ixx) / expected_Ixx < 0.001
        assert abs(moi["Iyy"] - expected_Iyy) / expected_Iyy < 0.001
        assert abs(moi["Izz"] - expected_Izz) / expected_Izz < 0.001

    def test_products_of_inertia_near_zero(self):
        moi = self.result["moments_of_inertia"]
        # Box centered at origin — all products of inertia should vanish
        assert abs(moi["Ixy"]) < 1.0
        assert abs(moi["Ixz"]) < 1.0
        assert abs(moi["Iyz"]) < 1.0

    def test_face_inventory_is_mesh(self):
        faces = self.result["face_inventory"]
        assert len(faces) == 1
        assert faces[0]["type"] == "mesh"
        assert faces[0]["triangle_count"] > 0

    def test_cross_sections_count(self):
        assert len(self.result["cross_sections"]) == 20

    def test_cross_sections_area(self):
        # Cross-sections along Z: each should be 10×20 = 200 mm²
        for cs in self.result["cross_sections"]:
            assert abs(cs["area"] - 200.0) < 1.0, (
                f"Cross-section at Z={cs['position']}: area={cs['area']:.2f}, expected ~200"
            )

    def test_radial_profile_has_data(self):
        assert len(self.result["radial_profile"]) == 15
        # At least some radii should be non-None
        any_hit = any(
            r is not None
            for rp in self.result["radial_profile"]
            for r in rp["radii"].values()
        )
        assert any_hit, "All radial profile rays missed the shape"


# ── surface mesh + Hausdorff distance ────────────────────────────────

class TestSurfaceMeshAndHausdorff:
    """Mesh capture and end-to-end Hausdorff distance on real B-rep parts."""

    def test_step_fingerprint_carries_a_mesh(self):
        from b123d_decompiler.fingerprint.analyze import analyze_step, decoded_mesh

        if not os.path.exists(PEGHEAD_STEP):
            pytest.skip("peghead7mm.step not found")
        mesh = analyze_step(PEGHEAD_STEP)["surface_mesh"]
        assert mesh["triangle_count"] > 100
        assert mesh["deflection"] > 0
        vertices, triangles = decoded_mesh(mesh)
        assert len(vertices) == mesh["vertex_count"]
        assert len(triangles) == mesh["triangle_count"]
        assert max(max(t) for t in triangles) < len(vertices)

    def test_mesh_covers_the_whole_part(self):
        """Every mesh vertex must sit inside the part's bounding box."""
        from build123d import Box
        from b123d_decompiler.fingerprint.analyze import decoded_mesh, mesh_shape

        box = Box(10, 20, 30)
        vertices, _ = decoded_mesh(mesh_shape(box))
        for x, y, z in vertices:
            assert abs(x) <= 5.001 and abs(y) <= 10.001 and abs(z) <= 15.001

    def test_identical_parts_have_zero_deviation(self):
        from build123d import Box
        from b123d_decompiler.fingerprint.analyze import decoded_mesh, mesh_shape
        from b123d_decompiler.fingerprint.hausdorff import hausdorff_distance

        mesh = decoded_mesh(mesh_shape(Box(10, 20, 30)))
        result = hausdorff_distance(mesh, mesh, samples=500)
        assert result["hausdorff"] < 1e-6

    def test_deviation_matches_a_known_offset(self):
        """A 0.4 mm shift must be measured as a 0.4 mm deviation."""
        from build123d import Box, Pos
        from b123d_decompiler.fingerprint.analyze import decoded_mesh, mesh_shape
        from b123d_decompiler.fingerprint.hausdorff import hausdorff_distance

        reference = decoded_mesh(mesh_shape(Box(10, 20, 30)))
        shifted = decoded_mesh(mesh_shape(Pos(0.4, 0, 0) * Box(10, 20, 30)))
        result = hausdorff_distance(reference, shifted, samples=1000)
        assert abs(result["hausdorff"] - 0.4) < 0.01

    def test_curved_part_deviation_below_mesh_resolution(self):
        """Two meshings of the same cylinder agree to within the deflection."""
        from build123d import Cylinder
        from b123d_decompiler.fingerprint.analyze import decoded_mesh, mesh_shape
        from b123d_decompiler.fingerprint.hausdorff import hausdorff_distance

        part = Cylinder(radius=8, height=20)
        coarse = decoded_mesh(mesh_shape(part, deflection=0.05))
        fine = decoded_mesh(mesh_shape(part, deflection=0.01))
        result = hausdorff_distance(coarse, fine, samples=1000)
        assert result["hausdorff"] < 0.06
        assert result["mean"] < 0.02


class TestGeneratedHausdorffTests:
    """The embedded Hausdorff code in generated test files actually runs."""

    @pytest.fixture(scope="class")
    def generated(self, tmp_path_factory):
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        step_path = str(tmp_path_factory.mktemp("gen") / "box.step")
        export_step(Box(10, 20, 30), step_path)
        fp = CadFingerprint.from_step(step_path)
        source = generate_test_file(fp, module_name="box")
        namespace = {}
        exec(compile(source, "generated_test_box.py", "exec"), namespace)
        return namespace

    def test_emits_reference_mesh_and_tests(self, generated):
        assert generated["REF_MESH"]["triangle_count"] >= 12
        assert generated["HAUSDORFF_SAMPLES"] > 0
        assert "TestSurfaceDeviation" in generated

    def test_reference_part_has_no_deviation(self, generated):
        from build123d import Box

        result = generated["_hausdorff_vs_reference"](Box(10, 20, 30))
        assert result["hausdorff"] < 1e-3
        assert result["mean"] < 1e-3

    def test_wrong_part_is_caught(self, generated):
        from build123d import Box

        result = generated["_hausdorff_vs_reference"](Box(10, 20, 30.8))
        # 0.8 mm longer, centred — each end face is 0.4 mm out
        assert abs(result["hausdorff"] - 0.4) < 0.02

    def test_results_are_cached_across_fixture_rebuilds(self, generated):
        """A function-scoped fixture rebuilds the part for every test.

        The cache has to key on geometry, not object identity, or the max and
        mean tests each pay for the whole measurement.
        """
        from build123d import Box

        first = generated["_hausdorff_vs_reference"](Box(10, 20, 30))
        second = generated["_hausdorff_vs_reference"](Box(10, 20, 30))
        assert second is first

    def test_cache_distinguishes_different_parts(self, generated):
        from build123d import Box

        same = generated["_hausdorff_vs_reference"](Box(10, 20, 30))
        other = generated["_hausdorff_vs_reference"](Box(10, 20, 30.8))
        assert other is not same
        assert other["hausdorff"] > same["hausdorff"]

    def test_fixture_part_is_not_mutated(self, generated):
        """Meshing the part under test must not alter the caller's shape."""
        from build123d import Box
        from OCP.BRep import BRep_Tool
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS

        part = Box(10, 20, 30)

        def triangulated_faces():
            count = 0
            explorer = TopExp_Explorer(part.wrapped, TopAbs_FACE)
            while explorer.More():
                face = TopoDS.Face_s(explorer.Current())
                if BRep_Tool.Triangulation_s(face, TopLoc_Location()) is not None:
                    count += 1
                explorer.Next()
            return count

        before = triangulated_faces()
        generated["_hausdorff_vs_reference"](part)
        assert triangulated_faces() == before


class TestCompareHausdorff:
    """The compare command reports surface deviation alongside the rest."""

    @staticmethod
    def _fingerprint(part, tmp_path, name, **kwargs):
        from build123d import export_step
        from b123d_decompiler.fingerprint import CadFingerprint

        path = str(tmp_path / f"{name}.step")
        export_step(part, path)
        return CadFingerprint.from_step(path, **kwargs)

    def test_reports_deviation_of_a_bad_part(self, tmp_path):
        from build123d import Box
        from b123d_decompiler.fingerprint.compare import compare_fingerprints, format_comparison

        ref = self._fingerprint(Box(10, 20, 30), tmp_path, "ref")
        impl = self._fingerprint(Box(10, 20, 30.8), tmp_path, "impl")
        result = compare_fingerprints(ref, impl, hausdorff_samples=500)

        h = result["hausdorff"]
        # Each end face is 0.4 mm out — past the 0.3 mm tolerance but
        # within 2x, which the report grades as "close".
        assert abs(h["max"] - 0.4) < 0.02
        assert h["status"] == "close"
        assert "Hausdorff" in format_comparison(result)

    def test_grossly_wrong_part_fails(self, tmp_path):
        from build123d import Box
        from b123d_decompiler.fingerprint.compare import compare_fingerprints

        ref = self._fingerprint(Box(10, 20, 30), tmp_path, "big_ref")
        impl = self._fingerprint(Box(10, 20, 32), tmp_path, "big_impl")
        result = compare_fingerprints(ref, impl, hausdorff_samples=500)
        assert abs(result["hausdorff"]["max"] - 1.0) < 0.02
        assert result["hausdorff"]["status"] == "fail"
        assert result["summary"]["fail"] >= 1

    def test_matching_parts_pass(self, tmp_path):
        from build123d import Box
        from b123d_decompiler.fingerprint.compare import compare_fingerprints

        ref = self._fingerprint(Box(10, 20, 30), tmp_path, "a")
        impl = self._fingerprint(Box(10, 20, 30), tmp_path, "b")
        result = compare_fingerprints(ref, impl, hausdorff_samples=500)
        assert result["hausdorff"]["status"] == "pass"
        assert result["hausdorff"]["max"] < 1e-3

    def test_skipped_when_meshes_were_not_captured(self, tmp_path):
        from build123d import Box
        from b123d_decompiler.fingerprint.compare import compare_fingerprints, format_comparison

        ref = self._fingerprint(Box(10, 20, 30), tmp_path, "c", capture_mesh=False)
        impl = self._fingerprint(Box(10, 20, 30), tmp_path, "d", capture_mesh=False)
        result = compare_fingerprints(ref, impl)
        assert "hausdorff" not in result
        # ...but the report must say the check did not run, rather than
        # looking complete without it
        assert "not measured" in format_comparison(result)


class TestMeshResolutionLimits:
    """Mesh resolution has to bound what the generated tolerances claim."""

    def test_coarse_reference_raises_the_tolerances(self, tmp_path):
        """A mesh too coarse to resolve 0.3 mm must not assert 0.3 mm."""
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file
        from b123d_decompiler.fingerprint.hausdorff import tolerance_floor

        path = str(tmp_path / "coarse.step")
        export_step(Sphere(radius=10), path)
        fp = CadFingerprint.from_step(path, mesh_deflection=1.0)
        floor = tolerance_floor(fp.surface_mesh)
        assert floor > 0.3, "1 mm facets on a 10 mm ball should exceed 0.3 mm"

        source = generate_test_file(fp, module_name="coarse")
        assert "Surface-deviation tolerance raised" in source
        assert f"max({floor}, result" in source

        namespace = {}
        exec(compile(source, "coarse_test.py", "exec"), namespace)
        assert namespace["REF_MESH"]["resolution"] > 0.1

    def test_flat_part_keeps_its_tolerances_however_coarse(self, tmp_path):
        """A box's facets are exact — deflection is irrelevant to its error."""
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "flat.step")
        export_step(Box(10, 20, 30), path)
        fp = CadFingerprint.from_step(path, mesh_deflection=2.0)
        assert fp.surface_mesh["resolution"] == 0.0

        source = generate_test_file(fp, module_name="flat")
        assert "Surface-deviation tolerance raised" not in source
        assert "max(0.3, result" in source

    def test_finer_deflection_tightens_the_angular_limit(self, tmp_path):
        """Asking for a finer mesh has to actually produce a finer one.

        On blends and small fillets the angular limit, not the linear one,
        is what bounds the chord error.
        """
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint

        path = str(tmp_path / "ball.step")
        export_step(Sphere(radius=10), path)
        default = CadFingerprint.from_step(path).surface_mesh
        finer = CadFingerprint.from_step(
            path, mesh_deflection=default["deflection"] / 10,
            max_mesh_triangles=200000,
        ).surface_mesh

        assert finer["angular_deflection"] < default["angular_deflection"]
        assert finer["triangle_count"] > default["triangle_count"]
        assert finer["resolution"] < default["resolution"]

    def test_triangle_budget_is_configurable(self, tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint

        path = str(tmp_path / "budget.step")
        export_step(Sphere(radius=10), path)
        small = CadFingerprint.from_step(path, max_mesh_triangles=500).surface_mesh
        assert small["triangle_count"] <= 500

    def test_fine_reference_keeps_the_requested_tolerances(self, tmp_path):
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "fine.step")
        export_step(Box(10, 20, 30), path)
        fp = CadFingerprint.from_step(path)
        source = generate_test_file(fp, module_name="fine")

        assert "Surface-deviation tolerance raised" not in source
        assert "max(0.3, result" in source
        assert "max(0.05, result" in source

    def test_stl_facet_error_is_measured(self, tmp_path):
        """A curved STL is itself an approximation — that must be recorded.

        Otherwise a coarsely exported reference holds implementations to a
        tolerance its own facets cannot meet.
        """
        from build123d import Box, Sphere, export_stl
        from b123d_decompiler.fingerprint.analyze import analyze_stl

        ball = str(tmp_path / "ball_res.stl")
        block = str(tmp_path / "block_res.stl")
        export_stl(Sphere(radius=10), ball)
        export_stl(Box(10, 20, 30), block)

        # Curved: facets are chords, so they carry a real error
        assert analyze_stl(ball)["surface_mesh"]["resolution"] > 0.0
        # Flat: a box's facets are exact, and its sharp edges are features,
        # not chord error
        assert analyze_stl(block)["surface_mesh"]["resolution"] == 0.0

    def test_stl_deflection_clusters_the_reference_mesh(self, tmp_path):
        """--mesh-deflection has to do something for STL, not silently pass."""
        from build123d import Sphere, export_stl
        from b123d_decompiler.fingerprint.analyze import analyze_stl

        path = str(tmp_path / "ball.stl")
        export_stl(Sphere(radius=10), path, tolerance=0.01)

        as_supplied = analyze_stl(path)["surface_mesh"]
        clustered = analyze_stl(path, mesh_deflection=1.5)["surface_mesh"]

        assert clustered["triangle_count"] < as_supplied["triangle_count"]
        assert clustered["cluster_cell"] == 1.5
        # Facet error and clustering displacement are independent, so the
        # reference mesh carries both
        assert clustered["resolution"] == (
            1.5 + clustered["facet_error"]
        )
        assert clustered["resolution"] > as_supplied["resolution"]

    def test_stl_triangle_budget_is_enforced(self, tmp_path):
        from build123d import Sphere, export_stl
        from b123d_decompiler.fingerprint.analyze import decoded_mesh, load_stl, mesh_shape
        from b123d_decompiler.fingerprint.hausdorff import hausdorff_distance

        path = str(tmp_path / "dense.stl")
        export_stl(Sphere(radius=10), path, tolerance=0.005)
        face = load_stl(path)

        full = mesh_shape(face, remesh=False, max_triangles=10**9)
        capped = mesh_shape(face, remesh=False, max_triangles=1500)
        assert full["triangle_count"] > 1500
        assert capped["triangle_count"] <= 1500

        # Decimation must stay within the resolution it reports
        deviation = hausdorff_distance(
            decoded_mesh(full), decoded_mesh(capped), samples=800,
        )
        assert deviation["hausdorff"] < capped["resolution"] * 2


class TestEmptyMesh:
    """A shape with no triangulation must degrade gracefully, not report inf."""

    def test_generator_skips_an_empty_mesh(self, tmp_path):
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "empty.step")
        export_step(Box(10, 20, 30), path)
        fp = CadFingerprint.from_step(path)
        fp.surface_mesh = {"encoding": "q16", "vertex_count": 0,
                           "triangle_count": 0, "bbox_min": (0.0, 0.0, 0.0),
                           "bbox_max": (0.0, 0.0, 0.0), "index_bits": 16,
                           "vertices": "", "triangles": "", "deflection": 0.1,
                           "resolution": 0.1}
        source = generate_test_file(fp, module_name="empty")
        assert "REF_MESH" not in source
        assert "TestSurfaceDeviation" not in source

    def test_compare_skips_an_empty_mesh(self, tmp_path):
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.compare import compare_fingerprints

        path = str(tmp_path / "cmp.step")
        export_step(Box(10, 20, 30), path)
        ref = CadFingerprint.from_step(path)
        impl = CadFingerprint.from_step(path)
        impl.surface_mesh = dict(impl.surface_mesh, triangle_count=0)
        assert "hausdorff" not in compare_fingerprints(ref, impl)


class TestCompareMeanTolerance:
    """--hausdorff-mean-tol has to reach compare, not just the generator."""

    def test_mean_tolerance_is_applied(self, tmp_path):
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.compare import compare_fingerprints

        ref_path = str(tmp_path / "a.step")
        impl_path = str(tmp_path / "b.step")
        export_step(Box(10, 20, 30), ref_path)
        export_step(Box(10, 20, 31), impl_path)
        ref = CadFingerprint.from_step(ref_path)
        impl = CadFingerprint.from_step(impl_path)

        relaxed = compare_fingerprints(ref, impl, hausdorff_mean_tol_mm=1.0,
                                       hausdorff_samples=500)
        strict = compare_fingerprints(ref, impl, hausdorff_mean_tol_mm=0.001,
                                      hausdorff_samples=500)
        assert relaxed["hausdorff"]["mean_status"] == "pass"
        assert strict["hausdorff"]["mean_status"] == "fail"
        assert strict["hausdorff"]["status"] == "fail"

    def test_mean_tolerance_is_never_below_the_mesh_resolution(self, tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.compare import compare_fingerprints

        path = str(tmp_path / "res.step")
        export_step(Sphere(radius=10), path)
        ref = CadFingerprint.from_step(path, mesh_deflection=1.0)
        impl = CadFingerprint.from_step(path, mesh_deflection=1.0)
        result = compare_fingerprints(ref, impl, hausdorff_mean_tol_mm=0.0001,
                                      hausdorff_samples=500)
        assert result["hausdorff"]["mean_tolerance"] > 0.0001
        assert result["hausdorff"]["resolution_limited"] is True


class TestThinPartClustering:
    """Clustering must never collapse a part's thinnest dimension."""

    def test_plate_keeps_its_thickness(self, tmp_path):
        from build123d import Box, export_stl
        from b123d_decompiler.fingerprint.analyze import decoded_mesh, load_stl, mesh_shape

        path = str(tmp_path / "plate.stl")
        export_stl(Box(200, 200, 1), path)
        face = load_stl(path)

        # A budget this small would ask for a cell several times the
        # plate's 1 mm thickness, snapping both faces into one layer.
        mesh = mesh_shape(face, remesh=False, max_triangles=200)
        vertices, _ = decoded_mesh(mesh)
        thickness = max(v[2] for v in vertices) - min(v[2] for v in vertices)
        assert thickness > 0.5, f"plate flattened to {thickness:.4f} mm"
        assert mesh["cluster_cell"] <= 1.0 / 3 + 1e-9

    def test_explicit_cell_is_also_capped(self, tmp_path):
        from build123d import Box, export_stl
        from b123d_decompiler.fingerprint.analyze import decoded_mesh, load_stl, mesh_shape

        path = str(tmp_path / "plate2.stl")
        export_stl(Box(200, 200, 1), path)
        mesh = mesh_shape(load_stl(path), deflection=25.0, remesh=False)
        vertices, _ = decoded_mesh(mesh)
        thickness = max(v[2] for v in vertices) - min(v[2] for v in vertices)
        assert thickness > 0.5


class TestCacheSignature:
    """The generated cache must not confuse two genuinely different parts."""

    def test_mirrored_feature_is_not_a_cache_hit(self, tmp_path):
        """Same volume, same envelope, feature on the other side.

        Exactly the defect class the surface-deviation test exists to catch,
        so a cache keyed only on volume and bounding box would hide it.
        """
        from build123d import Box, Pos, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        left = Box(40, 20, 10) - (Pos(-15, 0, 0) * Box(6, 6, 12))
        right = Box(40, 20, 10) - (Pos(15, 0, 0) * Box(6, 6, 12))

        path = str(tmp_path / "left.step")
        export_step(left, path)
        source = generate_test_file(CadFingerprint.from_step(path),
                                    module_name="left")
        namespace = {}
        exec(compile(source, "left_test.py", "exec"), namespace)

        signature = namespace["_shape_signature"]
        assert signature(left) != signature(right), (
            "mirrored parts share a cache key"
        )
        measured = namespace["_hausdorff_vs_reference"](left)
        mirrored = namespace["_hausdorff_vs_reference"](right)
        assert mirrored is not measured
        assert measured["hausdorff"] < 0.01
        assert mirrored["hausdorff"] > 1.0


class TestCliValidation:
    """Flags that would silently disable the measurement must be rejected."""

    def test_sample_count_must_be_positive(self):
        import argparse
        import pytest
        from b123d_decompiler.fingerprint.cli import _positive_int

        assert _positive_int("2000") == 2000
        for bad in ("0", "-5"):
            with pytest.raises(argparse.ArgumentTypeError):
                _positive_int(bad)

    def test_deflection_must_be_positive(self):
        import argparse
        import pytest
        from b123d_decompiler.fingerprint.cli import _positive_float

        assert _positive_float("0.05") == 0.05
        for bad in ("0", "-0.1"):
            with pytest.raises(argparse.ArgumentTypeError):
                _positive_float(bad)


class TestStlFacetError:
    """A coarse STL's own faceting cannot be guessed — it can be declared."""

    def test_declared_error_overrides_the_estimate(self, tmp_path):
        from build123d import Sphere, export_stl
        from b123d_decompiler.fingerprint.analyze import analyze_stl

        path = str(tmp_path / "ball_declared.stl")
        export_stl(Sphere(radius=10), path)

        estimated = analyze_stl(path)["surface_mesh"]
        declared = analyze_stl(path, stl_facet_error=0.8)["surface_mesh"]
        assert declared["resolution"] == 0.8
        assert declared["facet_error_declared"] is True
        assert estimated["facet_error_declared"] is False
        assert declared["resolution"] > estimated["resolution"]

    def test_declared_error_reaches_the_generated_tolerances(self, tmp_path):
        from build123d import Sphere, export_stl
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file
        from b123d_decompiler.fingerprint.hausdorff import tolerance_floor

        path = str(tmp_path / "ball_tol.stl")
        export_stl(Sphere(radius=10), path)
        fp = CadFingerprint.from_stl(path, stl_facet_error=1.0)

        floor = tolerance_floor(fp.surface_mesh)
        assert floor > 2.0
        source = generate_test_file(fp, module_name="ball")
        assert f"max({floor}, result" in source
        assert "--stl-facet-error" in source


class TestMeasuredStepResolution:
    """For STEP the chord error is measured against the real surface."""

    def test_flat_part_measures_zero_however_coarse(self, tmp_path):
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint

        path = str(tmp_path / "flat_measured.step")
        export_step(Box(10, 20, 30), path)
        assert CadFingerprint.from_step(
            path, mesh_deflection=5.0
        ).surface_mesh["resolution"] == 0.0

    def test_curved_part_measures_the_real_deviation(self, tmp_path):
        """The floor has to cover the noise between two tessellations.

        A coarse mesh is where a heuristic estimator collapsed to zero and
        emitted tolerances a correct implementation could not meet.
        """
        from build123d import Cylinder, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.analyze import _triangulate, load_step
        from b123d_decompiler.fingerprint.hausdorff import hausdorff_distance, tolerance_floor

        path = str(tmp_path / "coarse_cyl.step")
        export_step(Cylinder(radius=10, height=20), path)
        fp = CadFingerprint.from_step(path, mesh_deflection=3.0)
        mesh = fp.surface_mesh
        assert mesh["resolution"] > 0.0

        shape = load_step(path)
        noise = hausdorff_distance(
            _triangulate(shape, mesh["deflection"],
                         mesh["angular_deflection"], True),
            _triangulate(shape, mesh["deflection"] * 0.8,
                         mesh["angular_deflection"] * 0.8, True),
            samples=800,
        )["hausdorff"]
        assert noise < tolerance_floor(mesh), (
            f"meshing noise {noise:.4f} exceeds the floor "
            f"{tolerance_floor(mesh):.4f}"
        )


class TestCompareTessellationMatch:
    """compare must tessellate both sides identically, or it measures noise."""

    def test_angular_deflection_is_propagated(self, tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint

        path = str(tmp_path / "prop.step")
        export_step(Sphere(radius=10), path)
        ref = CadFingerprint.from_step(path, mesh_deflection=0.01,
                                       max_mesh_triangles=800)
        impl = CadFingerprint.from_step(
            path,
            mesh_deflection=ref.surface_mesh["deflection"],
            mesh_angular_deflection=ref.surface_mesh["angular_deflection"],
            max_mesh_triangles=10 ** 9,
        )
        assert (impl.surface_mesh["angular_deflection"]
                == ref.surface_mesh["angular_deflection"])
        assert (impl.surface_mesh["triangle_count"]
                == ref.surface_mesh["triangle_count"])


class TestCandidateMeshBudget:
    """The part under test gets the same triangle ceiling as the reference."""

    def test_budget_is_embedded_and_applied(self, tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "budget_gen.step")
        export_step(Sphere(radius=10), path)
        fp = CadFingerprint.from_step(path, max_mesh_triangles=600)
        source = generate_test_file(fp, module_name="budget")

        namespace = {}
        exec(compile(source, "budget_test.py", "exec"), namespace)
        budget = namespace["REF_MESH"]["max_triangles"]
        assert budget >= fp.surface_mesh["triangle_count"]

        # A far more detailed part must not blow past the ceiling
        result = namespace["_hausdorff_vs_reference"](Sphere(radius=10))
        assert result["hausdorff"] < 1.0


class TestHollowPartClustering:
    """Clustering must not weld a hollow part's inner and outer walls."""

    @staticmethod
    def _hollow_box(tmp_path, name, outer=100.0, wall=1.0):
        from build123d import Box, export_stl

        solid = Box(outer, outer, outer) - Box(
            outer - 2 * wall, outer - 2 * wall, outer - 2 * wall
        )
        path = str(tmp_path / f"{name}.stl")
        export_stl(solid, path)
        return path, solid.volume

    def test_thin_walls_survive_the_triangle_budget(self, tmp_path):
        """The part's envelope is 100 mm; only its walls are thin.

        A bounding-box ceiling sees nothing thin here, so the guard has to
        come from the mesh itself.
        """
        from b123d_decompiler.fingerprint.analyze import (
            _signed_volume, decoded_mesh, load_stl, mesh_shape,
        )

        path, material = self._hollow_box(tmp_path, "hollow")
        mesh = mesh_shape(load_stl(path), remesh=False, max_triangles=200)
        volume = abs(_signed_volume(*decoded_mesh(mesh)))
        assert abs(volume - material) / material < 0.05, (
            f"clustering changed the enclosed volume from {material:.0f} to "
            f"{volume:.0f} mm³ — the walls collapsed"
        )

    def test_signed_volume_matches_the_solid(self, tmp_path):
        from b123d_decompiler.fingerprint.analyze import (
            _signed_volume, _triangulate, load_stl,
        )

        path, material = self._hollow_box(tmp_path, "hollow2")
        vertices, triangles = _triangulate(load_stl(path), 0, 0, False)
        assert abs(abs(_signed_volume(vertices, triangles)) - material) < 1.0


class TestCandidateCoarseningTolerance:
    """Coarsening the candidate mesh has to move the tolerances with it."""

    def test_correct_part_still_passes_after_coarsening(self, tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "coarse_budget.step")
        export_step(Sphere(radius=10), path)
        # A tiny budget forces the reference coarse and makes the candidate
        # mesh hit the ceiling and get coarsened at test time.
        fp = CadFingerprint.from_step(path, max_mesh_triangles=400)
        source = generate_test_file(fp, module_name="coarse_budget")

        namespace = {}
        exec(compile(source, "coarse_budget_test.py", "exec"), namespace)
        result = namespace["_hausdorff_vs_reference"](Sphere(radius=10))
        assert result["floor"] > 0.0

        # The part is the reference: both assertions must hold
        suite = namespace["TestSurfaceDeviation"]()
        suite.test_max_deviation(Sphere(radius=10))
        suite.test_mean_deviation(Sphere(radius=10))

    def test_floor_tracks_the_deflection_actually_used(self, tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "floor_track.step")
        export_step(Sphere(radius=10), path)
        fp = CadFingerprint.from_step(path, max_mesh_triangles=400)
        namespace = {}
        exec(compile(generate_test_file(fp, module_name="ft"),
                     "ft_test.py", "exec"), namespace)

        result = namespace["_hausdorff_vs_reference"](Sphere(radius=10))
        # Whatever deflection the candidate ended up meshed at, the floor is
        # derived from that mesh rather than from the recorded one
        assert result["deflection"] >= namespace["REF_MESH"]["deflection"]
        assert result["floor"] >= 2.0 * namespace["REF_MESH"]["resolution"]


class TestUnreadableFacetErrorWarning:
    """The user has to be told when an STL's facet error cannot be read."""

    @staticmethod
    def _hex_prism_mesh(sides=6, radius=10.0, height=20.0):
        import math

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

    def test_warns_on_an_unreadable_mesh(self, capsys):
        from b123d_decompiler.fingerprint.cli import _warn_if_facet_error_unreadable
        from b123d_decompiler.fingerprint.hausdorff import encode_mesh

        mesh = encode_mesh(*self._hex_prism_mesh())
        mesh["facet_error"] = 0.0
        _warn_if_facet_error_unreadable(None, mesh)
        out = capsys.readouterr().out
        assert "--stl-facet-error" in out

    def test_silent_when_the_error_was_readable(self, capsys):
        from b123d_decompiler.fingerprint.cli import _warn_if_facet_error_unreadable
        from b123d_decompiler.fingerprint.hausdorff import encode_mesh

        mesh = encode_mesh(*self._hex_prism_mesh(sides=64))
        mesh["facet_error"] = 0.05
        _warn_if_facet_error_unreadable(None, mesh)
        assert capsys.readouterr().out == ""


class TestToleranceIsNotSelfReferential:
    """The part under test must not be able to widen its own tolerance."""

    def test_candidate_features_do_not_inflate_the_floor(self, tmp_path):
        """A chamfered candidate is a deviation, not extra mesh resolution.

        Reading chord error off the candidate's own facets let a shallow
        feature raise the tolerance until a genuinely wrong part passed.
        """
        import pytest
        from build123d import Box, chamfer, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "plain.step")
        plain = Box(20, 20, 20)
        export_step(plain, path)
        fp = CadFingerprint.from_step(path)
        namespace = {}
        exec(compile(generate_test_file(fp, module_name="plain"),
                     "plain_test.py", "exec"), namespace)

        chamfered = chamfer(Box(20, 20, 20).edges(), length=1.5)

        plain_floor = namespace["_hausdorff_vs_reference"](plain)["floor"]
        chamfered_result = namespace["_hausdorff_vs_reference"](chamfered)
        assert chamfered_result["floor"] == plain_floor, (
            "the candidate's own features changed the tolerance"
        )

        suite = namespace["TestSurfaceDeviation"]()
        with pytest.raises(AssertionError):
            suite.test_max_deviation(chamfered)


class TestGeneratedAdvice:
    """The 'how to tighten this' advice has to name the knob that works."""

    @staticmethod
    def _source_for(tmp_path, **mesh_overrides):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "advice.step")
        export_step(Sphere(radius=10), path)
        fp = CadFingerprint.from_step(path, mesh_deflection=1.0)
        fp.surface_mesh = dict(fp.surface_mesh, **mesh_overrides)
        return generate_test_file(fp, module_name="advice")

    def test_clustered_stl_is_told_to_raise_the_budget(self, tmp_path):
        source = self._source_for(
            tmp_path, remeshed=False, cluster_cell=0.9, facet_error=0.2,
        )
        assert "--max-mesh-triangles" in source
        assert "Export the STL more finely" not in source

    def test_faceted_stl_is_told_to_export_finer(self, tmp_path):
        source = self._source_for(
            tmp_path, remeshed=False, cluster_cell=0.0, facet_error=0.9,
        )
        assert "Export the STL more finely" in source
        assert "--stl-facet-error" in source


class TestStdoutSummary:
    """The default stdout dump stays readable."""

    def test_mesh_blob_is_summarised(self, tmp_path):
        import json
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.cli import _summarised_json

        path = str(tmp_path / "dump.step")
        export_step(Sphere(radius=10), path)   # thousands of triangles
        fp = CadFingerprint.from_step(path)

        summarised = json.loads(_summarised_json(fp))
        mesh = summarised["surface_mesh"]
        assert mesh["vertices"] is None
        assert mesh["truncated"] is True
        assert mesh["triangle_count"] == fp.surface_mesh["triangle_count"]
        assert len(_summarised_json(fp)) < len(fp.to_json()) / 2

    def test_truncated_mesh_will_not_decode(self, tmp_path):
        """It is valid JSON, so it must refuse to load rather than decode junk."""
        import json
        import pytest
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.analyze import decoded_mesh
        from b123d_decompiler.fingerprint.cli import _summarised_json

        path = str(tmp_path / "dump3.step")
        export_step(Sphere(radius=10), path)
        fp = CadFingerprint.from_step(path)

        reloaded = CadFingerprint(**json.loads(_summarised_json(fp)))
        with pytest.raises(ValueError, match="truncated"):
            decoded_mesh(reloaded.surface_mesh)

    def test_full_json_still_carries_the_mesh(self, tmp_path):
        import json
        from build123d import Box, export_step
        from b123d_decompiler.fingerprint import CadFingerprint

        path = str(tmp_path / "dump2.step")
        export_step(Box(10, 20, 30), path)
        fp = CadFingerprint.from_step(path)
        assert json.loads(fp.to_json())["surface_mesh"]["vertices"] == \
            fp.surface_mesh["vertices"]


class TestCompareMeshBudget:
    """compare caps the implementation mesh instead of meshing without limit."""

    def test_implementation_mesh_is_bounded(self, tmp_path, capsys):
        import argparse
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint.cli import _run_compare

        ref_path = str(tmp_path / "cmp_ref.step")
        impl_path = str(tmp_path / "cmp_impl.step")
        export_step(Sphere(radius=10), ref_path)
        export_step(Sphere(radius=10.02), impl_path)

        args = argparse.Namespace(
            ref_step=ref_path, impl_step=impl_path, axis="Z",
            cross_sections=6, radial_slices=4, angles=6,
            volume_tol=1.0, area_tol=2.0, bbox_tol=0.1, inertia_tol=2.0,
            xs_area_tol=3.0, xs_centroid_tol=0.2, xs_moment_tol=5.0,
            radial_tol=0.15, hausdorff_tol=0.3, hausdorff_mean_tol=0.05,
            hausdorff_samples=300, mesh_deflection=None,
            max_mesh_triangles=800, no_hausdorff=False,
            stl_facet_error=None,
        )

        from b123d_decompiler.fingerprint import fingerprint as fingerprint_module

        budgets = []
        original = fingerprint_module.CadFingerprint.from_step.__func__

        def recording(cls, path, **kwargs):
            budgets.append(kwargs.get("max_mesh_triangles"))
            return original(cls, path, **kwargs)

        fingerprint_module.CadFingerprint.from_step = classmethod(recording)
        try:
            _run_compare(args)
        except SystemExit as exit_code:
            assert exit_code.code == 1  # a failing metric, not a usage error
        finally:
            fingerprint_module.CadFingerprint.from_step = classmethod(original)

        out = capsys.readouterr().out
        assert "Surface Deviation" in out
        assert len(budgets) == 2, "reference and implementation were analysed"
        assert budgets[0] == 800                    # the reference's own budget
        assert budgets[1] == 20000                  # max(ref_count * 4, 20000)
        assert budgets[1] < 10 ** 9, "the implementation mesh must stay capped"


class TestCompareToleranceIsNotSelfReferential:
    """An implementation cannot widen the tolerance it is graded against."""

    @staticmethod
    def _pair(tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint

        ref_path = str(tmp_path / "sr_ref.step")
        impl_path = str(tmp_path / "sr_impl.step")
        export_step(Sphere(radius=10), ref_path)
        export_step(Sphere(radius=10.3), impl_path)
        return (CadFingerprint.from_step(ref_path),
                CadFingerprint.from_step(impl_path))

    def test_implementation_resolution_does_not_move_the_tolerance(self, tmp_path):
        """Its own measured chord error must not enter the floor.

        A part with more detail than the reference gets coarsened to fit the
        same triangle budget, which inflates that figure — and would excuse
        the very deviation being measured.
        """
        from b123d_decompiler.fingerprint.compare import compare_fingerprints

        ref, impl = self._pair(tmp_path)
        honest = compare_fingerprints(ref, impl, hausdorff_samples=400)

        impl.surface_mesh = dict(impl.surface_mesh, resolution=50.0)
        inflated = compare_fingerprints(ref, impl, hausdorff_samples=400)

        assert inflated["hausdorff"]["tolerance"] == honest["hausdorff"]["tolerance"]
        assert inflated["hausdorff"]["status"] == honest["hausdorff"]["status"]

    def test_floor_still_follows_the_deflection_actually_used(self, tmp_path):
        """Coarser meshing is a real loss of resolution, and must count."""
        from b123d_decompiler.fingerprint.compare import compare_fingerprints

        ref, impl = self._pair(tmp_path)
        fine = compare_fingerprints(ref, impl, hausdorff_samples=400)

        impl.surface_mesh = dict(
            impl.surface_mesh, deflection=impl.surface_mesh["deflection"] * 20
        )
        coarse = compare_fingerprints(ref, impl, hausdorff_samples=400)
        assert coarse["hausdorff"]["tolerance"] > fine["hausdorff"]["tolerance"]
        assert coarse["hausdorff"]["resolution_limited"] is True


class TestTruncatedMeshRefusesGeneration:
    """A display-truncated fingerprint must fail loudly, not at test time."""

    def test_generation_refuses_a_truncated_mesh(self, tmp_path):
        """Piping the stdout dump to a file and generating from it used to
        emit a test file that died inside zlib when run."""
        import json
        import pytest
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.cli import _summarised_json
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "trunc.step")
        export_step(Sphere(radius=10), path)
        fp = CadFingerprint.from_step(path)
        truncated = CadFingerprint(**json.loads(_summarised_json(fp)))

        with pytest.raises(ValueError, match="truncated"):
            generate_test_file(truncated, module_name="trunc")

    def test_full_fingerprint_still_generates(self, tmp_path):
        from build123d import Sphere, export_step
        from b123d_decompiler.fingerprint import CadFingerprint
        from b123d_decompiler.fingerprint.generate import generate_test_file

        path = str(tmp_path / "full.step")
        export_step(Sphere(radius=10), path)
        source = generate_test_file(CadFingerprint.from_step(path),
                                    module_name="full")
        assert "TestSurfaceDeviation" in source
