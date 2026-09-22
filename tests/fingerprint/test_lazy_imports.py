"""The pure-Python paths must not drag in a CAD kernel.

Reading a STEP or STL file needs build123d and OCC. Loading a saved
fingerprint and measuring surface deviation between two of them does not —
the fingerprint carries its own mesh, and the distance code is stdlib only.
These tests run each case in a subprocess with build123d and OCP blocked, so
an eager import creeping back into the package is caught immediately.
"""

import json
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

_BLOCK = """
import sys
# Poison the entries: any `import build123d` now raises ImportError, whether
# or not the real package is installed in this environment.
sys.modules["build123d"] = None
sys.modules["OCP"] = None
"""


def run_without_cad(body: str):
    """Run `body` in a subprocess where importing build123d fails."""
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run(
        [sys.executable, "-c", _BLOCK + body],
        capture_output=True, text=True, env=env,
    )


def check(body: str) -> str:
    result = run_without_cad(body)
    assert result.returncode == 0, (
        f"failed without build123d:\n{result.stdout}\n{result.stderr}"
    )
    return result.stdout


# ── what must work without a CAD kernel ──────────────────────────────

def test_package_imports():
    assert "ok" in check("import b123d_decompiler.fingerprint; print('ok')")


def test_hausdorff_module_imports():
    """The module is stdlib-only, but the package used to import OCC for it."""
    assert "ok" in check(
        "from b123d_decompiler.fingerprint.hausdorff import hausdorff_distance; print('ok')"
    )


def test_fingerprint_class_imports():
    assert "ok" in check("from b123d_decompiler.fingerprint import CadFingerprint; print('ok')")


def test_backwards_compatible_alias():
    assert "ok" in check(
        "from b123d_decompiler.fingerprint import StepFingerprint; print('ok')"
    )


def test_compare_module_imports():
    assert "ok" in check(
        "from b123d_decompiler.fingerprint.compare import compare_fingerprints; print('ok')"
    )


def test_two_saved_fingerprints_can_be_compared():
    """The payoff: a full surface-deviation comparison, no CAD kernel."""
    out = check("""
import json
from b123d_decompiler.fingerprint import CadFingerprint
from b123d_decompiler.fingerprint.compare import compare_fingerprints
from b123d_decompiler.fingerprint.hausdorff import encode_mesh


def cube(size, offset=0.0):
    # Shift along X only, so the expected distance is exactly the offset
    # rather than a corner diagonal the sampling may or may not land on.
    o = offset
    v = [
        (o, 0.0, 0.0), (o + size, 0.0, 0.0), (o + size, size, 0.0),
        (o, size, 0.0), (o, 0.0, size), (o + size, 0.0, size),
        (o + size, size, size), (o, size, size),
    ]
    t = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
        (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
    ]
    mesh = encode_mesh(v, t)
    mesh.update(deflection=0.01, candidate_resolution=0.0, resolution=0.0)
    return mesh


def fingerprint(mesh, volume):
    return CadFingerprint(
        file="synthetic", bounding_box={"min": (0, 0, 0), "max": (10, 10, 10),
                                        "size": (10, 10, 10)},
        volume_and_area={"volume": volume, "surface_area": 600.0,
                         "center_of_mass": (5.0, 5.0, 5.0)},
        moments_of_inertia={k: 1.0 for k in
                            ("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz")},
        topology={"faces": 12, "edges": 0, "vertices": 8},
        face_inventory=[], cross_sections=[], radial_profile=[],
        surface_mesh=mesh,
    )


ref = fingerprint(cube(10.0), 1000.0)
impl = fingerprint(cube(10.0, offset=0.5), 1000.0)
result = compare_fingerprints(ref, impl, hausdorff_samples=300)
print(json.dumps({"max": round(result["hausdorff"]["max"], 4),
                  "status": result["hausdorff"]["status"]}))
""")
    measured = json.loads(out.strip().splitlines()[-1])
    assert abs(measured["max"] - 0.5) < 0.01, measured
    assert measured["status"] == "fail"   # 0.5 mm against a 0.3 mm tolerance


def test_json_round_trip():
    """Saving and reloading a fingerprint needs no CAD kernel either."""
    assert "ok" in check("""
import os
import tempfile
from b123d_decompiler.fingerprint import CadFingerprint
from b123d_decompiler.fingerprint.hausdorff import encode_mesh

mesh = encode_mesh([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                   [(0, 1, 2)])
original = CadFingerprint(
    file="synthetic",
    bounding_box={"min": (0, 0, 0), "max": (1, 1, 1), "size": (1, 1, 1)},
    volume_and_area={"volume": 1.0, "surface_area": 6.0,
                     "center_of_mass": (0.5, 0.5, 0.5)},
    moments_of_inertia={}, topology={}, face_inventory=[],
    cross_sections=[], radial_profile=[], surface_mesh=mesh,
)
path = os.path.join(tempfile.mkdtemp(), "fp.json")
original.to_json(path)
reloaded = CadFingerprint.from_json(path)
assert reloaded.surface_mesh["triangle_count"] == 1
assert reloaded.volume_and_area["volume"] == 1.0
print("ok")
""")


# ── what must still require a CAD kernel ─────────────────────────────

def test_reading_a_step_file_still_needs_build123d():
    result = run_without_cad(
        "from b123d_decompiler.fingerprint import CadFingerprint;"
        "CadFingerprint.from_step('nonexistent.step')"
    )
    assert result.returncode != 0
    assert "build123d" in result.stderr


def test_analyze_is_reachable_when_the_kernel_is_present():
    """The lazy names must still resolve normally."""
    import b123d_decompiler.fingerprint

    assert b123d_decompiler.fingerprint.CadFingerprint.__name__ == "CadFingerprint"
    assert callable(b123d_decompiler.fingerprint.analyze_step)
    assert callable(b123d_decompiler.fingerprint.analyze_stl)
    assert b123d_decompiler.fingerprint.StepFingerprint is b123d_decompiler.fingerprint.CadFingerprint


def test_unknown_attribute_still_raises():
    import b123d_decompiler.fingerprint
    import pytest

    with pytest.raises(AttributeError, match="no attribute"):
        b123d_decompiler.fingerprint.does_not_exist


def test_dir_lists_the_public_names_and_submodules():
    import b123d_decompiler.fingerprint

    listed = set(dir(b123d_decompiler.fingerprint))
    assert {"CadFingerprint", "StepFingerprint", "analyze_step",
            "analyze_stl"} <= listed
    assert {"analyze", "compare", "generate", "hausdorff"} <= listed


def test_submodules_stay_reachable_as_attributes():
    """Eager imports used to bind these; lazy loading must not drop them.

    `import b123d_decompiler.fingerprint; b123d_decompiler.fingerprint.analyze.analyze_step(...)` worked
    before, so it has to keep working.
    """
    import b123d_decompiler.fingerprint

    assert b123d_decompiler.fingerprint.analyze.__name__ == "b123d_decompiler.fingerprint.analyze"
    assert b123d_decompiler.fingerprint.hausdorff.__name__ == "b123d_decompiler.fingerprint.hausdorff"
    assert callable(b123d_decompiler.fingerprint.compare.compare_fingerprints)
    assert callable(b123d_decompiler.fingerprint.generate.generate_test_file)


def test_pure_python_submodule_reachable_without_a_kernel():
    assert "ok" in check(
        "import b123d_decompiler.fingerprint;"
        "b123d_decompiler.fingerprint.hausdorff.hausdorff_distance;"
        "print('ok')"
    )


def test_cli_budget_default_matches_the_library():
    """Two copies of the default would drift silently."""
    from b123d_decompiler.fingerprint import analyze
    from b123d_decompiler.fingerprint.cli import main  # noqa: F401 — imports the module
    import argparse
    import b123d_decompiler.fingerprint.cli as cli

    parser = argparse.ArgumentParser()
    cli._add_analysis_args(parser)
    default = parser.get_default("max_mesh_triangles")
    assert default == analyze.DEFAULT_MAX_TRIANGLES
