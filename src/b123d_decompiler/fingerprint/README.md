# cad-fingerprint

Generate pytest test suites from STEP or STL reference files. Any procedural
[build123d](https://github.com/gumyr/build123d) implementation that passes all
tests is geometrically equivalent to the reference for manufacturing purposes.

## What it does

`cad-fingerprint` analyses a CAD file and extracts a **geometric fingerprint**:
a set of measurements that together fully characterise the shape without
exposing the internal B-rep structure. It then generates a self-contained
pytest file that encodes those measurements as assertions.

The intended workflow is:

```
reference.step  ──►  cad-fingerprint  ──►  test_part.py
                                                │
procedural build123d code  ──►  pytest  ◄───────┘
                                   │
                              pass / fail
```

This lets you reverse-engineer a STEP file into clean parametric code
incrementally — run the tests, read the failures, adjust the code, repeat —
without needing to compare B-rep trees or CAD history.

## Fingerprint measurements

| Measurement | What it checks |
|---|---|
| Volume & surface area | Overall material quantity |
| Bounding box (min/max) | Overall envelope |
| Centre of mass | Mass distribution |
| Moments of inertia (6-component tensor) | Mass distribution in 3D — very sensitive |
| Face inventory | Surface types (Plane, Cylinder, Torus, BSpline …), counts, areas, and key dimensions (diameters, radii) |
| Edge inventory | Edge types (Line, Circle, BSpline …), counts, and key dimensions |
| Cross-sections | Area at N evenly-spaced planes along the primary axis |
| Radial profile | Outer radius at M axial positions × K angles |
| Surface deviation | Hausdorff distance — worst-case point-to-surface error, both directions |
| Build quality | Wall thickness, sharp edges, free edges, non-manifold geometry |

The first measurements are all *aggregates*: volume, area, inertia and
section areas can agree while the surface is locally wrong — a fillet in
the wrong place, a boss shifted 1 mm, a chamfer that became a round. The
Hausdorff distance is the complementary check. Both surfaces are triangulated
and sampled, and the distance from every sample point to the other surface is
measured in both directions:

```
forward   = max over points on the reference of distance(point, part surface)
backward  = max over points on the part of distance(point, reference surface)
hausdorff = max(forward, backward)
```

The generated test file embeds the reference triangle mesh (16-bit quantised,
zlib-compressed, base64-encoded — around 30–60 KB) so the tests stay
self-contained. Two assertions are generated: worst-case deviation
(`--hausdorff-tol`, default 0.3 mm) and mean deviation
(`--hausdorff-mean-tol`, default 0.05 mm). Pass `--no-hausdorff` to skip the
mesh capture and both tests.

Both are sampled estimates — 2000 points per direction by default, so a defect
confined to a few triangles can slip through; raise `--hausdorff-samples` to
sample harder.

Tolerances are also floored at what the mesh can actually resolve, so a
reference that had to be coarsened never asserts a tolerance a correct
implementation cannot meet. For a STEP reference the chord error is *measured*:
the shape is re-meshed five times finer and the stored mesh is measured against
that. A flat-faced part comes out exact however coarsely it was meshed; a
curved one does not. That figure plus the error of the mesh the part under test
will be built at, doubled, is the floor. When it binds, the generated file
carries a comment saying so and what to change. To tighten it, lower
`--mesh-deflection` (which tightens the angular limit with it) and raise
`--max-mesh-triangles`, at the cost of a bigger test file.

For an STL reference the facets are used as supplied — there is no analytical
surface to re-mesh — so `--mesh-deflection` becomes the vertex-clustering cell
used to shrink an over-large mesh (capped so a thin wall cannot collapse), and
the triangle budget is met by clustering rather than by re-meshing. A coarsely
exported STL is itself an approximation of whatever it was exported from, and
that error is estimated from the facets: each interior edge's turn angle over
the span perpendicular to it gives the arc those facets approximate. It adds to
the clustering displacement, since the two are independent. Beyond a point the
estimate cannot be made at all — a hexagonal prism and a six-facet cylinder are
the same mesh, and nothing in the file says which was meant — so the tool
prints what it read, warns when it could not read anything, and
`--stl-facet-error` lets you state the export tolerance outright.

An implementation with more detail than the reference hits the same triangle
ceiling and gets meshed coarser, so the floor scales with the deflection its
mesh ended up at. It scales the *reference's* measured error, never anything
read off the part under test — otherwise a part could widen the tolerance it is
judged by simply by carrying more detail. `compare` and the generated tests
apply the same rule.

For STL files, face type classification is unavailable (no analytical surface
information exists in the mesh); all other measurements work normally. The
radial profile uses direct Möller-Trumbore ray-triangle intersection (rather
than OCCT's `IntCurvesFace_ShapeIntersector`, which only works on analytical
BREP surfaces) and shoots rays from the bounding-box centre so parts that are
not centred on the world origin still produce meaningful results.

## Installation

Requires Python ≥ 3.10 and [build123d](https://github.com/gumyr/build123d).

```bash
pip install cad-fingerprint
```

For development:

```bash
git clone https://github.com/pzfreo/cad-fingerprint.git
cd cad-fingerprint
pip install -e ".[dev]"
```

## Usage

### Generate a test file

```bash
cad-fingerprint reference.step -o tests/test_reference.py
cad-fingerprint reference.stl  -o tests/test_reference.py
```

Options:

| Flag | Default | Description |
|---|---|---|
| `-o / --output` | — | Output pytest file |
| `--json` | — | Also save the raw fingerprint as JSON |
| `--prompt` | — | Also generate a PROMPT.md reverse-engineering guide |
| `--name` | filename stem | Human-readable part name used in test docstrings |
| `--fixture` | `part_under_test` | pytest fixture name |
| `--axis` | `Z` | Primary axis (`X`, `Y`, or `Z`) |
| `--cross-sections` | 20 | Number of cross-section slices |
| `--radial-slices` | 15 | Number of axial positions for radial profile |
| `--angles` | 12 | Angular samples per radial position |
| `--hausdorff-samples` | 2000 | Surface sample points per direction |
| `--mesh-deflection` | diagonal/1000 | Reference mesh resolution — deflection (STEP) or clustering cell (STL) |
| `--max-mesh-triangles` | 12000 | Triangle budget for the reference mesh |
| `--stl-facet-error` | estimated | Export tolerance an STL reference's facets were written at |
| `--no-hausdorff` | — | Skip the surface mesh and Hausdorff tests |

Tolerance flags (all have sensible defaults):
`--volume-tol`, `--area-tol`, `--bbox-tol`, `--inertia-tol`,
`--xs-area-tol`, `--xs-centroid-tol`, `--xs-moment-tol`, `--radial-tol`,
`--hausdorff-tol`, `--hausdorff-mean-tol`

### Run the generated tests

The generated file expects a `part_under_test` pytest fixture that returns a
build123d `Part`. Create a `conftest.py` alongside the test file:

```python
import pytest
from my_part import build_part

@pytest.fixture
def part_under_test():
    return build_part()
```

Then run:

```bash
pytest tests/test_reference.py -v
```

### Compare two STEP files directly

```bash
cad-fingerprint compare reference.step implementation.step
```

Produces a colour-coded terminal report (pass / close / fail per metric) and
exits non-zero if any metric fails.

### Python API

```python
from cad_fingerprint import CadFingerprint

fp = CadFingerprint.from_step("reference.step")
fp = CadFingerprint.from_stl("reference.stl")

fp.to_json("fingerprint.json")          # save
fp2 = CadFingerprint.from_json("fingerprint.json")  # load

from cad_fingerprint.compare import compare_fingerprints, format_comparison
result = compare_fingerprints(fp, fp2)
print(format_comparison(result))
print(result["hausdorff"])   # max / forward / backward / mean / rms / p95

# Or measure surface deviation on its own:
from cad_fingerprint.analyze import decoded_mesh
from cad_fingerprint.hausdorff import hausdorff_distance
h = hausdorff_distance(decoded_mesh(fp.surface_mesh),
                       decoded_mesh(fp2.surface_mesh))
print(f"{h['hausdorff']:.4f} mm worst case, {h['mean']:.4f} mm mean")
```

### Comparing without a CAD kernel

Reading STEP or STL needs build123d and OCC, but a saved JSON fingerprint
carries its own surface mesh and the distance code is pure Python. So the
names that need a CAD kernel are imported on first use, and comparing two
saved fingerprints — Hausdorff distance included — runs anywhere Python does:

```python
# no build123d installed
from cad_fingerprint import CadFingerprint
from cad_fingerprint.compare import compare_fingerprints, format_comparison

ref = CadFingerprint.from_json("reference.json")
impl = CadFingerprint.from_json("implementation.json")
print(format_comparison(compare_fingerprints(ref, impl)))
```

`from_step` / `from_stl` still raise `ModuleNotFoundError` without build123d —
they have a CAD file to read.

## Approach

The analyser uses [OpenCASCADE](https://www.opencascade.com/) (via build123d's
OCP bindings) directly:

- **Global properties** — `BRepGProp` for volume, surface area, centre of mass,
  and the full 3×3 inertia tensor.
- **Bounding box** — axis-aligned bounding box from `Bnd_Box`.
- **Face inventory** — walks all faces with `TopExp_Explorer`; classifies each
  surface type via `BRepAdaptor_Surface`; extracts diameter/radius for
  cylinders, cones, spheres, and tori.
- **Edge inventory** — same approach for edges via `BRepAdaptor_Curve`;
  extracts radius for circular edges and arc length for all types.
- **Cross-sections** — slices the solid at N planes along the primary axis
  using `BRepAlgoAPI_Section`; computes area, centroid, and second moments of
  each cross-section polygon.
- **Radial profile** — for STEP: shoots rays from the axis at M axial heights ×
  K angles using `IntCurvesFace_ShapeIntersector`; records the outermost
  intersection radius at each angle. For STL: uses Möller-Trumbore
  ray-triangle intersection from the bounding-box centre to handle parts not
  aligned with the world origin.
- **Surface deviation** — triangulates a *copy* of the shape with
  `BRepMesh_IncrementalMesh` (STL meshes are used as supplied, vertex-clustered
  if over budget), samples points over the surface area-weighted with a
  deterministic low-discrepancy sequence, and measures point-to-triangle
  distances through a uniform spatial grid. Mesh chord error — measured against
  a far finer re-mesh for STEP, estimated from facet fold angles for STL —
  bounds the tolerances. No RNG is involved and the input shape is never
  mutated, so the same input always yields the same numbers.
- **Build quality** — `ShapeAnalysis_FreeBounds` for free/non-manifold edges;
  `BRepCheck_Analyzer` for invalid geometry; minimum wall thickness via
  ray-sampling.

The generated pytest file embeds all reference values as plain Python
constants. It has no dependency on `cad-fingerprint` at runtime — only on
`build123d` and `OCP` (which build123d already requires).

## Example

The `examples/` directory contains a guitar tuning-peg head (`peghead7mm.step`)
with its generated test file (`test_peghead.py`), a reverse-engineered
procedural implementation (`peghead_procedural.py`), and a reverse-engineering
guide (`PROMPT.md`).

```bash
cd examples
pytest test_peghead.py -v
```

## License

MIT
