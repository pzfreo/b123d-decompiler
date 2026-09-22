# b123d-decompiler — plan

Turn a STEP file into build123d source that rebuilds it, using
[quiddity](https://github.com/pzfreo/quiddity) for feature recognition, and measure how
close the rebuild is with an analysis tool built on the metrics already in
`cad-fingerprint`.

## Pipeline

```
part.step ──► quiddity recognition document (local frame, records, faces, association)
          ──► build plan   (stock + ordered feature operations, JSON, frame-local)
          ──► emit         part.py  (build123d algebra mode, comments name each feature)
          ──► execute      part.py in a subprocess ──► part.rebuilt.step
          ──► analyse      reference vs rebuilt ──► report.json + terminal summary
```

Every stage is a file on disk so any one of them can be inspected, hand-edited and re-run.
The build plan is the contract between recognition and code generation: it is what lets us
say *which* stage lost fidelity.

## Package layout

```
src/b123d_decompiler/
  fingerprint/   vendored cad-fingerprint (analyze, hausdorff, compare, fingerprint, generate, cli)
  recognise.py   load STEP (quiddity.import_step_geometry), build_recognition_document()
  plan.py        document ──► BuildPlan (stock inference, family adapters, ordering)
  model.py       BuildPlan / Stock / Op dataclasses, JSON in/out
  emit.py        BuildPlan ──► build123d source text
  execute.py     run generated source in a subprocess, export STEP, capture errors
  compare.py     metrics (thin wrapper over fingerprint.analyze / hausdorff + booleans)
  residual.py    boolean residuals (missing / extra material) and re-recognition of them
  batch.py       run a corpus, write per-part JSON and a CSV leaderboard
  cli.py         b123d-decompile {decompile, analyse, batch}
tests/
  test_roundtrip.py   build123d ──► STEP ──► decompile ──► metrics within tolerance
  test_plan.py        document fixtures ──► expected ops
  corpus/             symlink or copy of quiddity's mfcadpp / cadgenbench fixtures
```

Dependencies: `quiddity>=0.3.2` and `build123d`. `cad-fingerprint` is vendored as
`b123d_decompiler.fingerprint` (from pzfreo/cad-fingerprint commit 20c0eec, v0.3.0) and
maintained here; its tests live in `tests/fingerprint/` and its `cad-fingerprint` console
script still works. No new geometry code where it already has it: Hausdorff, inertia,
cross-sections, meshing.

## CLI

```
b123d-decompile part.step                 # writes part.py beside the input
b123d-decompile part.step -o out.py --plan plan.json --keep-frame
b123d-decompile analyse ref.step part.py  # executes part.py, compares, prints table
b123d-decompile analyse ref.step other.step --json report.json
b123d-decompile batch corpus/ -o results/ # per-part reports + results/summary.csv
```

`analyse` accepts either a `.py` (executed) or a `.step` (compared directly), so it also
scores hand-written or LLM-written reconstructions, not only our own.

## Stage 1: recognition

Use `quiddity.document.build_recognition_document(import_step_geometry(path))`. This gives
the framed result (records in an inferred local frame), the face roster, defining and
constituent faces per feature, and the association ratio. We keep the `frame` and use it
twice: to emit code in the local frame (cleaner numbers, axis-aligned), and to transform the
rebuilt solid back to caller coordinates before comparison. `--keep-frame` leaves the output
in the local frame when the user prefers a tidy model over a coordinate-exact one.

Also retain `association.unassociated_faces`. Faces quiddity did not claim are the first
place to look when a rebuild is wrong, and the report lists them by surface type and area.

## Stage 2: build plan

`BuildPlan = Stock + ordered list of Op`. Each `Op` carries the family, the source feature
index (so the report can trace a bad op to its record), and parameters already resolved
to build123d terms (a `Plane`, a size, a direction).

### Stock inference

Quiddity has no "base body" record, so stock is inferred, in this priority order:

| evidence | stock |
|---|---|
| `derived.turned_profiles` covering the body bounds | revolve a stepped profile about the axis |
| `PlanarOuterProfile` (body-owned outer loop of lines/arcs) | extrude the profile along its normal through the body extent |
| `polygonal_stock` | extrude the regular polygon |
| `plates` (axis, lo, hi, u, v) | `Box` aligned to the plate axis |
| nothing | `Box` of the local-frame bounding box |

Stock is always **at least** the bounding box: anything the rebuild is missing shows as
"missing material" in the residual rather than as extra, which is easier to diagnose.

### Family adapters

Each quiddity family gets an explicit adapter (the same pattern quid2pmi uses in
`adapters.py`). A family without an adapter is reported as *skipped*, never silently dropped.
First-cut mapping:

| family | build123d op |
|---|---|
| `holes` (with cbore / csink / spotface, blind flat or conical bottom) | `Hole` / `CounterBoreHole` / `CounterSinkHole` at `location`, along `axis`, `depth` or through |
| `hole_patterns` (bolt circle, linear, grid) | same holes, emitted as `PolarLocations` / `GridLocations` loop for readable code (second pass; geometry identical) |
| `section_recesses` | polyline+bulge profile in the record's `frame` (u, v) ──► `Face` ──► extrude along `run` over `run_interval`; `open` ends extend past stock, `capped` ends stop at the interval |
| `slots`, `oriented_slots`, `slot_patterns` | `SlotOverall` or rounded-rect sketch extruded across `d_lo..d_hi` |
| `grooves` | annular cut: `Cylinder(OD) - Cylinder(groove diameter)` over width, on the turned axis |
| `flats` | half-space cut (`Box` positioned outside the flat plane) |
| `through_steps`, `circular_blind_steps`, `rectangular_blind_slots`, `angled_steps`, `paired_ramp_steps` | box / wedge cuts from the spans in the record |
| `bosses` | `Cylinder` fused at `location` along `axis` with `height` |
| `polygonal_bosses`, `pads` | extruded polygon / box fused between the two levels |
| `gusset_ribs` | triangular prism fused |
| `chamfers` | `chamfer(edges nearest record.at along record.axis, length)` |
| `fillets`, `blends` | `fillet(edges nearest record.at along record.axis, radius)` |
| `step_levels`, `risers`, `plates`, `countersinks` (standalone) | evidence only; consumed by stock inference and hole adapters, not emitted |
| `section_recess_refusals` | reported as *refused* with the reason; nothing emitted |

Edge selection for fillets and chamfers is the fragile part. The record gives an axis and a
point; the emitted code selects `part.edges().filter_by(Axis.<axis>)` then sorts by distance
to the point and takes those within tolerance. That produces correct geometry but ugly code;
a later pass can replace it with `edges().group_by()` selectors when the edge lies on a known
level.

### Ordering

1. stock
2. subtractive features, largest volume first (section recesses, slots, steps, grooves, flats, holes)
3. additive features that put material back (pads, bosses, polygonal bosses, ribs)
4. blends last, fillets before chamfers, sorted by descending radius

Additive ops come **after** the cuts, not before. Under bounding-box stock an additive
feature already lies inside the stock, so the only one that can change anything is one
restoring material a cut removed: a pin standing in a bore is drilled away by the bore and
has to come back. Quiddity reports evidence, not history, so there is no recorded order to
recover and this one is what the stock demands. The case that wants the opposite order, a
hole drilled through a boss, is detected instead of guessed: if the region the two tools
share is open space in the reference, the fuse would refill a real void and is dropped.

Blends go last because every earlier op changes the edge set.

## Stage 3: emit

build123d **algebra mode** (`Box`, `Cylinder`, `Pos`, `Rot`, `+`, `-`): it maps one-to-one
onto the plan, is what quiddity's own docs use, and does not need context managers around
every cut. Output is a self-contained script:

```python
from build123d import *

# stock: plate, z from 0 to 20, 80 x 50  (quiddity plate #0)
part = Box(80, 50, 20)
# section recess #3: pocket, triangular section, run -7.866..-0.908 along x
_prof = Polyline(...)   # closed, in recess frame
part -= extrude(make_face(_prof), amount=...)
# hole #0: Ø2.0 x 6.41 flat-bottomed, axis +x
part -= Pos(...) * Rot(...) * Cylinder(1.0, 6.41, align=...)
...
if __name__ == "__main__":
    export_step(part, "part.rebuilt.step")
```

Numbers are rounded to the recogniser's tolerance (`gauge`), not printed at full float
precision, so the code reads like something a person wrote. The emitter also writes the plan
JSON when `--plan` is given.

## Stage 4: execute

Run the script in a subprocess with a timeout and capture stdout/stderr. Outcomes: `ok`,
`syntax_error`, `runtime_error` (with traceback and the op comment nearest the failing line),
`invalid_solid` (`Solid.is_valid()` false or zero volume), `timeout`. Any non-`ok` outcome
still produces a report; the batch table needs the failure class, not just a blank.

## Stage 5: analyse

`compare.py` wraps cad-fingerprint and adds the boolean metrics it does not have. Report per
part:

| group | metric | source |
|---|---|---|
| gate | executed, valid solid, solid count | execute.py |
| bulk | volume, surface area, bounding box (per axis) | fingerprint.analyze |
| mass | centre of mass offset (mm), inertia tensor (6 terms) | fingerprint.analyze |
| surface | Hausdorff max, mean, RMS, p95, both directions | fingerprint.hausdorff |
| boolean | IoU = vol(A∩B) / vol(A∪B); missing = vol(ref − rebuilt); extra = vol(rebuilt − ref) | OCP booleans in compare.py |
| topology | face and edge type counts, deltas | fingerprint.analyze |
| coverage | quiddity association ratio; ops emitted / skipped / refused per family | plan.py |

IoU is the headline number: it is what CADGenBench-style leaderboards use, it is symmetric,
and it is insensitive to sampling. Hausdorff is the complementary "worst local error" number.
Centre-of-mass and inertia catch a feature placed on the wrong side of a symmetric part,
which IoU alone under-reports on small features.

Comparison runs in caller coordinates after applying the frame transform, so a frame error
shows up as a bounding-box shift rather than being hidden.

### Residual diagnosis

`residual.py` computes the two boolean residuals as solids, and for each connected residual
lump reports its volume, bounding box and nearest source op. It then runs quiddity again on
the *missing* residual (as a solid) to name what was not rebuilt: "missing: 2 holes Ø6,
1 slot". That closes the loop from a bad number to the family adapter that needs work.

## Batch harness

`batch` runs the whole pipeline over a directory, writing `results/<stem>/{plan.json,
part.py, rebuilt.step, report.json}` and one `summary.csv` row per part (gate, IoU,
Hausdorff, COM offset, coverage, skipped families, failure class). A small script
aggregates per-family skip and refusal counts across the corpus so the next adapter to write
is the one that moves the most parts.

Corpora, in order of use:

1. **Synthetic round-trip** (in `tests/`): parts built in build123d with known features, exported
   to STEP. Ground truth is known, so tests assert IoU > 0.999 and Hausdorff below mesh
   resolution per family. Start with one fixture per adapter.
2. **quiddity `tests/corpus/mfcadpp*`** (74 STEP files): realistic prismatic parts, no ground
   truth, tracked as a leaderboard.
3. **quiddity `tests/corpus/cadgenbench` / `gramel` / `nist`**: harder; used for the skip and
   refusal census, not for pass/fail.

## Status

Requires quiddity 0.3.3 or newer, which publishes the material-side facts this tool used to
recover by probing the reference solid. Those fields exist because this work asked for them:
quiddity issue 735 and pull request 736.

Built and measured on three corpora, 40 development parts, 33 holdout parts and 28
realistic parts from NIST, build123d's Too Tall Toby set, CADGenBench and Gramel: recognise,
plan, emit, execute, compare, batch, with twelve family adapters plus stock trimming.
Benchmark medians of 0.968 and 0.951 say the adapters generalise. The realistic median is
0.386, and that gap is the honest state of the work: bounding-box stock suits a block and
not a bracket, real parts are mostly blends, and most of their unrecognised faces are
curved. See `README.md` for what the tool does and `RESULTS.md` for the numbers.

Two things the prototype learned that were not in this plan:

- **Per-op validation earns its place.** Executing each op's own generated source and
  measuring the tool against the reference caught a wrong hole direction and a dropped
  pin, and it is the reason a bad number points at one adapter.
- **The comparison boolean needs a retry.** A rebuild's faces coincide with the
  reference's by construction, and on that input OCCT can report success with an empty
  intersection. Taken at face value that scored a good reconstruction as zero. The
  intersection now retries with a fuzzy tolerance and says when it had to.

## Milestones

Each milestone ends with the batch harness giving a number on the mfcadpp corpus.

- **M0 scaffold + baseline.** Package, CLI, recognise, bbox-only stock, execute, analyse,
  batch. The "stock only" IoU per part is the baseline every adapter is measured against.
- **M1 the step families. Done.** `through_steps`, `circular_blind_steps`, `angled_steps` and
  `paired_ramp_steps`: the concave corner and notch cuts that turn a block into a bracket.
  Measured as the first priority, not assumed. 35 of the 40 corpus parts remove less than half
  the material they should, and 24 of those have a skipped step-family record. See
  `RESULTS.md`.
- **M1b stock inference.** Deferred, and not for lack of time: a stock built from the
  half-spaces the whole part lies behind comes out identical to the bounding box on all 40
  parts, because these envelopes are already convex. Revisit it for turned and profiled stock,
  where it should pay.
- **M2 subtractive adapters.** Holes (all variants), section recesses, slots, grooves, flats,
  steps. This is where most of the mfcadpp corpus is.
- **M3 additive adapters.** Pads, bosses, polygonal bosses, ribs.
- **M4 blends. Done**, though not as planned: chamfers and fillets are cut as the wedge and
  the sliver they remove, rather than selected with build123d's `chamfer()` and `fillet()`.
  Edge selection needs every earlier op to have reproduced the edge exactly; a wedge needs
  only the record.
- **M5 diagnosis and code quality.** Residual re-recognition, pattern loops
  (`PolarLocations` / `GridLocations`), level-based edge selectors, rounding.

## Known limits, stated up front

- Quiddity only recognises analytic surfaces; B-spline faces are unassociated and will show
  as missing material. The report says so rather than guessing.
- Quiddity reports evidence, not history: two records can describe one void, and reconciliation
  already picks one, but the decompiler still has to cope with overlapping cuts (harmless for
  subtraction, wrong for addition). Additive ops are fused only where their volume does not
  already intersect the stock beyond tolerance.
- Fillets and chamfers on edges that the earlier ops did not reproduce exactly will fail to
  select; the executor reports the op, and the part is still scored without the blend.
- The rebuilt code is correct before it is idiomatic. Readability passes come in M5, after
  the geometry numbers are stable.

## Decisions taken

- Algebra mode, not builder mode.
- Local-frame code by default, transformed back for comparison; `--keep-frame` to opt out.
- IoU as the headline metric, Hausdorff and COM as required secondaries.
- Vendor and maintain cad-fingerprint here rather than depend on it or reimplement it.
- One explicit adapter per family, skipped families reported, never dropped.
