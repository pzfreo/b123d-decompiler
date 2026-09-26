# b123d-decompiler

A STEP to Python decompiler. It recognises features with
[quiddity](https://github.com/pzfreo/quiddity), writes build123d source that rebuilds the
part, runs that source, and measures how close the rebuild is to the original.

```bash
uv sync
uv run b123d-decompile part.step --analyse
```

`PLAN.md` holds the full design. This file describes what is built.

## Pipeline

```
part.step ──► recognise   quiddity document: local frame, records, faces, association
          ──► plan        stock + ordered ops, each op validated against the reference
          ──► emit        part.py, standalone build123d algebra mode
          ──► execute     run part.py in a subprocess ──► part.rebuilt.step
          ──► compare     IoU, missing/extra material, Hausdorff, centre of mass
```

Every stage leaves a file behind, so a poor score can be traced to the stage that caused
it rather than to "the pipeline".

## Commands

```bash
b123d-decompile part.step                    # writes part.py beside the input
b123d-decompile part.step -o out.py --plan plan.json --analyse
b123d-decompile analyse ref.step out.py      # executes out.py, compares, prints a table
b123d-decompile analyse ref.step other.step --json report.json
b123d-decompile batch corpus/ -o results/    # per-part files plus results/summary.csv
```

`analyse` takes either a `.py` to execute or a `.step` to read, so it scores hand-written
and model-written reconstructions too, not only its own.

## What the generated code looks like

```python
"""build123d reconstruction of 205.step.

Decompiled by b123d-decompiler from quiddity feature recognition.
2 operations from 3 recognised features.
Quiddity associated 47% of the surface area with a feature.
"""

from build123d import *

# stock: bounding box 16.518 x 41.262 x 48.938
part = Pos(0.3929, 6.9728, -0.695) * Box(16.5177, 41.2615, 48.9385)

# section_recesses #4: pocket: triangular section, run x -11.171..-0.908
_prof = (
    Line((-3.816, 7.535), (-3.584, -6.078))
    + Line((-3.584, -6.078), (7.4, -1.458))
    + Line((7.4, -1.458), (-3.816, 7.535))
)
_plane = Plane(origin=(-11.1713, 1.702, -3.257), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
tool = extrude(_plane * make_face(_prof), amount=10.2633)
part -= tool

# back to the coordinate system of the source file
part = Plane(origin=(25.1642, 13.6579, 8.6518), x_dir=(0, 0, -1), z_dir=(1, 0, 0)) * part
```

Modelling happens in quiddity's inferred local frame, where the numbers are small and the
axes are aligned, and the last line maps the result back to the coordinates of the source
file. `--keep-frame` leaves it local.

## Stock

Quiddity publishes no base-body record, so the solid the part is cut from has to be inferred.
Three candidates, and whichever is smallest while still holding the whole part wins.

**The part's own outline.** Most machined parts are cut from plate or bar, so their silhouette
along some axis *is* the billet. It is built by slicing where the geometry changes, taking the
cross section in every slice, and running each one the full length of the axis; a slab's
section cannot change between two consecutive vertex positions on a prismatic part, so that is
exact there, and long slices are split so a taper loses little.

A part that is a plate with something formed into it has a shadow far bigger than itself along
every axis, because the formed region smears the whole length of the sweep. So the shadows
along all three axes are built and intersected, keeping an axis only if it takes away at least
2 %. Each shadow holds the part, so the intersection does too. The emitted source is run and
checked again, because curves written out as chords can fall inside the part. The search gives
up after 90 seconds an axis and four minutes in all.

**Bar turned to a stepped diameter**, from a turned profile, which is the one quiddity family
that describes a whole billet rather than a region of the body.

**The bounding box**, when neither of the others holds at least 99.5 % of the part. A bent or
swept part has no prismatic silhouette, and the envelope is never too small, so a miss shows up
as material left behind rather than as a hole in the part.

Stock was the single biggest thing in this tool. Moving from the bounding box to the outline
took the development corpus from a median IoU of 0.968 to 0.988 and its leftover material from
18.5 % to 4.2 %.

## Families modelled

| family | operation | notes |
|---|---|---|
| `section_recesses` | profile extruded along the record's run axis | lines and arcs; an open profile is closed both ways and the better one kept |
| `holes` | cylinder from the opening, with counterbore, spotface or countersink | through bores run clear of the envelope |
| `through_steps` | rectangular corner prism | the record's removed-prism midpoint fixes which corner went |
| `circular_blind_steps` | cylinder trimmed to the corner quadrant | trimming avoids having to know the sweep direction |
| `angled_steps` | triangular wedge on a corner, blind at one end | the corner comes from the record |
| `paired_ramp_steps` | V notch from the ridge out to the envelope | width and open end come from the record |
| `chamfers` | the wedge the chamfer removes | the corner comes from the record |
| `fillets` | the sliver behind the arc, or the quarter cylinder | whichever the reference shows went |
| `blends` | the same sliver, cut when convex and fused back when concave | the record states its own sense |
| `slots` | box, obround or rounded rectangle | end and corner radii are modelled |
| `bosses`, `pads` | fused cylinder or box | kept only where they restore material a cut removed |
| `step_levels` | the space past a horizontal face | speculative, see below |
| unclaimed faces | the space in front of a flat face no feature named | speculative, see below |

`risers`, `plates` and `countersinks` are evidence for other stages rather than operations
of their own. Every other family is counted as skipped and named in the generated file.

### Two kinds of evidence

Most of the table rebuilds a feature the recogniser named. The last two rows do not. A face
level says a flat face sits at this height over this patch; an unclaimed face says only that
material ends here. Neither says what was removed to leave it there, and the obvious reading
is wrong wherever something stands on the face. So those ops are marked **speculative** and
kept only if they prove they remove empty space alone; the rest are reported.

Trimming the stock back to unclaimed faces is what closes the gap on a part whose bulk outer
form is published as no feature at all. It is also the part of the result with no design
intent in it, so it is worth measuring on its own rather than letting one number hide which
half moved:

```bash
b123d-decompile batch corpus/ -o with-tracing/
b123d-decompile batch corpus/ -o features-only/ --no-trim
python tools/corpus_summary.py features-only/ with-tracing/
```

The comparison prints the difference over the parts both runs scored, and names the parts
that moved most in each direction. It is how the cumulative trimming budget below was found.

Speculative cuts also share a budget. Each one proves on its own that it removes only empty
space, but "only" means within a tolerance, and on one NIST part forty-two of them each sat
just inside the per-op limit and between them took most of the part away. The cheapest cuts
are taken first and the rest are dropped once a fixed share of the part has been spent.

## Validation, not hope

Each op is built by executing its own generated source, which is the same text the script
gets, so a plan that validates and a script that runs cannot disagree. The tool is then
measured against the reference, and the whole sequence is run in order, because an op can be
sound on its own and still ruin the part when applied after the others.

- a **cut** whose tool eats material the reference keeps is marked `overlaps`, and the
  amount is written into the script as a comment. The test is relative to the tool *and*
  absolute against the part, because a deliberately oversized tool passes a relative test
  while cutting a real piece away;
- a **speculative** op that cannot prove itself is dropped rather than flagged;
- a **fuse** already inside the stock that meets no cut is `inert`;
- a fuse that meets a cut is kept only when the shared region is material in the reference,
  which is how a pin standing in a bore survives the bore being drilled;
- an op that moves more material than its own tool contains has hit a boolean failure, not a
  geometric truth, and is dropped with the figures named.

The comparison is defended the same way. A rebuild's faces coincide with the reference's by
construction, which is exactly the input OCCT gets wrong: the intersection can come back
empty, or as a sliver of 1e-14, while reporting success. The intersection retries with a
fuzzy tolerance, and where the answer is still not believable the overlap is estimated by
classifying sample points instead, which the kernel can still do reliably. A comparison that
cannot be made at all is reported as not measurable, never as zero.

## Known limits

- **Stock is the bounding box.** Quiddity publishes no base-body record. A tighter convex
  stock was tried and measured: intersecting the half-spaces the whole part already lies
  behind reproduces the bounding box exactly on all 40 development parts, because these
  envelopes are already convex. What is missing is concave removal, not a smaller billet.
- **Additive features can only restore.** A boss lies inside its own bounding box, so it
  adds nothing unless a cut took its material away first.
- **Some records describe shapes these adapters do not.** A fillet whose two supporting
  planes are not a plain convex corner, an angled step whose slant centre fits no corner,
  and an open recess profile whose straight closing chord crosses itself are all reported
  unmodelled rather than placed somewhere plausible. A missing cut leaves material behind;
  a wrong one removes material the part keeps, which is worse.
- **Sloped and cylindrical recess end caps are modelled flat**, and slot end radii are cut
  square. Both are noted on the op and in the generated file.
- Quiddity recognises analytic surfaces, so B-spline faces stay unassociated and their
  material is left behind.

## Layout

```
src/b123d_decompiler/
  recognise.py  STEP to quiddity document plus the local working shape
  adapters.py   one function per quiddity family
  plan.py       stock, ordering and per-op validation
  model.py      Op and BuildPlan, JSON in and out
  emit.py       plan to build123d source
  execute.py    run the script, bring back the solid
  compare.py    the metrics
  geom.py       measurement, point probing, robust booleans
  pipeline.py   the stages joined up, and the batch summary
  cli.py        decompile, analyse, batch
  fingerprint/  vendored cad-fingerprint, maintained here
```

`b123d_decompiler.fingerprint` is the vendored
[cad-fingerprint](https://github.com/pzfreo/cad-fingerprint) package, which supplies the
meshing and Hausdorff measurement. The `cad-fingerprint` command still works; see
[its README](src/b123d_decompiler/fingerprint/README.md).

## Tests

```bash
uv run pytest                      # everything
uv run pytest tests/test_units.py  # no CAD kernel needed
```

`tests/test_roundtrip.py` builds parts from known features, exports them, decompiles them
and asserts the rebuild matches. Those are the only tests with ground truth, so a failure
there is a real regression in an adapter.

## Where it stands

Measured on 26 September 2026 against quiddity 0.3.5.

| | mfcadpp (40) | holdout (33) | rotational (5) | realistic (28) | CADGenBench edit, working half (16) |
|---|---|---|---|---|---|
| IoU median | 0.999 | 1.000 | 0.985 | 0.971 | 0.836 |
| parts at IoU 0.9 or above | 40 | 33 | 4 | 21 | 5 |
| missing material, mean | 0.19 % | 0.03 % | 0.20 % | 0.17 % | 1.04 % |
| extra material, mean | 0.3 % | 0.1 % | 3.3 % | 16.7 % | 42.7 % |
| did not score | 0 | 0 | 1 | 1 | 0 |

The first two are MFCAD++ benchmark blocks; adapters were developed against the first and
never run against the second until they were finished, and the close medians say they
generalise. The rotational five from Gramel and CADGenBench exercise the turned billet and the
swept chamfers that nothing in MFCAD++ touches. The realistic set is 28 parts from NIST,
build123d's Too Tall Toby set, CADGenBench and Gramel. The one part that did not score there
fails recognition on quiddity 0.3.5, a bug fixed on quiddity's main branch.

The last column is the hardest. It is half of the CADGenBench editing corpus, held back as a
target of IoU 0.9 for every part, with the other half kept unseen. Sheet-metal, thin-walled
and free-form panel parts are now drawn from quiddity's body-level records rather than cut
from a billet; what remains is mostly material left behind on moulded shells, an impeller and
a volute. Full numbers and the reasons in `RESULTS.md`.
