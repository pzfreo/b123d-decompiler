# Baselines

Five corpora. Adapters were developed against **mfcadpp** and never run against
**mfcadpp_holdout** until they were finished, so the holdout says whether they generalise.
The **realistic** set is 28 parts assembled from four sources with nothing synthetic about
them. The **CADGenBench edit** set is half of that benchmark's 32-part editing split, held as
a target of IoU 0.9 for every part; the other half is kept unseen. Row-level numbers are in
`baselines/`.

```bash
uv run b123d-decompile batch path/to/corpus -o results/
uv run python tools/corpus_summary.py results/
```

## Result

Measured on 26 September 2026 against quiddity 0.3.5: MFCAD++, holdout and realistic at
e8859c0, the CADGenBench working half at e46d433 (one commit earlier; the only change since
checks faces the trim rays refuse the slow way, which changed no part on the other three
corpora for the worse). Medians are the ordinary median of the parts that scored.

| measure | mfcadpp (40) | holdout (33) | rotational (5) | realistic (28) | CADGenBench edit, working half (16) |
|---|---|---|---|---|---|
| parts that scored | 40 | 33 | 4 | 27 | 16 |
| IoU mean | 0.995 | 0.999 | 0.968 | 0.906 | 0.762 |
| IoU median | 0.999 | 1.000 | 0.985 | 0.971 | 0.836 |
| parts at IoU 0.9 or above | 40 | 33 | 4 | 21 | 5 |
| missing material, mean | 0.19 % | 0.03 % | 0.20 % | 0.17 % | 1.04 % |
| extra material, mean | 0.3 % | 0.1 % | 3.3 % | 16.7 % | 42.7 % |

The previous measurement, on 24 September against quiddity 0.3.3, was:

| measure | mfcadpp (40) | holdout (33) | rotational (5) | realistic (28) | CADGenBench edit, working half (16) |
|---|---|---|---|---|---|
| parts that scored | 40 | 33 | 4 | 24 | 14 |
| IoU mean | 0.969 | 0.955 | 0.956 | 0.713 | 0.469 |
| IoU median | 0.997 | 1.000 | 0.964 | 0.818 | 0.416 |
| parts at IoU 0.9 or above | 37 | 29 | 3 | 10 | 2 |
| extra material, mean | 3.4 % | 9.1 % | 4.7 % | 149 % | 782 % |

### What moved

No part scores worse than it did on 24 September. The gains come from:

- **Choosing the billet by carrying several through to a finished rebuild.** Outlines are
  drawn exactly in 2D (shapely) as lines and arcs, stepped billets are offered alongside
  shadows, and the trials keep whichever ends best.
- **Trims measured against the mesh.** Containment is counted on a triangle mesh rather than
  asked of the kernel point by point, trim depth is found by casting rays, and a face the
  rays refuse is measured by building its prism.
- **Parts drawn as what they are, from quiddity 0.3.5's body-level records.**
  - A **sheet-metal** part is its flanges extruded through the sheet and its bends as ring
    sectors: Too Tall Toby's sm-hanger went from stopped-after-three-hours to 1.000.
  - A **thin-walled** part with flat or cylindrical walls is its outer faces pushed in by the
    wall: cgb245 went from 0.138 to 0.901, NIST CTC 03 from 0.046 to 0.975.
  - A **thin-walled panel with free-form skins** is its plan outline cut back by the skins,
    written exactly as B-spline surfaces, less its cavity: cgb241 went from 0.011 to 0.825.

  These stocks are the part itself, so they are checked strictly: every cut is measured by a
  boolean, since sampling misses 2 mm walls, and no trims are applied. Across all 117 parts
  they were chosen only for those four.
- **Speed.** The part's bounding box is taken from its cached mesh, not asked of the kernel
  for every tool: 1.8 s a call on a thousand-face part had put cgb243 and cgb247 past the time
  limit. Both now finish, and cgb247 scores 0.910.

### Open problems

| part | corpus | now | why |
|---|---|---|---|
| cgb-threaded_connector_109 | rotational and realistic | not recognised | a quiddity 0.3.5 bug (pzfreo/quiddity#754), fixed on its main branch; 0.938 there |
| cgb207 | CADGenBench | 0.362 | 3 mm moulded shell whose walls meet through fillets quiddity does not pair |
| cgb203 | CADGenBench | 0.400 | impeller: seven swept blades; quiddity now reports the pattern, the decompiler has no way to build a blade |
| cgb249 | CADGenBench | 0.492 | volute casing; its walls are not of constant thickness, so none of the new routes applies |
| cgb243 | CADGenBench | 0.558 | not yet diagnosed |

## Realistic parts, by source

| source | parts | scored | median IoU | best |
|---|---|---|---|---|
| NIST CTC/FTC | 10 | 10 | 0.962 | 0.987 |
| build123d Too Tall Toby | 13 | 13 | 0.983 | 1.000 |
| CADGenBench | 2 | 1 | 0.900 | 0.900 |
| Gramel | 3 | 3 | 1.000 | 1.000 |

NIST was the weakest source on 24 September, at a median of 0.533; the mesh-counted trims
and ray-cast depths took it to 0.962. The one CADGenBench part not scored is the threaded
connector (see above).

## CADGenBench editing corpus

The 32 parts of `splits/editing32.txt` were sorted and split alternately. The working half is
the table below; the holdout half (202, 204, 206, 208, 211, 214, 217, 224, 229, 231, 240, 242,
244, 246, 248, 250) has not been run since the split. The goal for the working half is IoU
0.9 for every part.

| part | first run | 24 September | now | missing | extra | stock | surface quiddity associated |
|---|---|---|---|---|---|---|---|
| cgb215 | crashed | 0.9457 | 0.9463 | 0.40 % | 5.2 % | outline | 49 % |
| cgb209 | 0.4353 | 0.7746 | 0.9347 | 0.06 % | 6.9 % | outline | 60 % |
| cgb218 | 0.9024 | 0.9017 | 0.9322 | 0.04 % | 7.2 % | outline | 25 % |
| cgb247 | 0.3450 | 0.4103 | 0.9102 | 0.46 % | 9.4 % | outline | 41 % |
| cgb245 | 0.0674 | 0.1383 | 0.9008 | 5.21 % | 5.2 % | thin wall | 100 % |
| cgb230 | 0.1511 | 0.2332 | 0.8976 | 0.02 % | 11.4 % | outline | 53 % |
| cgb238 | 0.3198 | 0.5825 | 0.8526 | 1.48 % | 15.6 % | outline | 32 % |
| cgb205 | 0.4944 | 0.4219 | 0.8457 | 4.21 % | 13.3 % | outline | 56 % |
| cgb225 | 0.1691 | 0.7653 | 0.8270 | 0.27 % | 20.6 % | outline | 21 % |
| cgb241 | 0.0116 | 0.0113 | 0.8247 | 1.34 % | 19.6 % | thin panel | 100 % |
| cgb201 | crashed | 0.4621 | 0.7994 | 0.52 % | 24.4 % | outline | 40 % |
| cgb212 | 0.3985 | crashed | 0.7108 | 0.63 % | 39.8 % | outline | 48 % |
| cgb243 | crashed | 0.3782 | 0.5577 | 1.29 % | 77.0 % | outline | 59 % |
| cgb249 | 0.3414 | 0.3634 | 0.4916 | 0.25 % | 102.9 % | outline | 6 % |
| cgb203 | 0.1818 | 0.1845 | 0.3996 | 0.00 % | 150.3 % | outline | 30 % |
| cgb207 | crashed | crashed | 0.3621 | 0.40 % | 175.1 % | outline | 91 % |

Five parts meet the target, against two on 24 September, and all sixteen now score; the mean
went from 0.469 to 0.762. Four more are within 0.08 of it. What remains is still mostly
extra material, and the four worst have distinct causes, listed under "Open problems".

## Why realistic parts were hard (analysis of 24 September and earlier)

The error is overwhelmingly material left behind, not material wrongly removed: 149 %
extra against 1.1 % missing on the current run. The analysis below was made on the previous
run, when that was 591 % against 4.3 %, and the causes have not changed in kind.

**The bounding box is hopeless for these shapes.** A bracket or a hanger occupies a small
fraction of its envelope, and `ttt-23-02-02-sm-hanger` left 1257 % extra. Intersecting the
part's shadows along three axes now does for plates with formed features what the turned
billet does for rotational parts, and that is most of the gain since the last measurement.

**Blends are everywhere.** 91 fillet and 67 blend records go unmodelled on 28 parts, against
2 on 40 MFCAD++ parts. Real parts are mostly rounded, and each round this tool cannot place
is material it cannot remove.

**Trimming clears far less of the excess than it does on blocks.** Features alone leave
699 % extra on a realistic part and trimming brings that only to 599 %, against 28 % to 19 %
on a block. It is not that the faces are unusable: 61 % of unclaimed surface area is a flat
face with room in front that the trimmer does attempt, and nearly all of those are kept.
The trouble is that a prism from one face clears only the space directly in front of it, and
on a folded part the empty space wraps around rather than sitting in front of anything.

What is **not** the reason is recognition coverage, which was the obvious suspect and is
wrong. Quiddity names 62 % of a realistic part's surface area against 36 % of a benchmark
block's, and the decompiler places the same median of 6 feature operations per part on all
three corpora. Realistic parts are better recognised, not worse.

The difference is what the shape is made of. A benchmark block *is* its bounding box with a
few things cut out of it, so envelope stock plus six features is nearly the whole answer. A
bracket's shape is its outer form, and no feature in the vocabulary describes outer form. The
recogniser can name every face of it and the decompiler still has nothing to build from.

## What changed to get here

| measure | 5 adapters | 12 adapters | plus stock trimming |
|---|---|---|---|
| mfcadpp IoU median | 0.783 | 0.821 | 0.968 |
| holdout IoU median | 0.789 | 0.789 | 0.951 |
| mfcadpp parts at 0.9+ | 7 | 14 | 24 |

Trimming the stock back to faces no feature claimed was the single biggest step, and it is
a different kind of evidence from the rest: it reads faces the recogniser could not name
rather than rebuilding features it did name.

## Tracing, measured on its own

Trimming the stock back to unnamed faces is the half of the result with no design intent in
it, so the tool can run with `--no-trim` and `tools/corpus_summary.py` compares two runs.
This comparison has not been rerun since the September changes. Over the 26 realistic parts
both runs scored at the time:

| | features only | plus tracing |
|---|---|---|
| IoU median | 0.315 | 0.409 |
| extra material, mean | 699 % | 599 % |
| missing material, mean | 0.99 % | 1.00 % |

That comparison paid for itself immediately. The first time it ran it showed tracing raising
mean missing material from 1.0 % to 4.3 % and dragging one NIST part from 0.189 to 0.028.
Each trim had passed the per-op test, but "removes only empty space" means within a
tolerance, and forty-two of them each sat just inside the limit and between them took most of
the part. Speculative cuts now share a budget of half a percent of the part, cheapest first.
Missing material went back to 1.0 %, no part regresses from tracing any more, and the
benchmark corpus did not move at all, because there the trims genuinely do prove empty.

## Measurement, defended

Seven failures found and fixed, each of which had been quietly producing wrong numbers.

**Intersections that fail silently.** A rebuild's faces coincide with the reference's, which
is the input OCCT gets wrong. It returned empty, or a sliver of 1e-14, while reporting
success. One holdout part scored 0.0 and read as 100 % missing when the true figure was
0.845 and 2 %. Intersections now retry with a fuzzy tolerance, and an answer too small to
believe is replaced by an estimate from 20,000 classified sample points.

**A safety test defeated by a big tool.** Cuts were checked for eating real material as a
fraction of their own volume. An open profile closed by running its ends outward makes a
tool hundreds of times larger than the cut, which passed the relative test while removing a
real piece of the part. The check is now relative *and* absolute against the part.

**Booleans that blow up.** A 21 mm³ trim deleted a 13,000 mm³ part, and the sequence check
did not notice. A cut cannot remove more than its own tool contains, and that invariant now
holds; the part went from 0.0 to 0.948.

**A kernel crash taking the batch with it.** One NIST part segfaults OCCT outright. Each
part now runs in its own process, so that is one row saying `crashed` instead of a corpus
run dying at part 20.

**Measuring the part changed the part.** OCCT booleans are allowed to widen the tolerances of
the shapes they are given, in place. After a few hundred intersections against the reference,
some with a fuzzy retry, its vertex tolerance had grown several times over, the classifier
called every nearby point part of it, and intersections came back empty. Once point counting
was added to catch failed intersections, this made the count read every trim on cgb225 as
solid material, and 94 of 96 trims that were genuinely empty were refused. Intersections are
now non-destructive and the stock is sliced from a copy.

**An intersection that fails in both directions.** On a tool whose wall lies on a face of the
part, the intersection sometimes finds nothing where the tool plainly cuts material, and
sometimes returns the whole tool where it is almost entirely clear. Reading a failed
intersection as zero had been passing destructive trims as proved: on cgb225, six trims that
between them destroyed a third of the part. Every reading is now
cross-checked by counting points: the count wins where they disagree, and the larger wins
where they agree. On cgb225 the two fixes together took trims proved empty from 2 of 96 to
71 and the IoU from 0.520 to 0.765, with no material missing.

## Ground truth

None of the corpora has ground truth, so those are relative numbers. The absolute check is
`tests/test_roundtrip.py`, which builds parts from known features, exports them and
decompiles them back. Eight fixtures rebuild at IoU 1.0000: a through hole, a blind hole, a
counterbored hole, a rectangular pocket, a through step, a chamfered edge, a rounded corner
and a quarter-round corner cut.
