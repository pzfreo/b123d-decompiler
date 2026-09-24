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

Measured on 24 September 2026 against quiddity 0.3.3. Medians are the ordinary median of the
parts that scored, so they differ slightly from the batch printout, which takes the upper of
the two middle values.

| measure | mfcadpp (40) | holdout (33) | rotational (5) | realistic (28) | CADGenBench edit, working half (16) |
|---|---|---|---|---|---|
| parts that scored | 40 | 33 | 4 | 24 | 14 |
| IoU mean | 0.969 | 0.955 | 0.956 | 0.713 | 0.469 |
| IoU median | 0.997 | 1.000 | 0.964 | 0.818 | 0.416 |
| parts at IoU 0.9 or above | 37 | 29 | 3 | 10 | 2 |
| missing material, mean | 0.16 % | 0.05 % | 0.12 % | 1.12 % | 0.63 % |
| extra material, mean | 3.4 % | 9.1 % | 4.7 % | 149 % | 782 % |

The previous measurement, before the changes described under "Measurement, defended" and the
three-axis stock, was:

| measure | mfcadpp (40) | holdout (33) | rotational (5) | realistic (28) |
|---|---|---|---|---|
| IoU median | 0.988 | 0.988 | 0.939 | 0.576 |
| parts at IoU 0.9 or above | 35 | 26 | 4 | 5 |
| extra material, mean | 4.2 % | 13.8 % | 4.9 % | 196 % |

### What moved

On the four older corpora, nothing that scored before scores worse by more than 0.03, and
many parts rose sharply: on
the realistic set, NIST CTC 04 from 0.504 to 0.950, Too Tall Toby ppp0106 from 0.576 to 0.969
and curved-support from 0.643 to 1.000; on the holdout, parts 181 and 238 went to 1.000 from
0.823 and 0.781.

Five parts got worse, and they are the open problems:

| part | corpus | before | now |
|---|---|---|---|
| grm-string_post | rotational and realistic | 0.939 | kernel crash |
| nist_ctc_02 | realistic | 0.439 | kernel crash |
| ttt-ppp0101 | realistic | 0.802 | rebuilt solid invalid |
| ttt-23-02-02-sm-hanger | realistic | 0.139 | stopped after three hours |
| 340 | holdout | 0.851 | 0.825 |

The hanger ran at full CPU for three hours before I stopped it. The batch has no per-part
time limit, so a part that never finishes holds the whole run.

## Realistic parts, by source

| source | parts | scored | median IoU | best |
|---|---|---|---|---|
| NIST CTC/FTC | 10 | 9 | 0.533 | 0.953 |
| build123d Too Tall Toby | 13 | 11 | 0.863 | 1.000 |
| CADGenBench | 2 | 2 | 0.915 | 0.936 |
| Gramel | 3 | 2 | 0.996 | 1.000 |

The Too Tall Toby parts moved most, from a median of 0.346 to 0.863. They are plates with
things formed into them, and intersecting the part's shadows along three axes is what fits
them. NIST is now the weakest source: its parts are large, have hundreds of faces, and lose
the most to shapes no axis-aligned shadow comes close to.

## CADGenBench editing corpus

The 32 parts of `splits/editing32.txt` were sorted and split alternately. The working half is
the table below; the holdout half (202, 204, 206, 208, 211, 214, 217, 224, 229, 231, 240, 242,
244, 246, 248, 250) has not been run since the split.

| part | first run | now | missing | extra | surface quiddity associated |
|---|---|---|---|---|---|
| cgb215 | crashed | 0.9457 | 0.13 % | 5.6 % | 49 % |
| cgb218 | 0.9024 | 0.9017 | 0.00 % | 11.1 % | 25 % |
| cgb209 | 0.4353 | 0.7746 | 0.00 % | 29.1 % | 60 % |
| cgb225 | 0.1691 | 0.7653 | 0.00 % | 30.7 % | 21 % |
| cgb238 | 0.3198 | 0.5825 | 1.20 % | 69.6 % | 32 % |
| cgb201 | crashed | 0.4621 | 0.05 % | 116.3 % | 40 % |
| cgb205 | 0.4944 | 0.4219 | 0.25 % | 136.4 % | 56 % |
| cgb247 | 0.3450 | 0.4103 | 0.27 % | 143.1 % | 42 % |
| cgb243 | crashed | 0.3782 | 5.98 % | 148.6 % | 58 % |
| cgb249 | 0.3414 | 0.3634 | 0.23 % | 174.5 % | 6 % |
| cgb230 | 0.1511 | 0.2332 | 0.00 % | 328.8 % | 53 % |
| cgb203 | 0.1818 | 0.1845 | 0.00 % | 442.0 % | 30 % |
| cgb245 | 0.0674 | 0.1383 | 0.30 % | 621.0 % | 15 % |
| cgb241 | 0.0116 | 0.0113 | 0.46 % | 8689.5 % | 6 % |
| cgb207 | crashed | crashed | | | |
| cgb212 | 0.3985 | crashed | | | |

Two parts meet the target. The mean went from 0.318 to 0.469. What remains is nearly all
extra material, which has three distinct causes:

- **No shadow resembles the part.** cgb241 is splines and tubes; its tightest shadow along any
  axis is 25 times its own volume, and quiddity associates 6 % of its surface. Stock-and-cut
  cannot build it. That is a recognition gap, not a decompiler defect.
- **No shadow finishes in time.** cgb203 still falls back to the bounding box, which is why it
  has not moved.
- **The part is far from its hull.** cgb225's rebuild holds all of the part and 31 % more; the
  remaining trims that would remove it cut into real material and are refused.

cgb205 fell from 0.494 to 0.422 and cgb212 now crashes. Neither has been diagnosed yet.

## Why realistic parts are hard

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
