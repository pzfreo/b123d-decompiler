# Baselines

Three corpora. Adapters were developed against **mfcadpp** and never run against
**mfcadpp_holdout** until they were finished, so the holdout says whether they generalise.
The **realistic** set is 28 parts assembled from four sources with nothing synthetic about
them. Row-level numbers are in `baselines/`.

```bash
uv run b123d-decompile batch path/to/corpus -o results/
uv run python tools/corpus_summary.py results/
```

## Result

Measured against quiddity 0.3.3.

| measure | mfcadpp (40) | holdout (33) | rotational (5) | realistic (28) |
|---|---|---|---|---|
| parts that executed | 40 | 33 | 5 | 25, 2 kernel crashes |
| IoU mean | 0.962 | 0.933 | 0.954 | 0.603 |
| IoU median | 0.988 | 0.988 | 0.939 | 0.576 |
| parts at IoU 0.9 or above | 35 | 26 | 4 | 5 |
| missing material, mean | 0.16 % | 0.07 % | 0.09 % | 0.36 % |
| extra material, mean | 4.2 % | 13.8 % | 4.9 % | 196 % |

The rotational five, from Gramel and CADGenBench, are kept as their own baseline because
they exercise the turned billet and the swept chamfers that nothing in MFCAD++ touches.
They went from a median of 0.611 to 0.939 once the billet followed the part's own radius
profile and the swept treatments stopped being mirrored.

## Realistic parts, by source

| source | parts | median IoU | best |
|---|---|---|---|
| NIST CTC/FTC | 9 | 0.419 | 0.922 |
| build123d Too Tall Toby | 13 | 0.346 | 0.768 |
| CADGenBench | 2 | 0.492 | 0.794 |
| Gramel | 3 | 0.611 | 0.702 |

Turned parts do best, because a turned profile is the one record that describes a whole
billet and the stock comes out right. The build123d practice parts do worst: they are thin,
folded sheet-metal-like shapes whose volume is a few percent of any box around them.

## Why realistic parts are hard

The error is overwhelmingly material left behind, not material wrongly removed: 591 %
extra against 4.3 % missing. Three things drive it.

**The bounding box is hopeless for these shapes.** A bracket or a hanger occupies a small
fraction of its envelope, and `ttt-23-02-02-sm-hanger` leaves 1257 % extra. Turned stock
helps the rotational parts and there is no equivalent for the rest.

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
Over the 26 realistic parts both runs scored:

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

Five failures found and fixed, each of which had been quietly producing wrong numbers.

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

## Ground truth

None of the corpora has ground truth, so those are relative numbers. The absolute check is
`tests/test_roundtrip.py`, which builds parts from known features, exports them and
decompiles them back. Eight fixtures rebuild at IoU 1.0000: a through hole, a blind hole, a
counterbored hole, a rectangular pocket, a through step, a chamfered edge, a rounded corner
and a quarter-round corner cut.
