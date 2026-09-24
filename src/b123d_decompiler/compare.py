"""Stage 5: how close is the rebuild?

Bulk and surface metrics come from the vendored cad-fingerprint code. The boolean
metrics are here because that package compares measurements of two shapes, not the
shapes themselves, and "which material is missing" is the question a decompiler has
to answer.
"""

from __future__ import annotations

import math

from build123d import Part
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from .fingerprint.analyze import decoded_mesh, mesh_shape
from .fingerprint.hausdorff import hausdorff_distance
from .geom import _volume, robust_common


def _volume_properties(part: Part) -> GProp_GProps:
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(part.wrapped, properties)
    return properties


def measure(part: Part) -> dict:
    properties = _volume_properties(part)
    centre = properties.CentreOfMass()
    box = part.bounding_box()
    return {
        "volume": float(properties.Mass()),
        "area": float(part.area),
        "centre_of_mass": [centre.X(), centre.Y(), centre.Z()],
        "bbox_min": [box.min.X, box.min.Y, box.min.Z],
        "bbox_max": [box.max.X, box.max.Y, box.max.Z],
        "solids": len(part.solids()),
    }


def _envelope_overlap(left: dict, right: dict) -> float:
    """How much of the smaller bounding box the two share, 0 to 1."""
    shared, smaller = 1.0, float("inf")
    for k in range(3):
        low = max(left["bbox_min"][k], right["bbox_min"][k])
        high = min(left["bbox_max"][k], right["bbox_max"][k])
        shared *= max(high - low, 0.0)
    for box in (left, right):
        volume = 1.0
        for k in range(3):
            volume *= max(box["bbox_max"][k] - box["bbox_min"][k], 0.0)
        smaller = min(smaller, volume)
    return shared / smaller if smaller > 0 else 0.0


def sampled_overlap(reference: Part, rebuilt: Part, points: int = 200000) -> dict:
    """Estimate how two solids overlap by classifying points, when booleans will not.

    A kernel that cannot intersect two shapes can still say whether a point is inside
    one, so a failed boolean does not have to mean no answer at all. The points are
    asked of each shape's mesh rather than of the kernel's classifier: a rebuild read
    back from STEP can come out flagged invalid, and the classifier then gives wrong
    answers without saying so. On one NIST part that read as 5 % of the part missing
    when 0.6 % was. Counting crossings of a mesh only needs the surface, and it is fast
    enough to use ten times the points.
    """
    import numpy as np

    from .inside import MeshInside

    left, right = MeshInside(reference), MeshInside(rebuilt)
    low, high = np.minimum(left.low, right.low), np.maximum(left.high, right.high)
    volume = float(np.prod(high - low))
    places = np.random.default_rng(20260921).uniform(low, high, size=(points, 3))
    here, there = left.contains(places), right.contains(places)
    in_reference, in_rebuilt = int(here.sum()), int(there.sum())
    in_both = int((here & there).sum())
    either = in_reference + in_rebuilt - in_both
    scale = volume / points
    return {
        "points": points,
        "iou": (in_both / either) if either else None,
        "shared_volume": in_both * scale,
        "missing_pct": (
            100 * (in_reference - in_both) / in_reference if in_reference else None
        ),
        "extra_pct": (
            100 * (in_rebuilt - in_both) / in_reference if in_reference else None
        ),
    }


def compare_parts(reference: Part, rebuilt: Part, *, samples: int = 2000,
                  surface: bool = True) -> dict:
    """Compare two solids: bulk, boolean and surface agreement."""
    ref, test = measure(reference), measure(rebuilt)
    shape, note = robust_common(reference.wrapped, rebuilt.wrapped)
    shared = _volume(shape) if shape is not None else 0.0
    union = ref["volume"] + test["volume"] - shared
    # An intersection that could not be computed is an unknown, not a zero. Reporting
    # it as zero turns a measurement failure into a perfect-looking total loss, and
    # drags down any average it lands in.
    # A boolean that returns a sliver where the envelopes plainly coincide has failed
    # just as surely as one that returns nothing, and it does not announce it. Whenever
    # the answer looks too small for two shapes sitting on top of each other, check it
    # by classifying points instead, which the kernel can still do reliably.
    envelope = _envelope_overlap(ref, test)
    # Cross-check whenever the two occupy much the same space and the boolean claims
    # they barely meet. The threshold used to be a hundredth of the smaller solid,
    # which a failure can clear while still being wrong by a factor of twenty, and the
    # part then reads as almost entirely missing when it is in fact entirely there.
    doubtful = shared < 0.5 * min(ref["volume"], test["volume"]) and envelope > 0.5
    measured = not (shared <= 0.0 and note)
    sampled = sampled_overlap(reference, rebuilt) if doubtful else None
    report = {
        "boolean_note": note,
        "reference": ref,
        "rebuilt": test,
        "volume_error_pct": (
            100 * (test["volume"] - ref["volume"]) / ref["volume"] if ref["volume"] else None
        ),
        "area_error_pct": (
            100 * (test["area"] - ref["area"]) / ref["area"] if ref["area"] else None
        ),
        "iou": (shared / union if union else None) if measured else None,
        "missing_volume": max(ref["volume"] - shared, 0.0) if measured else None,
        "extra_volume": max(test["volume"] - shared, 0.0) if measured else None,
        "missing_pct": (
            (100 * max(ref["volume"] - shared, 0.0) / ref["volume"] if ref["volume"] else None)
            if measured
            else None
        ),
        "extra_pct": (
            (100 * max(test["volume"] - shared, 0.0) / ref["volume"] if ref["volume"] else None)
            if measured
            else None
        ),
        "centre_of_mass_offset": math.dist(ref["centre_of_mass"], test["centre_of_mass"]),
        "bbox_offset": max(
            max(abs(a - b) for a, b in zip(ref[key], test[key]))
            for key in ("bbox_min", "bbox_max")
        ),
    }

    if sampled is not None and sampled["iou"] is not None:
        # Believe the boolean when the two agree; a sampled estimate is coarser.
        estimate = sampled["shared_volume"]
        if shared > 0 and 0.5 < estimate / shared < 2.0:
            sampled = None
    if sampled is not None:
        report["sampled"] = sampled
        report["iou"] = sampled["iou"]
        report["missing_pct"] = sampled["missing_pct"]
        report["extra_pct"] = sampled["extra_pct"]
        report["missing_volume"] = None
        report["extra_volume"] = None
        report["boolean_note"] = (
            f"{note + '; ' if note else ''}the intersection was not believable, "
            f"so the overlap was estimated from {sampled['points']} sample points"
        )

    if surface:
        try:
            ref_mesh = decoded_mesh(mesh_shape(reference))
            test_mesh = decoded_mesh(mesh_shape(rebuilt))
            report["hausdorff"] = hausdorff_distance(ref_mesh, test_mesh, samples=samples)
        except Exception as error:  # noqa: BLE001 - a failed mesh is a reportable outcome
            report["hausdorff"] = {"error": f"{type(error).__name__}: {error}"}
    return report


def format_report(report: dict) -> str:
    """One compact block per part, for a terminal."""
    def percent(key):
        value = report.get(key)
        return f"{value:.2f} %" if value is not None else "not measurable"

    rows = [
        ("IoU", f"{report['iou']:.4f}" if report.get("iou") is not None else "not measurable"),
        ("volume error", f"{report['volume_error_pct']:+.2f} %"),
        ("missing material", percent("missing_pct")),
        ("extra material", percent("extra_pct")),
        ("surface area error", f"{report['area_error_pct']:+.2f} %"),
        ("centre of mass off by", f"{report['centre_of_mass_offset']:.4f} mm"),
        ("bounding box off by", f"{report['bbox_offset']:.4f} mm"),
    ]
    if report.get("boolean_note"):
        rows.append(("boolean", report["boolean_note"]))
    haus = report.get("hausdorff") or {}
    if "error" in haus:
        rows.append(("surface deviation", haus["error"]))
    elif haus:
        rows.append(("surface deviation max", f"{haus['hausdorff']:.4f} mm"))
        rows.append(("surface deviation mean", f"{haus['mean']:.4f} mm"))
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"  {label.rjust(width)}  {value}" for label, value in rows)
