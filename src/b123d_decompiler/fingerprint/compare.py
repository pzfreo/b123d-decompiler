"""Compare two fingerprints by diffing their geometric measurements.

Produces a structured comparison with pass/close/fail status for each
metric, and a color-coded terminal summary. Works on fingerprints loaded
from JSON, which needs no CAD kernel.
"""

from __future__ import annotations

from .fingerprint import CadFingerprint


def _status(actual, ref, tol, is_absolute=False):
    """Return (status, diff_display) for a single value comparison."""
    if is_absolute:
        diff = abs(actual - ref)
        pct = None
        if diff < tol:
            return "pass", f"{diff:.4f}"
        elif diff < tol * 2:
            return "close", f"{diff:.4f}"
        else:
            return "fail", f"{diff:.4f}"
    else:
        if abs(ref) < 1e-9:
            diff = abs(actual - ref)
            if diff < 1.0:
                return "pass", f"{diff:.4f}"
            else:
                return "fail", f"{diff:.4f}"
        pct = abs(actual - ref) / abs(ref) * 100
        if pct < tol:
            return "pass", f"{pct:.1f}%"
        elif pct < tol * 2:
            return "close", f"{pct:.1f}%"
        else:
            return "fail", f"{pct:.1f}%"


def compare_fingerprints(
    ref: CadFingerprint,
    actual: CadFingerprint,
    volume_tol_pct: float = 1.0,
    area_tol_pct: float = 2.0,
    bbox_tol_mm: float = 0.1,
    inertia_tol_pct: float = 2.0,
    cross_section_area_tol_pct: float = 3.0,
    cross_section_centroid_tol_mm: float = 0.2,
    radial_tol_mm: float = 0.15,
    hausdorff_tol_mm: float = 0.3,
    hausdorff_mean_tol_mm: float = 0.05,
    hausdorff_samples: int = 2000,
) -> dict:
    """Compare two fingerprints and return structured results.

    Each metric gets a status: "pass", "close", or "fail".
    """
    results = {}

    # Volume
    rv = ref.volume_and_area
    av = actual.volume_and_area
    s, d = _status(av["volume"], rv["volume"], volume_tol_pct, is_absolute=False)
    results["volume"] = {
        "ref": rv["volume"], "actual": av["volume"], "diff": d, "status": s,
    }

    # Surface area
    s, d = _status(av["surface_area"], rv["surface_area"], area_tol_pct, is_absolute=False)
    results["surface_area"] = {
        "ref": rv["surface_area"], "actual": av["surface_area"], "diff": d, "status": s,
    }

    # Bounding box
    bb_results = []
    for i, axis_name in enumerate(("X", "Y", "Z")):
        for bound in ("min", "max"):
            r_val = ref.bounding_box[bound][i]
            a_val = actual.bounding_box[bound][i]
            s, d = _status(a_val, r_val, bbox_tol_mm, is_absolute=True)
            bb_results.append({
                "label": f"{bound}.{axis_name}", "ref": r_val, "actual": a_val,
                "diff": d, "status": s,
            })
    results["bounding_box"] = bb_results

    # Moments of inertia
    moi_results = {}
    for key in ("Ixx", "Iyy", "Izz"):
        r_val = ref.moments_of_inertia[key]
        a_val = actual.moments_of_inertia[key]
        s, d = _status(a_val, r_val, inertia_tol_pct, is_absolute=False)
        moi_results[key] = {"ref": r_val, "actual": a_val, "diff": d, "status": s}
    for key in ("Ixy", "Ixz", "Iyz"):
        r_val = ref.moments_of_inertia[key]
        a_val = actual.moments_of_inertia[key]
        tol = max(abs(r_val) * inertia_tol_pct / 100, 1.0)
        s, d = _status(a_val, r_val, tol, is_absolute=True)
        moi_results[key] = {"ref": r_val, "actual": a_val, "diff": d, "status": s}
    results["moments_of_inertia"] = moi_results

    # Face inventory
    ref_types = {}
    for f in ref.face_inventory:
        ref_types[f["type"]] = ref_types.get(f["type"], 0) + 1
    act_types = {}
    for f in actual.face_inventory:
        act_types[f["type"]] = act_types.get(f["type"], 0) + 1
    face_results = {}
    for ftype in set(list(ref_types.keys()) + list(act_types.keys())):
        rc = ref_types.get(ftype, 0)
        ac = act_types.get(ftype, 0)
        diff = abs(ac - rc)
        s = "pass" if diff <= 2 else ("close" if diff <= 4 else "fail")
        face_results[ftype] = {"ref": rc, "actual": ac, "diff": diff, "status": s}
    results["face_inventory"] = face_results

    # Cross-sections
    xs_results = []
    for r_cs, a_cs in zip(ref.cross_sections, actual.cross_sections):
        r_area = r_cs["area"]
        a_area = a_cs["area"]
        if r_area < 0.01:
            s = "pass" if a_area < 0.5 else "fail"
            d = f"{a_area:.4f}"
        else:
            s, d = _status(a_area, r_area, cross_section_area_tol_pct, is_absolute=False)
        xs_results.append({
            "position": r_cs["position"], "ref_area": r_area, "actual_area": a_area,
            "diff": d, "status": s,
        })
    results["cross_sections"] = xs_results

    # Radial profile
    rp_results = []
    for r_rp, a_rp in zip(ref.radial_profile, actual.radial_profile):
        angle_diffs = {}
        worst = "pass"
        for deg_str, ref_r in r_rp["radii"].items():
            if ref_r is None:
                continue
            act_r_val = a_rp["radii"].get(deg_str)
            if act_r_val is None:
                angle_diffs[deg_str] = {"status": "fail", "diff": "no intersection"}
                worst = "fail"
                continue
            s, d = _status(act_r_val, ref_r, radial_tol_mm, is_absolute=True)
            angle_diffs[deg_str] = {"ref": ref_r, "actual": act_r_val, "diff": d, "status": s}
            if s == "fail":
                worst = "fail"
            elif s == "close" and worst == "pass":
                worst = "close"
        rp_results.append({
            "position": r_rp["position"], "angles": angle_diffs, "status": worst,
        })
    results["radial_profile"] = rp_results

    # Hausdorff distance — needs the surface mesh from both fingerprints.
    if ref.surface_mesh.get("triangle_count") and actual.surface_mesh.get(
        "triangle_count"
    ):
        # decode_mesh rather than analyze.decoded_mesh: this path stays
        # free of build123d, so two saved fingerprints can be compared
        # without a CAD kernel installed.
        from .hausdorff import decode_mesh, hausdorff_distance, tolerance_floor

        h = hausdorff_distance(
            decode_mesh(ref.surface_mesh),
            decode_mesh(actual.surface_mesh),
            samples=hausdorff_samples,
        )
        # Never grade below what the meshes can resolve — but derive that
        # from the reference, scaled to the deflection the implementation was
        # meshed at. Taking the implementation's own measured resolution
        # would let a part with more detail than the reference, coarsened to
        # fit the same budget, widen the tolerance it is judged by.
        floor = tolerance_floor(
            ref.surface_mesh, actual.surface_mesh.get("deflection")
        )
        max_tol = max(hausdorff_tol_mm, floor)
        mean_tol = max(hausdorff_mean_tol_mm, floor / 4.0)
        max_status, _ = _status(h["hausdorff"], 0.0, max_tol, is_absolute=True)
        mean_status, _ = _status(h["mean"], 0.0, mean_tol, is_absolute=True)
        rank = {"pass": 0, "close": 1, "fail": 2}
        worst = max((max_status, mean_status), key=lambda st: rank[st])
        results["hausdorff"] = {
            "max": h["hausdorff"],
            "forward": h["forward"]["max"],
            "backward": h["backward"]["max"],
            "mean": h["mean"],
            "rms": h["rms"],
            "p95": h["p95"],
            "samples": h["samples"],
            "tolerance": max_tol,
            "mean_tolerance": mean_tol,
            "requested_tolerance": hausdorff_tol_mm,
            "requested_mean_tolerance": hausdorff_mean_tol_mm,
            "resolution_limited": (max_tol > hausdorff_tol_mm
                                   or mean_tol > hausdorff_mean_tol_mm),
            "max_status": max_status,
            "mean_status": mean_status,
            "status": worst,
        }

    # Summary
    all_statuses = []
    all_statuses.append(results["volume"]["status"])
    all_statuses.append(results["surface_area"]["status"])
    all_statuses.extend(b["status"] for b in results["bounding_box"])
    all_statuses.extend(m["status"] for m in results["moments_of_inertia"].values())
    all_statuses.extend(f["status"] for f in results["face_inventory"].values())
    all_statuses.extend(x["status"] for x in results["cross_sections"])
    all_statuses.extend(r["status"] for r in results["radial_profile"])
    if "hausdorff" in results:
        all_statuses.append(results["hausdorff"]["status"])

    results["summary"] = {
        "pass": all_statuses.count("pass"),
        "close": all_statuses.count("close"),
        "fail": all_statuses.count("fail"),
    }

    return results


# ANSI color codes
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_STATUS_COLORS = {"pass": _GREEN, "close": _YELLOW, "fail": _RED}
_STATUS_SYMBOLS = {"pass": "✓", "close": "~", "fail": "✗"}


def _colored(status: str, text: str) -> str:
    return f"{_STATUS_COLORS[status]}{text}{_RESET}"


def format_comparison(result: dict) -> str:
    """Format comparison results as color-coded terminal output."""
    lines = []

    lines.append(f"{_BOLD}STEP Comparison Results{_RESET}")
    lines.append("")

    # Global properties
    lines.append(f"{_BOLD}Global Properties{_RESET}")
    for key in ("volume", "surface_area"):
        r = result[key]
        sym = _colored(r["status"], _STATUS_SYMBOLS[r["status"]])
        lines.append(f"  {sym} {key}: {r['actual']:.4f} vs ref {r['ref']:.4f} ({r['diff']})")

    # Bounding box
    lines.append(f"{_BOLD}Bounding Box{_RESET}")
    for b in result["bounding_box"]:
        sym = _colored(b["status"], _STATUS_SYMBOLS[b["status"]])
        lines.append(f"  {sym} {b['label']}: {b['actual']:.4f} vs ref {b['ref']:.4f} ({b['diff']})")

    # Moments of inertia
    lines.append(f"{_BOLD}Moments of Inertia{_RESET}")
    for key, m in result["moments_of_inertia"].items():
        sym = _colored(m["status"], _STATUS_SYMBOLS[m["status"]])
        lines.append(f"  {sym} {key}: {m['actual']:.4f} vs ref {m['ref']:.4f} ({m['diff']})")

    # Face inventory
    lines.append(f"{_BOLD}Face Inventory{_RESET}")
    for ftype, f in result["face_inventory"].items():
        sym = _colored(f["status"], _STATUS_SYMBOLS[f["status"]])
        lines.append(f"  {sym} {ftype}: {f['actual']} vs ref {f['ref']}")

    # Cross-sections (compact — only show non-pass)
    lines.append(f"{_BOLD}Cross-Sections{_RESET}")
    xs = result["cross_sections"]
    pass_count = sum(1 for x in xs if x["status"] == "pass")
    if pass_count == len(xs):
        lines.append(f"  {_colored('pass', '✓')} All {len(xs)} cross-sections pass")
    else:
        lines.append(f"  {pass_count}/{len(xs)} pass")
        for x in xs:
            if x["status"] != "pass":
                sym = _colored(x["status"], _STATUS_SYMBOLS[x["status"]])
                lines.append(f"  {sym} pos={x['position']}: area {x['actual_area']:.4f} "
                             f"vs ref {x['ref_area']:.4f} ({x['diff']})")

    # Radial profile (compact)
    lines.append(f"{_BOLD}Radial Profile{_RESET}")
    rp = result["radial_profile"]
    rp_pass = sum(1 for r in rp if r["status"] == "pass")
    if rp_pass == len(rp):
        lines.append(f"  {_colored('pass', '✓')} All {len(rp)} radial positions pass")
    else:
        lines.append(f"  {rp_pass}/{len(rp)} pass")
        for r in rp:
            if r["status"] != "pass":
                sym = _colored(r["status"], _STATUS_SYMBOLS[r["status"]])
                fail_angles = [a for a, d in r["angles"].items() if d["status"] != "pass"]
                lines.append(f"  {sym} pos={r['position']}: {len(fail_angles)} angle(s) off")

    # Hausdorff distance
    if "hausdorff" in result:
        h = result["hausdorff"]
        max_sym = _colored(h["max_status"], _STATUS_SYMBOLS[h["max_status"]])
        mean_sym = _colored(h["mean_status"], _STATUS_SYMBOLS[h["mean_status"]])
        lines.append(f"{_BOLD}Surface Deviation (Hausdorff){_RESET}")
        lines.append(f"  {max_sym} max {h['max']:.4f} mm "
                     f"(tolerance {h['tolerance']:.4f} mm, "
                     f"{h['samples']} samples/direction)")
        lines.append(f"  {mean_sym} mean {h['mean']:.4f} mm "
                     f"(tolerance {h['mean_tolerance']:.4f} mm)")
        lines.append(f"    ref→impl {h['forward']:.4f}, impl→ref {h['backward']:.4f}, "
                     f"RMS {h['rms']:.4f}, 95th pct {h['p95']:.4f}")
        if h["resolution_limited"]:
            raised = []
            if h["max_status"] is not None and h["tolerance"] > h["requested_tolerance"]:
                raised.append(f"max {h['requested_tolerance']} → "
                              f"{h['tolerance']:.4f}")
            if h["mean_tolerance"] > h["requested_mean_tolerance"]:
                raised.append(f"mean {h['requested_mean_tolerance']} → "
                              f"{h['mean_tolerance']:.4f}")
            lines.append(f"    (raised to the mesh resolution: "
                         f"{'; '.join(raised)} mm — mesh finer for a "
                         f"tighter check)")
    else:
        lines.append(f"{_BOLD}Surface Deviation (Hausdorff){_RESET}")
        lines.append("  - not measured: one or both fingerprints have no "
                     "surface mesh")

    # Summary
    lines.append("")
    s = result["summary"]
    pass_str = _colored('pass', f'{s["pass"]} pass')
    close_str = _colored('close', f'{s["close"]} close')
    fail_str = _colored('fail', f'{s["fail"]} fail')
    lines.append(f"{_BOLD}Summary:{_RESET} {pass_str}, {close_str}, {fail_str}")

    return "\n".join(lines)
