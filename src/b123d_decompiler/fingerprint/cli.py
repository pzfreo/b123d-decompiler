"""CLI for cad-fingerprint.

Usage:
    cad-fingerprint reference.step                    # print JSON fingerprint
    cad-fingerprint reference.stl  -o test_part.py    # generate pytest file
    cad-fingerprint reference.step --json fp.json     # save JSON fingerprint
    cad-fingerprint compare ref.step impl.step        # compare two STEP files
"""

import argparse
import sys
from pathlib import Path


def _positive_int(value: str) -> int:
    """argparse type for counts that must be at least 1."""
    import argparse as _argparse

    number = int(value)
    if number < 1:
        raise _argparse.ArgumentTypeError(
            f"must be at least 1, got {number} — zero samples would report "
            f"a perfect match for any part"
        )
    return number


def _positive_float(value: str) -> float:
    """argparse type for lengths that must be greater than zero."""
    import argparse as _argparse

    number = float(value)
    if number <= 0:
        raise _argparse.ArgumentTypeError(
            f"must be greater than 0, got {number}"
        )
    return number


def _add_tolerance_args(parser):
    """Add shared tolerance arguments to a parser."""
    parser.add_argument("--volume-tol", type=float, default=1.0,
                        help="Volume tolerance %% (default: 1.0)")
    parser.add_argument("--area-tol", type=float, default=2.0,
                        help="Surface area tolerance %% (default: 2.0)")
    parser.add_argument("--bbox-tol", type=float, default=0.1,
                        help="Bounding box tolerance mm (default: 0.1)")
    parser.add_argument("--inertia-tol", type=float, default=2.0,
                        help="Inertia tolerance %% (default: 2.0)")
    parser.add_argument("--xs-area-tol", type=float, default=3.0,
                        help="Cross-section area tolerance %% (default: 3.0)")
    parser.add_argument("--xs-centroid-tol", type=float, default=0.2,
                        help="Cross-section centroid tolerance mm (default: 0.2)")
    parser.add_argument("--xs-moment-tol", type=float, default=5.0,
                        help="Cross-section 2D moment tolerance %% (default: 5.0)")
    parser.add_argument("--radial-tol", type=float, default=0.15,
                        help="Radial profile tolerance mm (default: 0.15)")
    parser.add_argument("--hausdorff-tol", type=float, default=0.3,
                        help="Max surface deviation mm (default: 0.3)")
    parser.add_argument("--hausdorff-mean-tol", type=float, default=0.05,
                        help="Mean surface deviation mm (default: 0.05)")


def _add_analysis_args(parser):
    """Add shared analysis arguments to a parser."""
    parser.add_argument(
        "--axis", default="Z", choices=["X", "Y", "Z"],
        help="Primary axis of the part (default: Z)",
    )
    parser.add_argument(
        "--cross-sections", type=int, default=20,
        help="Number of cross-section slices (default: 20)",
    )
    parser.add_argument(
        "--radial-slices", type=int, default=15,
        help="Number of radial profile slices (default: 15)",
    )
    parser.add_argument(
        "--angles", type=int, default=12,
        help="Number of angular samples per radial slice (default: 12)",
    )
    parser.add_argument(
        "--hausdorff-samples", type=_positive_int, default=2000,
        help="Surface sample points per direction for the Hausdorff "
             "distance (default: 2000)",
    )
    parser.add_argument(
        "--mesh-deflection", type=_positive_float, default=None,
        help="Surface mesh resolution in mm — the triangulation deflection "
             "for a STEP reference and for the part under test; for an STL "
             "reference, whose facets cannot be re-meshed, the "
             "vertex-clustering cell, capped at a third of the part's "
             "thinnest dimension (default: bounding-box diagonal / 1000)",
    )
    parser.add_argument(
        "--max-mesh-triangles", type=_positive_int, default=12000,
        help="Triangle budget for the reference surface mesh — raise it for "
             "tighter surface-deviation tolerances at the cost of a bigger "
             "test file (default: 12000)",
    )
    parser.add_argument(
        "--stl-facet-error", type=_positive_float, default=None,
        help="For an STL reference, the export tolerance its facets were "
             "written at, in mm. Estimated from the facets when omitted — "
             "which is impossible for a mesh coarse enough to be a faceted "
             "part in its own right",
    )
    parser.add_argument(
        "--no-hausdorff", action="store_true",
        help="Skip the surface mesh and Hausdorff distance measurements",
    )


def _warn_if_facet_error_unreadable(fp, mesh):
    """Warn when an STL's facet error cannot be read from the file.

    Past a certain coarseness the folds between facets are indistinguishable
    from real edges — a hexagonal prism and a six-facet cylinder are the same
    mesh — and the estimate comes back at zero. If the part is genuinely
    faceted that is correct; if it approximates curved surfaces it is not,
    and only the person who exported it knows which.
    """
    from .analyze import decoded_mesh
    from .hausdorff import coarse_fold_fraction

    if mesh["facet_error"] > 0.0:
        return
    vertices, triangles = decoded_mesh(mesh)
    if coarse_fold_fraction(vertices, triangles) < 0.2:
        return
    print("  Note: this mesh folds too sharply for its facet error to be "
          "read. If it\n        approximates curved surfaces, pass "
          "--stl-facet-error <mm> with the\n        tolerance it was "
          "exported at, or the generated tests will hold your\n        "
          "implementation to a tolerance the reference cannot meet.")


def _run_analyze(args):
    """Run the fingerprint analysis workflow."""
    cad_path = Path(args.cad_file)
    if not cad_path.exists():
        print(f"Error: file not found: {cad_path}", file=sys.stderr)
        sys.exit(1)

    suffix = cad_path.suffix.lower()
    is_stl = suffix == ".stl"
    is_step = suffix in (".step", ".stp")
    if not is_stl and not is_step:
        print(f"Error: unsupported format '{suffix}' (expected .step, .stp, or .stl)",
              file=sys.stderr)
        sys.exit(1)

    module_name = args.name or cad_path.stem

    print(f"Analyzing {cad_path}...")
    from .fingerprint import CadFingerprint
    if is_stl:
        fp = CadFingerprint.from_stl(
            cad_path,
            axis=args.axis,
            num_cross_sections=args.cross_sections,
            num_radial_slices=args.radial_slices,
            num_angles=args.angles,
            capture_mesh=not args.no_hausdorff,
            mesh_deflection=args.mesh_deflection,
            max_mesh_triangles=args.max_mesh_triangles,
            stl_facet_error=args.stl_facet_error,
        )
        print("  (STL mode: face inventory shows mesh stats only, no surface type classification)")
    else:
        fp = CadFingerprint.from_step(
            cad_path,
            axis=args.axis,
            num_cross_sections=args.cross_sections,
            num_radial_slices=args.radial_slices,
            num_angles=args.angles,
            capture_mesh=not args.no_hausdorff,
            mesh_deflection=args.mesh_deflection,
            max_mesh_triangles=args.max_mesh_triangles,
        )

    va = fp.volume_and_area
    bb = fp.bounding_box
    print(f"  Volume: {va['volume']:.2f} mm³")
    print(f"  Surface area: {va['surface_area']:.2f} mm²")
    print(f"  Bounding box: {bb['size'][0]:.2f} × {bb['size'][1]:.2f} × {bb['size'][2]:.2f} mm")
    print(f"  Faces: {fp.topology['faces']}, Edges: {fp.topology['edges']}")
    if fp.surface_mesh:
        mesh_kb = (len(fp.surface_mesh["vertices"])
                   + len(fp.surface_mesh["triangles"])) // 1024
        print(f"  Surface mesh: {fp.surface_mesh['triangle_count']} triangles "
              f"at {fp.surface_mesh['deflection']:.4f} mm deflection ({mesh_kb} KB embedded)")
        if is_stl:
            mesh = fp.surface_mesh
            source = ("declared" if mesh.get("facet_error_declared")
                      else "estimated from the facets")
            print(f"  STL facet error: {mesh['facet_error']:.4f} mm ({source})")
            if mesh.get("cluster_cell"):
                print(f"  Clustered to fit the triangle budget: "
                      f"{mesh['cluster_cell']:.4f} mm cell "
                      f"(raise --max-mesh-triangles to keep more detail)")
            if not mesh.get("facet_error_declared"):
                _warn_if_facet_error_unreadable(fp, mesh)
        if mesh_kb > 250:
            print(f"  Note: that mesh is large for an embedded test file — "
                  f"lower --max-mesh-triangles or raise --mesh-deflection, "
                  f"or pass --no-hausdorff to skip it.")

    if args.json:
        fp.to_json(args.json)
        print(f"  JSON fingerprint saved to: {args.json}")

    if args.output:
        from .generate import generate_test_file
        generate_test_file(
            fp,
            output_path=args.output,
            module_name=module_name,
            fixture_name=args.fixture,
            axis=args.axis,
            volume_tol_pct=args.volume_tol,
            area_tol_pct=args.area_tol,
            bbox_tol_mm=args.bbox_tol,
            inertia_tol_pct=args.inertia_tol,
            cross_section_area_tol_pct=args.xs_area_tol,
            cross_section_centroid_tol_mm=args.xs_centroid_tol,
            cross_section_moment_tol_pct=args.xs_moment_tol,
            radial_tol_mm=args.radial_tol,
            hausdorff_tol_mm=args.hausdorff_tol,
            hausdorff_mean_tol_mm=args.hausdorff_mean_tol,
            hausdorff_samples=args.hausdorff_samples,
        )
        print(f"  Test file generated: {args.output}")

    if args.prompt:
        from .generate import generate_prompt
        generate_prompt(
            fp,
            output_path=args.prompt,
            module_name=module_name,
            axis=args.axis,
        )
        print(f"  Prompt file generated: {args.prompt}")

    if not args.output and not args.json and not args.prompt:
        # The mesh blob is tens of thousands of base64 characters; summarise
        # it rather than burying the fingerprint the user asked to see.
        print(_summarised_json(fp))
        print("# surface mesh omitted above — use --json to write it",
              file=sys.stderr)


def _summarised_json(fp) -> str:
    """Fingerprint JSON with the encoded mesh replaced by its shape."""
    import json
    from dataclasses import asdict

    data = asdict(fp)
    mesh = data.get("surface_mesh")
    if mesh and mesh.get("triangle_count"):
        # Null rather than prose, plus a flag decode_mesh refuses: this is
        # still valid JSON, so it must not load back as a plausible mesh.
        data["surface_mesh"] = dict(
            mesh, vertices=None, triangles=None, truncated=True,
        )
    return json.dumps(data, indent=2)


def _run_compare(args):
    """Compare two STEP files."""
    from .fingerprint import CadFingerprint
    from .compare import compare_fingerprints, format_comparison

    ref_path = Path(args.ref_step)
    impl_path = Path(args.impl_step)

    for p in (ref_path, impl_path):
        if not p.exists():
            print(f"Error: STEP file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Analyzing reference: {ref_path}...")
    ref_fp = CadFingerprint.from_step(
        ref_path,
        axis=args.axis,
        num_cross_sections=args.cross_sections,
        num_radial_slices=args.radial_slices,
        num_angles=args.angles,
        capture_mesh=not args.no_hausdorff,
        mesh_deflection=args.mesh_deflection,
        max_mesh_triangles=args.max_mesh_triangles,
    )

    print(f"Analyzing implementation: {impl_path}...")
    impl_fp = CadFingerprint.from_step(
        impl_path,
        axis=args.axis,
        num_cross_sections=args.cross_sections,
        num_radial_slices=args.radial_slices,
        num_angles=args.angles,
        capture_mesh=not args.no_hausdorff,
        # Match the reference's tessellation exactly — different meshing
        # settings on the two sides would show up as surface deviation.
        mesh_deflection=(
            args.mesh_deflection
            if args.mesh_deflection is not None
            else ref_fp.surface_mesh.get("deflection")
        ),
        mesh_angular_deflection=ref_fp.surface_mesh.get("angular_deflection"),
        max_mesh_triangles=(
            # Room to be more detailed than the reference, but still a cap:
            # meshing an intricate part at the reference's deflection can
            # otherwise run into millions of triangles. Same rule the
            # generated tests apply to the part under test.
            max(ref_fp.surface_mesh["triangle_count"] * 4, 20000)
            if ref_fp.surface_mesh.get("triangle_count")
            else args.max_mesh_triangles
        ),
    )

    print()
    result = compare_fingerprints(
        ref_fp, impl_fp,
        volume_tol_pct=args.volume_tol,
        area_tol_pct=args.area_tol,
        bbox_tol_mm=args.bbox_tol,
        inertia_tol_pct=args.inertia_tol,
        cross_section_area_tol_pct=args.xs_area_tol,
        cross_section_centroid_tol_mm=args.xs_centroid_tol,
        radial_tol_mm=args.radial_tol,
        hausdorff_tol_mm=args.hausdorff_tol,
        hausdorff_mean_tol_mm=args.hausdorff_mean_tol,
        hausdorff_samples=args.hausdorff_samples,
    )
    print(format_comparison(result))

    if result["summary"]["fail"] > 0:
        sys.exit(1)


def main():
    # Check if first arg is "compare" for backward compatibility
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        parser = argparse.ArgumentParser(
            description="Compare two STEP files geometrically",
            prog="cad-fingerprint compare",
        )
        parser.add_argument("_cmd", help=argparse.SUPPRESS)  # consume "compare"
        parser.add_argument("ref_step", help="Reference STEP file")
        parser.add_argument("impl_step", help="Implementation STEP file to compare")
        _add_analysis_args(parser)
        _add_tolerance_args(parser)
        args = parser.parse_args()
        _run_compare(args)
    else:
        parser = argparse.ArgumentParser(
            description="Generate geometric fingerprint tests from STEP or STL files",
            prog="cad-fingerprint",
        )
        parser.add_argument("cad_file", help="Path to the STEP or STL file to fingerprint")
        parser.add_argument(
            "-o", "--output",
            help="Output pytest file path (e.g. test_my_part.py)",
        )
        parser.add_argument(
            "--json",
            help="Save fingerprint as JSON (for inspection or later use)",
        )
        parser.add_argument(
            "--prompt",
            help="Output prompt/guide file path (e.g. PROMPT.md)",
        )
        parser.add_argument(
            "--name", default=None,
            help="Human-readable part name (default: stem of input filename)",
        )
        parser.add_argument(
            "--fixture", default="part_under_test",
            help="pytest fixture name (default: part_under_test)",
        )
        _add_analysis_args(parser)
        _add_tolerance_args(parser)
        args = parser.parse_args()
        _run_analyze(args)


if __name__ == "__main__":
    main()
