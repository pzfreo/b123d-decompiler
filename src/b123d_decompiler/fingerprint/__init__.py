"""cad-fingerprint: Generate geometric fingerprint tests from STEP and STL files.

Vendored from https://github.com/pzfreo/cad-fingerprint at commit 20c0eec (v0.3.0) and
maintained here as ``b123d_decompiler.fingerprint``. The ``cad-fingerprint`` console script
still points at this package's :mod:`cli`.

Usage:
    cad-fingerprint reference.step -o tests/test_reference.py
    cad-fingerprint reference.stl  -o tests/test_reference.py

Generates a pytest test file containing a comprehensive geometric fingerprint
of the reference file. Any procedural build123d implementation that passes all
tests is geometrically equivalent to the reference for manufacturing purposes.

Reading CAD files needs build123d and OCC, but measuring surface deviation
does not: a saved JSON fingerprint carries its own mesh, and
:mod:`b123d_decompiler.fingerprint.hausdorff` is pure Python. So the names that pull in a
CAD kernel resolve on first use rather than at import, and comparing two saved
fingerprints works with nothing but the standard library.
"""

__all__ = ["CadFingerprint", "StepFingerprint", "analyze_step", "analyze_stl"]

_SUBMODULES = ("analyze", "cli", "compare", "fingerprint", "generate",
               "hausdorff")


def __getattr__(name):
    """Import the CAD-backed names on demand (PEP 562)."""
    if name in ("CadFingerprint", "StepFingerprint"):
        from .fingerprint import CadFingerprint

        return CadFingerprint
    if name in ("analyze_step", "analyze_stl"):
        from . import analyze

        return getattr(analyze, name)
    if name in _SUBMODULES:
        # Eager imports used to bind these as package attributes, so
        # `b123d_decompiler.fingerprint.analyze` kept working without an explicit import.
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_SUBMODULES))
