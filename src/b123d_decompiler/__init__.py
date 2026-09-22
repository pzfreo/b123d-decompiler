"""b123d-decompiler: STEP to build123d source, via quiddity feature recognition.

The pipeline is five stages, each leaving a file behind so a bad result can be
traced to the stage that lost it: recognise, plan, emit, execute, compare.

Names that need a CAD kernel resolve on first use rather than at import. The
vendored :mod:`b123d_decompiler.fingerprint` promises that comparing two saved
fingerprints works with nothing but the standard library, and a package whose
import pulled in OCCT would quietly break that promise for everything beneath it.
"""

__all__ = [
    "BuildPlan", "Op", "analyse", "build_plan", "compare_parts",
    "decompile", "measure", "recognise", "render",
]

_ORIGIN = {
    "BuildPlan": "model",
    "Op": "model",
    "render": "emit",
    "build_plan": "plan",
    "recognise": "recognise",
    "compare_parts": "compare",
    "measure": "compare",
    "analyse": "pipeline",
    "decompile": "pipeline",
}


def __getattr__(name):
    """Import the stage that owns a name on demand (PEP 562)."""
    if name in _ORIGIN:
        import importlib

        module = importlib.import_module(f".{_ORIGIN[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(_ORIGIN))
