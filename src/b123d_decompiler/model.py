"""Plan data model: the contract between recognition and code generation.

A :class:`BuildPlan` is stock plus an ordered list of :class:`Op`. Every op carries
the build123d source that produces its tool solid, so the planner can execute that
exact source to validate the op and the emitter has nothing to re-derive. One
source of truth, checked end to end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

#: Op.status values.
PLANNED = "planned"  # not yet validated
OK = "ok"  # tool built, and for a cut it does not eat material that should stay
INERT = "inert"  # tool built but changes nothing (e.g. a fuse inside the stock)
OVERLAPS = "overlaps"  # a cut that removes material present in the reference
UNPROVED = "unproved"  # a speculative op that could not show it only removes empty space
FAILED = "failed"  # the tool source raised


def fmt(value: float, places: int = 4) -> str:
    """Format a float for generated source: fixed places, no trailing noise."""
    text = f"{float(value):.{places}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def fmt_tuple(values, places: int = 4) -> str:
    return "(" + ", ".join(fmt(v, places) for v in values) + ")"


@dataclass
class Op:
    """One modelling operation and the source that builds its tool solid."""

    kind: str  # "cut" | "fuse"
    family: str
    index: int  # feature index in the recognition document
    label: str  # one-line comment for the generated source
    code: list[str]  # lines defining `tool`
    status: str = PLANNED
    note: str = ""
    volume: float | None = None  # tool volume
    overlap: float | None = None  # tool ∩ reference solid (cuts only)
    #: Proposed from evidence that describes a face rather than a feature. It has to
    #: prove it removes only empty space, or it is dropped rather than flagged.
    speculative: bool = False

    @property
    def emitted(self) -> bool:
        """Ops that earn a place in the generated script."""
        return self.status in (OK, OVERLAPS)


@dataclass
class BuildPlan:
    source: str
    frame: dict
    stock_label: str
    stock_code: list[str]  # lines defining `part`
    ops: list[Op] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)  # family -> count, no adapter
    refusals: list[str] = field(default_factory=list)
    association: dict = field(default_factory=dict)
    stock_trials: list = field(default_factory=list)  # billets tried and how each ended
    timings: dict = field(default_factory=dict)  # seconds spent in each stage

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> BuildPlan:
        raw = json.loads(text)
        ops = [Op(**o) for o in raw.pop("ops", [])]
        return cls(ops=ops, **raw)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for op in self.ops:
            out[op.status] = out.get(op.status, 0) + 1
        return out
