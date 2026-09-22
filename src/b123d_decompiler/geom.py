"""Geometry services the planner needs: measurement, point probing, and running
the generated source for one op.

The planner never builds a tool solid by hand. It executes the same source lines
the emitter will write into the script, so a plan that validates and a script that
runs cannot disagree about what an op means.
"""

from __future__ import annotations

import math

from build123d import Part
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON


def dot(a, b) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def unit(v):
    length = math.sqrt(dot(v, v))
    return tuple(float(x) / length for x in v) if length else (0.0, 0.0, 1.0)


class Context:
    """Measurements of the local reference solid, shared by every adapter."""

    def __init__(self, part: Part):
        self.part = part
        box = part.bounding_box()
        self.bb_min = (box.min.X, box.min.Y, box.min.Z)
        self.bb_max = (box.max.X, box.max.Y, box.max.Z)
        self.diagonal = math.dist(self.bb_min, self.bb_max)
        #: How far a tool runs past the stock when a feature is open-ended.
        self.margin = max(self.diagonal * 0.05, 1.0)
        self._classifier = BRepClass3d_SolidClassifier(part.wrapped)

    def note_cylinders(self, cylinders) -> None:
        """Record the round features a turned treatment can sit on.

        A turned chamfer or fillet is a cone or a torus swept about some axis, and its
        record says where that axis is but not how wide the thing is. The hole or the
        shaft it sits on does, so the two are matched up.
        """
        self.cylinders = list(cylinders)

    def sweep_axes(self, direction, point, limit: int = 3):
        """Axis lines a turned treatment could be swept about, nearest first.

        A turned chamfer's reported point lies on the cone itself, part way out from
        the axis, not on the axis. So the axis has to come from the round features the
        part already has, and the reported point's distance from it gives the radius.
        Only the nearest few are worth trying: a part with dozens of holes would
        otherwise cost thousands of booleans to place one chamfer.
        """
        lines = []
        for axis, anchor, _diameter, _kind in getattr(self, "cylinders", ()):
            if abs(abs(dot(axis, direction)) - 1.0) > 1e-6:
                continue
            lines.append((self.radius_about(point, axis, anchor), axis, anchor))
        lines.sort(key=lambda entry: entry[0])
        seen, chosen = set(), []
        for away, axis, anchor in lines:
            key = round(away, 3)
            if key in seen:
                continue
            seen.add(key)
            chosen.append((axis, anchor))
            if len(chosen) >= limit:
                break
        return chosen

    def radius_about(self, point, axis, anchor) -> float:
        offset = tuple(point[k] - anchor[k] for k in range(3))
        along = dot(offset, axis)
        return math.sqrt(max(sum(v * v for v in offset) - along * along, 0.0))

    def on_axis(self, point, axis, anchor):
        """The point on an axis line level with a given point."""
        offset = tuple(point[k] - anchor[k] for k in range(3))
        along = dot(offset, axis)
        return tuple(anchor[k] + axis[k] * along for k in range(3))

    @property
    def envelope(self):
        """The part's bounding box as a solid, built once and kept."""
        cached = getattr(self, "_envelope", None)
        if cached is None:
            from build123d import Box, Pos

            size = tuple(self.bb_max[k] - self.bb_min[k] for k in range(3))
            centre = tuple((self.bb_max[k] + self.bb_min[k]) / 2 for k in range(3))
            cached = Pos(*centre) * Box(*size)
            self._envelope = cached
        return cached

    def held_by(self, stock, points: int = 4000) -> float:
        """What share of the part a candidate billet actually contains.

        Asked by classifying points rather than by intersecting solids. The billet's
        faces coincide with the part's wherever it was turned to size, which is exactly
        the input the kernel gets wrong: it returns an empty intersection and the
        candidate is thrown away for holding none of a part it holds entirely.
        """
        import random

        inside = Context(stock)
        generator = random.Random(20260921)
        found = held = 0
        for _ in range(points):
            spot = tuple(
                generator.uniform(self.bb_min[k], self.bb_max[k]) for k in range(3)
            )
            if not self.inside_solid(spot):
                continue
            found += 1
            if inside.inside_solid(spot):
                held += 1
        return held / found if found else 0.0

    def corners(self):
        for i in range(8):
            yield tuple(
                (self.bb_max if (i >> k) & 1 else self.bb_min)[k] for k in range(3)
            )

    def extent_along(self, direction) -> tuple[float, float]:
        """Min and max of the bounding box projected onto a direction."""
        values = [dot(c, direction) for c in self.corners()]
        return min(values), max(values)

    def inside_box(self, point, slack: float = 1e-6) -> bool:
        return all(
            self.bb_min[k] - slack <= point[k] <= self.bb_max[k] + slack for k in range(3)
        )

    def inside_solid(self, point, tol: float = 1e-6) -> bool:
        self._classifier.Perform(gp_Pnt(*[float(v) for v in point]), tol)
        return self._classifier.State() in (TopAbs_IN, TopAbs_ON)

    def _perpendicular(self, direction):
        """Any unit vector at right angles to a direction."""
        smallest = min(range(3), key=lambda k: abs(direction[k]))
        seed = [0.0, 0.0, 0.0]
        seed[smallest] = 1.0
        cross = (
            direction[1] * seed[2] - direction[2] * seed[1],
            direction[2] * seed[0] - direction[0] * seed[2],
            direction[0] * seed[1] - direction[1] * seed[0],
        )
        return unit(cross)

    def _samples(self, origin, direction, distance: float, radius: float, reach: float = 0.6):
        """Points along a ray, and around it when the feature has a radius.

        A bore with a pin standing in it is solid along its own axis, so an on-axis
        probe finds no void either way and cannot tell which way the hole was drilled.
        Sampling off-axis as well is what makes an annular feature readable.
        """
        offsets = [(0.0, 0.0, 0.0)]
        if radius > 0.0:
            first = self._perpendicular(direction)
            second = (
                direction[1] * first[2] - direction[2] * first[1],
                direction[2] * first[0] - direction[0] * first[2],
                direction[0] * first[1] - direction[1] * first[0],
            )
            step_out = radius * reach
            for vector in (first, second):
                offsets.append(tuple(v * step_out for v in vector))
                offsets.append(tuple(-v * step_out for v in vector))
        for fraction in (0.2, 0.5, 0.8):
            for offset in offsets:
                yield tuple(
                    origin[k] + direction[k] * distance * fraction + offset[k] for k in range(3)
                )

    def void_score(self, origin, direction, distance: float, radius: float = 0.0) -> int:
        """How many samples lie in a void: inside the envelope, outside the solid.

        This is what tells a blind hole which way it was drilled. Quiddity canonicalises
        record axes, so the published axis need not point into the material.
        """
        return sum(
            1
            for point in self._samples(origin, direction, distance, radius)
            if self.inside_box(point) and not self.inside_solid(point)
        )

    def material_score(self, origin, direction, distance: float, radius: float = 0.0) -> int:
        """How many samples lie in material. Used for additive features.

        Sampled near the rim rather than mid-way out. A boss standing on the end of a
        shaft is solid along its own axis in both directions, so a probe that stays
        close to the axis cannot tell which way the boss actually goes.
        """
        return sum(
            1
            for point in self._samples(origin, direction, distance, radius, reach=0.85)
            if self.inside_solid(point)
        )


def run_source(code: list[str], name: str = "tool") -> Part:
    """Execute generated source and return the shape it binds to `name`."""
    import build123d

    namespace = {key: getattr(build123d, key) for key in dir(build123d)}
    exec("\n".join(code), namespace)  # noqa: S102 - our own generated source
    shape = namespace.get(name)
    if shape is None:
        raise ValueError(f"generated source defined no `{name}`")
    return shape


def _volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return max(float(properties.Mass()), 0.0)


#: Fuzzy tolerances tried when a boolean returns nothing but the shapes clearly meet.
#: A rebuild's faces coincide with the reference's by construction, and OCCT can
#: report success with an empty result on exactly that input.
FUZZY_RETRIES = (1e-3, 1e-2)


def _bounding_box(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    return box


def _boxes_meet(a, b) -> bool:
    return not _bounding_box(a).IsOut(_bounding_box(b))


def _plain_common(a, b, fuzzy: float = 0.0):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.TopTools import TopTools_ListOfShape

    builder = BRepAlgoAPI_Common()
    arguments, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    arguments.Append(a)
    tools.Append(b)
    builder.SetArguments(arguments)
    builder.SetTools(tools)
    if fuzzy:
        builder.SetFuzzyValue(fuzzy)
    builder.Build()
    return builder.Shape() if builder.IsDone() else None


def robust_common(a, b) -> tuple[object | None, str]:
    """Intersect two raw shapes, and say so when the answer could not be trusted.

    An empty intersection between shapes whose envelopes overlap is usually OCCT
    giving up rather than a real answer, and reporting it as "no shared volume" turns
    a good rebuild into a zero score. A failure does not always come back as exactly
    zero either: a result of 1e-14 against two solids of several thousand is the same
    failure wearing a number, so the answer has to be significant as well as positive.
    """
    floor = 1e-9 * min(_volume(a), _volume(b))

    def usable(shape) -> bool:
        return shape is not None and _volume(shape) > floor

    shape = _plain_common(a, b)
    if usable(shape):
        return shape, ""
    if not _boxes_meet(a, b):
        return shape, ""
    for fuzzy in FUZZY_RETRIES:
        retried = _plain_common(a, b, fuzzy)
        if usable(retried):
            return retried, f"intersection needed a {fuzzy:g} mm fuzzy tolerance"
    return None, "intersection came back empty although the envelopes overlap"


def _common(a, b):
    """Intersection of two raw OCCT shapes, or None when the boolean fails."""
    return robust_common(a, b)[0]


def common_volume(a: Part, b: Part) -> float:
    """Volume of the intersection of two solids, 0.0 when they do not meet."""
    shape = _common(a.wrapped, b.wrapped)
    return _volume(shape) if shape is not None else 0.0


def shared_material(tool: Part, other: Part, reference: Part) -> tuple[float, float]:
    """(volume shared by two tools, how much of it is material in the reference)."""
    shared = _common(tool.wrapped, other.wrapped)
    if shared is None:
        return 0.0, 0.0
    material = _common(shared, reference.wrapped)
    return _volume(shared), (_volume(material) if material is not None else 0.0)


def overlap_after_restore(tool: Part, reference: Part, restorers) -> float:
    """Volume of `tool` that eats reference material no restorer puts back.

    A cut through a pin standing in a bore overlaps material legitimately, because a
    later fuse returns it. Measuring the cut alone would call that op wrong.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

    shape = _common(tool.wrapped, reference.wrapped)
    if shape is None:
        return 0.0
    for restorer in restorers:
        cut = BRepAlgoAPI_Cut(shape, restorer.wrapped)
        if not cut.IsDone():
            return _volume(shape)
        shape = cut.Shape()
    return _volume(shape)
