"""Unit tests for the parts of the pipeline that need no CAD kernel."""

from __future__ import annotations

import math

import pytest

from b123d_decompiler.adapters import _segments
from b123d_decompiler.model import BuildPlan, Op, fmt, fmt_tuple


def test_fmt_trims_without_losing_value():
    assert fmt(1.50000) == "1.5"
    assert fmt(2.0) == "2"
    assert fmt(-0.00001) == "0"  # below the printed precision, not a negative zero
    assert fmt(1.23456789) == "1.2346"
    assert fmt_tuple((1.0, -2.5, 0.0)) == "(1, -2.5, 0)"


def test_closed_profile_wraps_to_the_first_point():
    boundary = [
        {"point": [0.0, 0.0], "bulge": 0.0},
        {"point": [2.0, 0.0], "bulge": 0.5},
        {"point": [2.0, 1.0], "bulge": 0.0},
    ]
    segments = _segments(boundary, "closed")
    assert [s[0] for s in segments] == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)]
    assert segments[-1][1] == (0.0, 0.0), "closed profiles return to the start"
    # A bulge belongs to the segment leaving its point, DXF style.
    assert segments[1][2] == pytest.approx(0.5)
    assert segments[0][2] == 0.0


def test_open_profile_is_closed_with_a_straight_chord():
    boundary = [
        {"point": [-1.0, 0.0], "bulge": -1.0},
        {"point": [1.0, 0.0], "bulge": 0.0},
    ]
    segments = _segments(boundary, "open")
    assert len(segments) == 2
    assert segments[0][2] == pytest.approx(-1.0)
    assert segments[1] == ((1.0, 0.0), (-1.0, 0.0), 0.0), "the chord carries no bulge"


def test_degenerate_segments_are_dropped():
    boundary = [
        {"point": [0.0, 0.0], "bulge": 0.0},
        {"point": [0.0, 0.0], "bulge": 0.0},
        {"point": [1.0, 0.0], "bulge": 0.0},
        {"point": [1.0, 1.0], "bulge": 0.0},
    ]
    assert all(math.dist(a, b) > 0 for a, b, _ in _segments(boundary, "closed"))


def test_plan_round_trips_through_json():
    plan = BuildPlan(
        source="part.step",
        frame={"origin": [0, 0, 0], "x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]},
        stock_label="stock",
        stock_code=["part = Box(1, 1, 1)"],
        ops=[Op("cut", "holes", 0, "hole", ["tool = Cylinder(1, 2)"], status="ok")],
        skipped={"chamfers": 2},
    )
    restored = BuildPlan.from_json(plan.to_json())
    assert restored.ops[0].family == "holes"
    assert restored.ops[0].emitted
    assert restored.skipped == {"chamfers": 2}


def test_only_buildable_ops_reach_the_script():
    assert Op("cut", "f", 0, "l", [], status="ok").emitted
    assert Op("cut", "f", 0, "l", [], status="overlaps").emitted
    assert not Op("cut", "f", 0, "l", [], status="inert").emitted
    assert not Op("cut", "f", 0, "l", [], status="failed").emitted
    assert not Op("cut", "f", 0, "l", [], status="planned").emitted
