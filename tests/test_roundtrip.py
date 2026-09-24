"""Round-trip tests: build a part, export it, decompile it, measure the rebuild.

These are the only tests with ground truth. Each fixture is built from known
features, so a rebuild that scores below tolerance is a real regression in an
adapter rather than a judgement call about a corpus part.
"""

from __future__ import annotations

import pytest
from build123d import Align, Axis, Box, Cylinder, Pos, chamfer, export_step, fillet

from b123d_decompiler import model
from b123d_decompiler.pipeline import analyse, decompile

PLATE = (60, 40, 15)


def through_hole():
    return Box(*PLATE) - Pos(10, 5, 0) * Cylinder(4, 40)


def blind_hole():
    return Box(*PLATE) - Pos(-10, 0, 7.5) * Cylinder(
        3, 8, align=(Align.CENTER, Align.CENTER, Align.MAX)
    )


def pocket():
    return Box(*PLATE) - Pos(0, 0, 7.5) * Box(
        24, 12, 6, align=(Align.CENTER, Align.CENTER, Align.MAX)
    )


def counterbore():
    part = Box(*PLATE) - Cylinder(3, 40)
    return part - Pos(0, 0, 7.5) * Cylinder(
        6, 4, align=(Align.CENTER, Align.CENTER, Align.MAX)
    )


def boss():
    return Box(*PLATE) + Pos(0, 0, 7.5) * Cylinder(
        7, 6, align=(Align.CENTER, Align.CENTER, Align.MIN)
    )


def through_step():
    """A rectangular corner run clear through the plate."""
    return Box(*PLATE) - Pos(-22, 14, 0) * Box(16, 12, 40)


def chamfer_edge():
    plate = Box(*PLATE)
    return chamfer(plate.edges().filter_by(Axis.Y).sort_by(Axis.X)[-1], 4)


def corner_round():
    plate = Box(*PLATE)
    return fillet(plate.edges().filter_by(Axis.Y).sort_by(Axis.X)[-1], 5)


def quarter_round_corner():
    """A quarter cylinder taken out of a corner, stopping part way along it."""
    tool = Pos(30, 20, 0) * Cylinder(9, 26, align=(Align.CENTER, Align.CENTER, Align.MIN))
    corner = Pos(25.5, 15.5, 0) * Box(9, 9, 26, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return Box(*PLATE) - (corner & tool)


EXACT = {
    "through_hole": through_hole,
    "blind_hole": blind_hole,
    "pocket": pocket,
    "counterbore": counterbore,
    "through_step": through_step,
    "chamfer_edge": chamfer_edge,
    "corner_round": corner_round,
    "quarter_round_corner": quarter_round_corner,
}


def run(factory, name, tmp_path, *, surface=False):
    step = tmp_path / f"{name}.step"
    export_step(factory(), str(step))
    script = tmp_path / f"{name}.py"
    plan, source = decompile(step, script)
    report = analyse(step, script, tmp_path / f"{name}.rebuilt.step", samples=500,
                     surface=surface)
    return plan, source, report


@pytest.mark.parametrize("name", sorted(EXACT))
def test_feature_rebuilds_exactly(name, tmp_path):
    plan, _source, report = run(EXACT[name], name, tmp_path)
    assert report["status"] == "ok", report.get("execution")
    features = [op for op in plan.ops if not op.speculative]
    assert features, "the fixture should recognise at least one feature"
    assert [op.status for op in features] == [model.OK] * len(features)
    # A speculative trim may fail to prove itself and be dropped; it must never be
    # kept while cutting material the part keeps.
    assert not [op for op in plan.ops if op.status == model.OVERLAPS]
    assert report["iou"] > 0.99, f"IoU {report['iou']:.4f}"
    assert report["missing_pct"] < 0.5
    assert report["extra_pct"] < 0.5


def test_rebuild_lands_in_the_coordinates_of_the_source(tmp_path):
    _plan, _source, report = run(pocket, "pocket", tmp_path)
    # The model is built in quiddity's local frame and mapped back; a wrong inverse
    # shows up here as a displaced envelope rather than as a quietly poor IoU.
    assert report["bbox_offset"] < 0.01
    assert report["centre_of_mass_offset"] < 0.01


def test_surface_deviation_is_measured(tmp_path):
    _plan, _source, report = run(blind_hole, "blind_hole", tmp_path, surface=True)
    assert "error" not in report["hausdorff"]
    assert report["hausdorff"]["hausdorff"] < 0.5


def test_generated_script_is_standalone(tmp_path):
    _plan, source, _report = run(through_hole, "through_hole", tmp_path)
    assert "b123d_decompiler" not in source
    assert source.count("from build123d import *") == 1
    compile(source, "generated.py", "exec")


def test_a_boss_is_accounted_for_once_the_stock_is_the_parts_outline(tmp_path):
    """A boss has to be in the rebuild, whether the stock left room for it or not.

    Under bounding-box stock this fixture scored 0.733: the envelope buried the boss
    and the op was dropped, but the envelope also left the material around it. Stock
    cut from the part's own outline follows the boss instead, so it is accounted for
    either by the stock or by the op. Which of the two it is depends on how tight the
    outline comes out, and is not the point; the point is that the boss is there and
    that nothing cut it away.
    """
    plan, _source, report = run(boss, "boss", tmp_path)
    assert [op.family for op in plan.ops if not op.speculative] == ["bosses"]
    assert plan.ops[0].kind == "fuse"
    assert plan.ops[0].status in (model.OK, model.INERT)
    assert "outline" in plan.stock_label
    assert report["iou"] > 0.98, f"IoU {report['iou']:.4f}"
    assert report["missing_pct"] < 0.5


def test_identical_solids_score_a_perfect_match(tmp_path):
    """Coincident faces are the case OCCT gets wrong, so pin it.

    A rebuild's faces sit exactly on the reference's. At default tolerance the
    intersection can come back empty while reporting success, which scored a good
    reconstruction as zero until the boolean learned to retry.
    """
    from quiddity import import_step_geometry

    from b123d_decompiler.compare import compare_parts

    step = tmp_path / "same.step"
    export_step(pocket(), str(step))
    report = compare_parts(
        import_step_geometry(str(step)), import_step_geometry(str(step)), surface=False
    )
    assert report["iou"] == pytest.approx(1.0, abs=1e-6)
    assert report["missing_pct"] == pytest.approx(0.0, abs=1e-6)


def test_disjoint_solids_share_nothing_and_need_no_excuse():
    from b123d_decompiler.geom import _volume, robust_common

    left = Pos(-50, 0, 0) * Box(10, 10, 10)
    right = Pos(50, 0, 0) * Box(10, 10, 10)
    shape, note = robust_common(left.wrapped, right.wrapped)
    assert note == "", "separated shapes are a real answer, not a boolean failure"
    assert (_volume(shape) if shape is not None else 0.0) == pytest.approx(0.0)


def test_a_round_that_is_not_a_plain_corner_is_left_alone(tmp_path):
    """Refusing to model a record beats placing it somewhere plausible.

    The fillet adapter reads a radius and a face centre, which describe a round on a
    convex corner of the material. Where the two supporting planes are not that, the
    record is reported unmodelled rather than cut in the wrong place, because a wrong
    cut removes material the part keeps and a missing one only leaves some behind.
    """
    from b123d_decompiler.adapters import fillet as fillet_adapter
    from b123d_decompiler.geom import Context

    step = tmp_path / "plain.step"
    export_step(Box(*PLATE), str(step))
    from quiddity import import_step_geometry

    ctx = Context(import_step_geometry(str(step)))
    feature = {
        "index": 0,
        "record": {"axis": "y", "radius": 5.0, "at": [0.0, 0.0, 0.0], "turned": False},
    }
    # A radius reported at the centre of a solid block fits no corner of it.
    assert fillet_adapter(feature, ctx) is None


def pocket_with_rounded_corners():
    """A pocket whose two long bottom corners are rounded, which quiddity calls a blend."""
    from build123d import fillet as fillet_edges

    plate = pocket()
    edges = plate.edges().filter_by(Axis.X).group_by(Axis.Z)[1].sort_by(Axis.Y)[1:3]
    return fillet_edges(edges, 2)


def test_an_internal_blend_is_material_to_put_back(tmp_path):
    """A concave blend fills an internal corner, so it is an addition, not a cut.

    Quiddity's Blend record states its own sense and its own rolling path, so neither
    has to be recovered from the geometry the way a Fillet record's do.
    """
    from quiddity import import_step_geometry
    from quiddity.document import build_recognition_document

    from b123d_decompiler.adapters import blend
    from b123d_decompiler.geom import Context, run_source
    from b123d_decompiler.recognise import to_local

    step = tmp_path / "blend.step"
    export_step(pocket_with_rounded_corners(), str(step))
    part = import_step_geometry(str(step))
    document = build_recognition_document(part)
    ctx = Context(to_local(part, document["frame"]))

    blends = [f for f in document["features"] if f["family"] == "blends"]
    assert blends, "the fixture should carry a blend"
    assert blends[0]["record"]["side"] == "concave"

    op = blend(blends[0], ctx)
    assert op is not None, "a concave blend on a plain corner should be modelled"
    assert op.kind == "fuse"
    assert run_source(op.code).volume > 0


def test_a_rounded_slot_cut_along_y_is_not_mirrored():
    """The section plane for a y-depth slot has its second axis pointing down z.

    The slot adapter draws its own rounded profile, and for a while did so without the
    correction the other section adapters apply, so every such slot came out mirrored
    across the part. On NIST FTC 10 that took 18 % of the part away. This is that
    record, and the tool has to sit where the record says.
    """
    from build123d import Box

    from b123d_decompiler.adapters import slot
    from b123d_decompiler.geom import Context, run_source

    record = {
        "width_axis": "x", "long_axis": "z", "width": 50.0, "length": 78.0,
        "w_center": 0.01, "lo": -63.02, "hi": 14.98, "d_lo": -14.61, "d_hi": 0.39,
        "end_radius": None, "corner_radius": 2.0,
    }
    op = slot({"record": record, "index": 0}, Context(Box(80, 40, 160)))
    box = run_source(op.code).bounding_box()
    assert abs(box.min.Z - -63.02) < 1e-3 and abs(box.max.Z - 14.98) < 1e-3
    assert abs(box.min.X - -24.99) < 1e-3 and abs(box.max.X - 25.01) < 1e-3
    assert abs(box.min.Y - -14.61) < 1e-3 and abs(box.max.Y - 0.39) < 1e-3


def test_an_unverified_plan_still_builds_its_features(tmp_path):
    """When checking the ops is not possible, the recognised features still go in.

    The batch retries a part without verification after the kernel crashes in it. For
    a while that plan emitted nothing at all, because only a checked op is emitted,
    and the part came back as bare stock.
    """
    from b123d_decompiler.pipeline import decompile

    step = tmp_path / "pocket.step"
    export_step(pocket(), str(step))
    plan, _source = decompile(step, tmp_path / "pocket.py", verify=False)
    features = [op for op in plan.ops if not op.speculative]
    assert features and all(op.emitted for op in features)
    assert not any(op.emitted for op in plan.ops if op.speculative)


def test_the_billet_kept_is_the_one_that_ended_best(tmp_path):
    """Several billets are carried through to a finished rebuild and the best is kept.

    Which billet ends best cannot be read off the billet, so the choice is made on
    the finished rebuilds. Within a small tie margin the simpler billet wins.
    """
    from b123d_decompiler.plan import STOCK_TIE
    from b123d_decompiler.pipeline import decompile

    step = tmp_path / "boss.step"
    export_step(boss(), str(step))
    plan, _source = decompile(step, tmp_path / "boss.py")
    trials = plan.stock_trials
    if len(trials) < 2:
        return  # only one billet was worth trying on this part
    chosen = [trial for trial in trials if trial["chosen"]]
    assert len(chosen) == 1
    assert chosen[0]["stock"] == plan.stock_label
    assert max(trial["iou"] for trial in trials) - chosen[0]["iou"] <= STOCK_TIE
