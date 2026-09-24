"""The command surface, exercised the way a user meets it."""

from __future__ import annotations

import json

from build123d import Align, Box, Pos, export_step

from b123d_decompiler.cli import main


def plate_with_pocket():
    return Box(60, 40, 15) - Pos(0, 0, 7.5) * Box(
        24, 12, 6, align=(Align.CENTER, Align.CENTER, Align.MAX)
    )


def test_decompile_writes_a_script_and_a_plan(tmp_path):
    step = tmp_path / "part.step"
    export_step(plate_with_pocket(), str(step))
    script, plan = tmp_path / "part.py", tmp_path / "plan.json"

    assert main([str(step), "-o", str(script), "--plan", str(plan)]) == 0

    assert "from build123d import" in script.read_text()
    saved = json.loads(plan.read_text())
    assert saved["source"] == "part.step"
    families = [op["family"] for op in saved["ops"] if not op["speculative"]]
    assert families == ["section_recesses"]


def test_analyse_scores_a_step_against_itself(tmp_path):
    step = tmp_path / "part.step"
    export_step(plate_with_pocket(), str(step))
    report = tmp_path / "report.json"

    assert main(["analyse", str(step), str(step), "--json", str(report),
                 "--no-surface"]) == 0

    scored = json.loads(report.read_text())
    assert scored["iou"] > 0.999
    assert scored["status"] == "ok"


def test_analyse_reports_a_script_that_will_not_run(tmp_path):
    step = tmp_path / "part.step"
    export_step(plate_with_pocket(), str(step))
    broken = tmp_path / "broken.py"
    broken.write_text("raise SystemExit('no part here')\n")
    report = tmp_path / "report.json"

    assert main(["analyse", str(step), str(broken), "--json", str(report)]) == 1

    scored = json.loads(report.read_text())
    assert scored["status"] == "error"
    assert "no part here" in scored["execution"]["stderr"]


def test_analyse_refuses_a_script_that_exports_nothing_solid(tmp_path):
    """A script can succeed and still export an empty shape. Score nothing then."""
    step = tmp_path / "part.step"
    export_step(plate_with_pocket(), str(step))
    empty = tmp_path / "empty.py"
    empty.write_text(
        "import sys\n"
        "from build123d import Box, export_step\n"
        "export_step(Box(10, 10, 10) - Box(20, 20, 20), sys.argv[1])\n"
    )
    report = tmp_path / "report.json"

    assert main(["analyse", str(step), str(empty), "--json", str(report)]) == 1
    assert json.loads(report.read_text())["status"] == "invalid_solid"


def test_every_result_says_how_it_reads(tmp_path):
    """The aim is the designer's approach, so a result carries more than its IoU."""
    from pathlib import Path

    from b123d_decompiler.pipeline import run_one

    step = tmp_path / "pocket.step"
    export_step(plate_with_pocket(), str(step))
    record = run_one(Path(step), tmp_path / "out", samples=200)
    reading = record["readability"]
    assert reading["stock_kind"] in {"turned", "outline", "box", "other"}
    assert reading["script_lines"] > 0 and reading["stock_lines"] > 0
    assert reading["feature_ops"] >= 1
    assert 0.0 <= reading["feature_share"] <= 1.0
