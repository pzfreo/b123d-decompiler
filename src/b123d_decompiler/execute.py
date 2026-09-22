"""Stage 4: run generated source and bring back the solid it exports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def execute(script: Path, step_out: Path, timeout: float = 900.0) -> dict:
    """Run a generated script, asking it to export to `step_out`.

    Returns the outcome rather than raising: a script that fails is a result the
    report has to carry, not an error that stops the batch.
    """
    try:
        done = subprocess.run(
            [sys.executable, str(script), str(step_out)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stderr": f"no result within {timeout:.0f}s"}

    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-12:])
        return {"status": "error", "stderr": tail, "returncode": done.returncode}
    if not step_out.exists():
        return {"status": "error", "stderr": "script exported no STEP file"}
    return {"status": "ok", "stderr": done.stderr.strip()[-400:], "step": str(step_out)}
