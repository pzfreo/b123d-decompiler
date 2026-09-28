"""Assemble the public regression corpus that CI runs every night.

    uv run --with huggingface_hub python tools/regression_corpus.py QUIDDITY_CHECKOUT OUT_DIR

Thirteen parts, all from public sources, under the names the baselines use: MFCAD++ and
NIST parts and one CADGenBench fixture from quiddity's own test corpus, and four
CADGenBench editing inputs (ODC-BY) from the dataset on Hugging Face, pinned to one
revision so the inputs cannot change under the baseline.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

#: name in the baseline -> path inside a quiddity checkout.
FROM_QUIDDITY = {
    "cgb-flanged_spool_132.step": "tests/corpus/cadgenbench/flanged_spool_132.step",
    "h205.step": "tests/corpus/mfcadpp_holdout/205.step",
    "h238.step": "tests/corpus/mfcadpp_holdout/238.step",
    "h419.step": "tests/corpus/mfcadpp_holdout/419.step",
    "h467.step": "tests/corpus/mfcadpp_holdout/467.step",
    "m10247.step": "tests/corpus/mfcadpp/10247.step",
    "nist-nist_ftc_06_asme1_rd.step": "tests/corpus/nist/nist_ftc_06_asme1_rd.stp",
    "nist-nist_ftc_08_asme1_rc.step": "tests/corpus/nist/nist_ftc_08_asme1_rc.stp",
    "nist-nist_ftc_10_asme1_rb.step": "tests/corpus/nist/nist_ftc_10_asme1_rb.stp",
}

DATASET = "HuggingAI4Engineering/cadgenbench-data"
REVISION = "f76f965585817c621d6ea0d150d745adf670e66e"
FROM_DATASET = {f"cgb{case}.step": f"{case}/input.step" for case in (212, 215, 225, 249)}


def assemble(quiddity: Path, out: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, relative in FROM_QUIDDITY.items():
        written.append(Path(shutil.copyfile(quiddity / relative, out / name)))
    for name, filename in FROM_DATASET.items():
        cached = hf_hub_download(
            DATASET, filename, repo_type="dataset", revision=REVISION, token=False
        )
        written.append(Path(shutil.copyfile(cached, out / name)))
    return written


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    for path in assemble(Path(args[0]), Path(args[1])):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
