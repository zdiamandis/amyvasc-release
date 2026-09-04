#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["antspyx==0.6.3"]
# ///
"""Prepare the Figure 3 VENAT atlas with ANTsPy 0.6.3 in a separate environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.venat import prepare_venat  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venat",
        type=Path,
        required=True,
        help="Original VENAT_PartialVolume.nii.gz in NLin2009cAsym",
    )
    parser.add_argument(
        "--fixed-template",
        type=Path,
        required=True,
        help="1-mm MNI152NLin6Asym T1w template",
    )
    parser.add_argument(
        "--moving-template",
        type=Path,
        required=True,
        help="1-mm MNI152NLin2009cAsym T1w template",
    )
    parser.add_argument(
        "--fixed-mask",
        type=Path,
        required=True,
        help="Brain mask for the fixed template",
    )
    parser.add_argument(
        "--moving-mask",
        type=Path,
        required=True,
        help="Brain mask for the moving template",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare_venat(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
