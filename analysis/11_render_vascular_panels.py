#!/usr/bin/env python3
"""Render Figure 3 and Supplements S2/S3 from the prepared images in a JSON manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.vascular import load_manifest, render  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="JSON manifest described in analysis/vascular_inputs.md",
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["3", "S2", "S3"], default=["3", "S2", "S3"]
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vascular"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    render(load_manifest(args.inputs), args.figures, args.output_dir, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
