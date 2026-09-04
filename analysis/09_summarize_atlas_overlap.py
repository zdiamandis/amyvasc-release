"""Summarize atlas overlap with the peri-amygdalar trace for Figure S7."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.atlas_overlap import (  # noqa: E402
    prepare_harvard_oxford_amygdala_mask,
    summarize_atlas_trace_overlap,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-exclusion-trace", type=Path, required=True)
    parser.add_argument("--primary-trace", type=Path, required=True)
    parser.add_argument("--cit168-p50-mask", type=Path, required=True)
    for threshold in (50, 25):
        parser.add_argument(
            f"--harvard-oxford-maxprob{threshold}",
            type=Path,
            required=True,
            help="Binary amygdala mask, or FSL maxprob atlas with --harvard-oxford-labels",
        )
    parser.add_argument(
        "--harvard-oxford-labels",
        type=Path,
        help="FSL HarvardOxford-Subcortical.xml; extract binary masks beside --output",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ho50 = args.harvard_oxford_maxprob50
    ho25 = args.harvard_oxford_maxprob25
    if args.harvard_oxford_labels is not None:
        masks = []
        for threshold, atlas in ((50, ho50), (25, ho25)):
            mask = prepare_harvard_oxford_amygdala_mask(
                atlas,
                args.harvard_oxford_labels,
                args.output.parent
                / f"harvard_oxford_maxprob{threshold}_amygdala.nii.gz",
            )
            masks.append(mask)
            print(mask)
        ho50, ho25 = masks
    table = summarize_atlas_trace_overlap(
        pre_exclusion_trace_path=args.pre_exclusion_trace,
        primary_trace_path=args.primary_trace,
        cit168_p50_path=args.cit168_p50_mask,
        harvard_oxford_maxprob50_path=ho50,
        harvard_oxford_maxprob25_path=ho25,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, sep="\t", index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
