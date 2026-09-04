#!/usr/bin/env python3
"""Render HCP image panels after group maps, masks, and ROI tables are prepared."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figures.hcp_maps import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-root",
        type=Path,
        required=True,
        help="Output root of 02_fit_hcp_tasks.py --group-maps.",
    )
    parser.add_argument(
        "--background",
        type=Path,
        required=True,
        help="T1 template in MNI152NLin6Asym space.",
    )
    parser.add_argument("--amygdala-mask", type=Path, required=True)
    parser.add_argument("--amygdala-probability", type=Path, required=True)
    parser.add_argument(
        "--bvr-labels",
        type=Path,
        required=True,
        help="Six-label image from 01_prepare_masks.py.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/source_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/image_panels"))
    parser.add_argument("--ho50", type=Path, help="Binary HO maxprob50 amygdala mask.")
    parser.add_argument("--ho25", type=Path, help="Binary HO maxprob25 amygdala mask.")
    args = parser.parse_args()
    paths = render(
        group_root=args.group_root,
        background_path=args.background,
        amygdala_path=args.amygdala_mask,
        probability_path=args.amygdala_probability,
        bvr_labels_path=args.bvr_labels,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        ho50_path=args.ho50,
        ho25_path=args.ho25,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
