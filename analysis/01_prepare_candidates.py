#!/usr/bin/env python3
"""Prepare atlas candidates and displayed composite regions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.atlases import prepare_source_candidates  # noqa: E402
from amyvasc_release.gray_matter import build_gray_matter_support  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--hcp-mmp1-probability", type=Path, required=True)
    result.add_argument("--lh-annot", type=Path, required=True)
    result.add_argument("--rh-annot", type=Path, required=True)
    result.add_argument("--hcp-mmp1-crosswalk", type=Path)
    result.add_argument("--tian-atlas", type=Path, required=True)
    result.add_argument("--tian-labels", type=Path, required=True)
    result.add_argument(
        "--mni-template-brain",
        type=Path,
        required=True,
        help=(
            "FSL MNI152_T1_2mm_brain.nii.gz, used as the brain-extracted "
            "MNI152NLin6Asym T1 template for FAST."
        ),
    )
    result.add_argument(
        "--hcp-subcortical",
        type=Path,
        required=True,
        help=("TemplateFlow tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz."),
    )
    result.add_argument("--minimum-probability", type=float, default=0.20)
    result.add_argument("--flirt-bin", default="flirt")
    result.add_argument("--fast-bin", default="fast")
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    path = prepare_source_candidates(
        reference_path=args.reference,
        hcp_mmp1_probability=args.hcp_mmp1_probability,
        lh_annot=args.lh_annot,
        rh_annot=args.rh_annot,
        crosswalk_path=args.hcp_mmp1_crosswalk,
        tian_atlas=args.tian_atlas,
        tian_labels=args.tian_labels,
        output_dir=args.output_dir,
        minimum_probability=args.minimum_probability,
        flirt_bin=args.flirt_bin,
    )
    support_path, provenance_path = build_gray_matter_support(
        reference_path=args.reference,
        template_path=args.mni_template_brain,
        hcp_subcortical_path=args.hcp_subcortical,
        output_path=args.output_dir / "gray_matter_support.nii.gz",
        fast_bin=args.fast_bin,
    )
    print(path)
    print(support_path)
    print(provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
