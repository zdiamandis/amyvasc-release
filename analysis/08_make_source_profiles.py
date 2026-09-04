#!/usr/bin/env python3
"""Compare 23-condition peri-amygdalar and candidate-region profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.source_profiles import (  # noqa: E402
    CONDITIONS,
    CandidateSpec,
    analyze_profile,
    analyze_s4_positive_controls,
    extract_profiles,
    extract_reference_support,
    extract_target_voxels,
    load_candidates,
    load_s4_positive_controls,
    load_targets,
    modeled_blur,
    read_subjects,
    segment_grid,
    validate_s4_positive_control_support,
)
from amyvasc_release.masks import require_binary_mask  # noqa: E402

S4_UNRESTRICTED_TARGETS = (
    "glasser__v2",
    "glasser__lbelt",
    "glasser__mbelt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--fixed-effects-root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--gray-matter-support",
        type=Path,
        required=True,
        help=(
            "FAST pGM>=0.50 union HCP-subcortical support written by "
            "01_prepare_candidates.py; applied only to source candidates."
        ),
    )
    parser.add_argument(
        "--amygdala-mask",
        type=Path,
        required=True,
        help="Binary CIT168 pAmy >= 0.50 mask from 01_prepare_masks.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20_260_710)
    parser.add_argument(
        "--primary-target",
        help="Target key used for the leave-Emotion-out sensitivity analysis.",
    )
    parser.add_argument(
        "--point-spread-target",
        help="Optional target key for the Figure 7 partial-volume blur comparison.",
    )
    parser.add_argument(
        "--amygdala-probability",
        type=Path,
        help="CIT168 probability map required with --point-spread-target.",
    )
    parser.add_argument(
        "--s4-positive-controls",
        type=Path,
        help=(
            "Optional s4_positive_controls.tsv from 01_prepare_masks.py. "
            "Runs the V2/V1 and auditory-belt/A1 atlas screens."
        ),
    )
    return parser


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    if bool(args.point_spread_target) != bool(args.amygdala_probability):
        raise ValueError(
            "--point-spread-target and --amygdala-probability must be supplied together"
        )

    subjects = read_subjects(args.subjects)
    targets = load_targets(args.targets)
    primary_target = args.primary_target
    if primary_target is None:
        declared = [target.key for target in targets if target.primary]
        if len(declared) != 1:
            raise ValueError(
                "Declare one primary target in the target table or pass "
                "--primary-target."
            )
        primary_target = declared[0]
    if primary_target not in {target.key for target in targets}:
        raise ValueError(f"Unknown primary target: {primary_target}")
    candidates = load_candidates(args.candidates)
    if any(candidate.key == "amy_proper" for candidate in candidates):
        raise ValueError("The candidate table must not redeclare amy_proper")
    require_binary_mask(args.amygdala_mask)
    require_binary_mask(args.gray_matter_support)
    candidates = [
        CandidateSpec(
            key="amy_proper",
            label="Amy proper",
            path=args.amygdala_mask,
            values=(1,),
            family="CIT168",
        ),
        *candidates,
    ]

    wide_tables, inventory, reference = extract_profiles(
        subjects=subjects,
        fixed_effects_root=args.fixed_effects_root,
        targets=targets,
        candidates=candidates,
        gray_matter_support_path=args.gray_matter_support,
        unrestricted_profile_keys=(
            S4_UNRESTRICTED_TARGETS if args.s4_positive_controls is not None else ()
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(inventory, args.output_dir / "candidate_inventory.tsv")
    _write(
        pd.DataFrame(
            [
                {
                    "target_key": target.key,
                    "target_label": target.label,
                    "segment": target.segment,
                    "exclusion": target.exclusion,
                    "primary": target.primary,
                    "bootstrap_seed_offset": target.bootstrap_seed_offset,
                    "bootstrap_draws": target.bootstrap_draws or args.bootstrap,
                }
                for target in targets
            ]
        ),
        args.output_dir / "target_inventory.tsv",
    )

    analyses = []
    correlation_tables = []
    paired_tables = []
    commonality_tables = []
    condition_tables = []
    emotion_tables = []
    emotion_paired_tables = []
    for target in targets:
        wide = wide_tables[target.key]
        _write(wide, args.output_dir / "participant_profiles" / f"{target.key}.tsv")
        result = analyze_profile(
            wide,
            target,
            candidates,
            n_bootstrap=target.bootstrap_draws or args.bootstrap,
            seed=args.seed + target.bootstrap_seed_offset,
        )
        analyses.append(result)
        correlation_tables.append(result["correlations"])
        paired_tables.append(result["paired"])
        commonality_tables.append(result["commonality"])
        condition_tables.append(result["profiles"])

        if target.key == primary_target:
            excluded_result = analyze_profile(
                wide,
                target,
                candidates,
                n_bootstrap=target.bootstrap_draws or args.bootstrap,
                seed=args.seed + target.bootstrap_seed_offset,
                exclude_tasks={"emotion"},
            )
            excluded = excluded_result["correlations"].copy()
            excluded.insert(1, "scenario", "exclude_emotion")
            emotion_tables.append(excluded)
            excluded_paired = excluded_result["paired"].copy()
            excluded_paired.insert(1, "scenario", "exclude_emotion")
            emotion_paired_tables.append(excluded_paired)

    _write(
        pd.concat(correlation_tables, ignore_index=True),
        args.output_dir / "source_profile_correlations.tsv",
    )
    _write(
        pd.concat(paired_tables, ignore_index=True),
        args.output_dir / "paired_candidate_differences.tsv",
    )
    _write(
        pd.concat(commonality_tables, ignore_index=True),
        args.output_dir / "pairwise_commonality.tsv",
    )
    _write(
        pd.concat(condition_tables, ignore_index=True),
        args.output_dir / "group_condition_profiles.tsv",
    )
    if emotion_tables:
        _write(
            pd.concat(emotion_tables, ignore_index=True),
            args.output_dir / "emotion_exclusion_sensitivity.tsv",
        )
        _write(
            pd.concat(emotion_paired_tables, ignore_index=True),
            args.output_dir / "emotion_exclusion_paired_candidate_differences.tsv",
        )
    _write(
        segment_grid(analyses, targets, inventory=inventory),
        args.output_dir / "segment_exclusion_grid.tsv",
    )

    if args.s4_positive_controls is not None:
        controls = load_s4_positive_controls(args.s4_positive_controls)
        validate_s4_positive_control_support(
            controls,
            candidates,
            reference,
            inventory=inventory,
            extraction_target_key=primary_target,
        )
        metrics, rankings = analyze_s4_positive_controls(
            wide_tables[primary_target],
            controls,
            candidates,
        )
        _write(
            metrics,
            args.output_dir / "figureS4_pairwise_positive_controls.tsv",
        )
        _write(
            rankings,
            args.output_dir / "figureS4_positive_control_atlas_rankings.tsv",
        )

    if args.point_spread_target:
        target_lookup = {target.key: target for target in targets}
        if args.point_spread_target not in target_lookup:
            raise ValueError(
                f"Unknown --point-spread-target: {args.point_spread_target}"
            )
        target = target_lookup[args.point_spread_target]
        voxel_profiles, voxel_ijk, voxel_hemispheres = extract_target_voxels(
            subjects=subjects,
            fixed_effects_root=args.fixed_effects_root,
            target=target,
            reference=reference,
        )
        wide = wide_tables[target.key]
        amy = wide.groupby(["condition_key", "hemisphere"], sort=False)[
            "amy_proper"
        ].mean()
        condition_order = [condition.key for condition in CONDITIONS]
        amygdala_profiles = {
            hemisphere: np.asarray(
                [amy.loc[(condition, hemisphere)] for condition in condition_order]
            )
            for hemisphere in ("left", "right")
        }
        probability_image = nib.load(str(args.amygdala_probability))
        if probability_image.shape[:3] != reference.shape[:3] or not np.allclose(
            probability_image.affine, reference.affine, atol=1e-4
        ):
            raise ValueError(
                "Amygdala probability map must be on the fixed-effect grid"
            )
        reference_ijk, reference_hemispheres, reference_support = (
            extract_reference_support(
                subjects=subjects,
                fixed_effects_root=args.fixed_effects_root,
                amygdala_mask=args.amygdala_mask,
                gray_matter_support=args.gray_matter_support,
                target=target,
                reference=reference,
            )
        )
        blur = modeled_blur(
            voxel_profiles=voxel_profiles,
            voxel_ijk=voxel_ijk,
            voxel_hemispheres=voxel_hemispheres,
            amygdala_profiles=amygdala_profiles,
            amygdala_probability=np.asarray(probability_image.dataobj, dtype=float),
            reference_ijk=reference_ijk,
            reference_hemispheres=reference_hemispheres,
            reference_support=reference_support,
            reference=reference,
        )
        _write(blur, args.output_dir / "modeled_blur.tsv")

    print(
        f"Wrote source-profile results for {len(targets)} targets to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
