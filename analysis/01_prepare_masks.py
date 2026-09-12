#!/usr/bin/env python3
"""Prepare CIT168 amygdala and BVR segment masks on an fMRI grid."""

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

from amyvasc_release.masks import (  # noqa: E402
    BVR_LABELS,
    S4_CONTROL_COMPONENT_KEYS,
    build_bvr_label_image,
    build_cit168_probability,
    build_label_mask,
    build_s4_control_masks,
    build_s4_mesencephalic_target,
    candidate_label_values,
    exclude_probability,
    load_image,
    load_labels,
    resample_probability_sinc,
    resample_to_reference,
    save_mask,
    threshold_probability,
    validate_dseg_labels,
)


def parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse ordered, unique probability thresholds."""
    thresholds = tuple(dict.fromkeys(float(item) for item in value.split(",")))
    invalid = any(not np.isfinite(item) or not 0 <= item <= 1 for item in thresholds)
    if not thresholds or invalid:
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated values in [0, 1]"
        )
    return thresholds


def threshold_tag(threshold: float) -> str:
    """Return a compact whole-percent filename tag."""
    return f"{int(round(threshold * 100)):02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="A 3D or 4D image defining the target fMRI grid.",
    )
    parser.add_argument("--cit168-pseg", type=Path, required=True)
    parser.add_argument(
        "--cit168-labels",
        type=Path,
        required=True,
        help="CIT168 label names in pseg volume order.",
    )
    parser.add_argument(
        "--cit168-dseg",
        type=Path,
        help="Optional CIT168 dseg used to validate label indices.",
    )
    bvr = parser.add_mutually_exclusive_group(required=True)
    bvr.add_argument(
        "--bvr-labels",
        type=Path,
        help="Existing BVR label image: 1/2 striate, 3/4 peduncular, 5/6 mesencephalic.",
    )
    bvr.add_argument(
        "--bvr-group-effect",
        type=Path,
        help="HCP group fear-minus-shape effect map used to reconstruct the labels.",
    )
    parser.add_argument(
        "--bvr-effect-threshold",
        type=float,
        default=0.50,
        help="Percent-signal-change threshold used with --bvr-group-effect.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=(0.50, 0.20, 0.05),
        help="Amygdala-overlap exclusions (default: 0.50,0.20,0.05).",
    )
    parser.add_argument(
        "--s4-tian-s2-label-image",
        type=Path,
        help="Tian S2 integer-label image used to exclude bilateral THA-DP.",
    )
    parser.add_argument(
        "--s4-tian-s2-labels",
        type=Path,
        help="Tian S2 labels in integer-value order.",
    )
    parser.add_argument(
        "--s4-hcp-mmp1-label-image",
        type=Path,
        help="Prepared HCP-MMP1 p20 maximum-probability label image.",
    )
    parser.add_argument(
        "--s4-source-candidates",
        type=Path,
        help="Candidate table containing HCP-MMP1 keys and integer values.",
    )
    parser.add_argument("--flirt-bin", default="flirt")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tian_inputs = (args.s4_tian_s2_label_image, args.s4_tian_s2_labels)
    if any(value is not None for value in tian_inputs) and not all(
        value is not None for value in tian_inputs
    ):
        raise ValueError(
            "--s4-tian-s2-label-image and --s4-tian-s2-labels are required together"
        )
    control_inputs = (args.s4_hcp_mmp1_label_image, args.s4_source_candidates)
    if any(value is not None for value in control_inputs) and not all(
        value is not None for value in control_inputs
    ):
        raise ValueError(
            "--s4-hcp-mmp1-label-image and --s4-source-candidates are required together"
        )
    reference = load_image(args.reference)
    pseg = load_image(args.cit168_pseg, ndim=4)
    labels = load_labels(args.cit168_labels)
    if args.cit168_dseg is not None:
        validate_dseg_labels(load_image(args.cit168_dseg, ndim=3), labels)

    probability_native = build_cit168_probability(pseg, labels)
    probability_img = resample_probability_sinc(
        probability_native,
        reference,
        flirt_bin=args.flirt_bin,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = args.output_dir / "cit168_amygdala_probability.nii.gz"
    nib.save(probability_img, probability_path)
    probability = probability_img.get_fdata(dtype=np.float32)

    rows: list[dict[str, object]] = [
        {
            "kind": "probability",
            "segment": "amygdala",
            "hemisphere": "bilateral",
            "pamy_exclusion": np.nan,
            "n_voxels": int(np.count_nonzero(probability > 0)),
            "path": str(probability_path),
        }
    ]
    for threshold in args.thresholds:
        mask = threshold_probability(probability, threshold)
        path = args.output_dir / f"cit168_amygdala_p{threshold_tag(threshold)}.nii.gz"
        save_mask(mask, reference, path)
        rows.append(
            {
                "kind": "binary",
                "segment": "amygdala",
                "hemisphere": "bilateral",
                "pamy_exclusion": threshold,
                "n_voxels": int(mask.sum()),
                "path": str(path),
            }
        )

    if args.bvr_group_effect is not None:
        group_effect = load_image(args.bvr_group_effect, ndim=3)
        if group_effect.shape[:3] != reference.shape[:3] or not np.allclose(
            group_effect.affine, reference.affine
        ):
            raise ValueError("BVR group effect and reference grids differ")
        bvr_img = build_bvr_label_image(
            group_effect,
            threshold=args.bvr_effect_threshold,
        )
        nib.save(bvr_img, args.output_dir / "bvr_labels.nii.gz")
    else:
        bvr_img = resample_to_reference(
            load_image(args.bvr_labels, ndim=3),
            reference,
            interpolation="nearest",
        )
    bvr_data = bvr_img.get_fdata(dtype=np.float32)
    observed = {int(value) for value in np.unique(np.rint(bvr_data)) if value > 0}
    missing = set(range(1, 7)) - observed
    if missing:
        raise ValueError(f"BVR label image is missing values: {sorted(missing)}")

    for segment, hemispheres in BVR_LABELS.items():
        for hemisphere, values in hemispheres.items():
            base = build_label_mask(bvr_data, values)
            base_path = args.output_dir / f"bvr_{segment}_{hemisphere}.nii.gz"
            save_mask(base, reference, base_path)
            rows.append(
                {
                    "kind": "binary",
                    "segment": segment,
                    "hemisphere": hemisphere,
                    "pamy_exclusion": np.nan,
                    "n_voxels": int(base.sum()),
                    "path": str(base_path),
                }
            )
            for threshold in args.thresholds:
                excluded = exclude_probability(base, probability, threshold)
                path = args.output_dir / (
                    f"bvr_{segment}_{hemisphere}_pamy{threshold_tag(threshold)}_excluded.nii.gz"
                )
                save_mask(excluded, reference, path)
                rows.append(
                    {
                        "kind": "binary",
                        "segment": segment,
                        "hemisphere": hemisphere,
                        "pamy_exclusion": threshold,
                        "n_voxels": int(excluded.sum()),
                        "path": str(path),
                    }
                )

    if args.s4_tian_s2_label_image is not None:
        tian_img = resample_to_reference(
            load_image(args.s4_tian_s2_label_image, ndim=3),
            reference,
            interpolation="nearest",
        )
        tian_data = tian_img.get_fdata(dtype=np.float32)
        raw_mesencephalic, s4_target, counts = build_s4_mesencephalic_target(
            bvr_data,
            tian_data,
            load_labels(args.s4_tian_s2_labels),
        )
        raw_path = args.output_dir / "s4_mesencephalic_display_mask.nii.gz"
        target_path = args.output_dir / "s4_mesencephalic_outside_tha_dp.nii.gz"
        save_mask(raw_mesencephalic, reference, raw_path)
        save_mask(s4_target, reference, target_path)
        rows.extend(
            [
                {
                    "kind": "supplementary_display",
                    "segment": "mesencephalic",
                    "hemisphere": "bilateral",
                    "pamy_exclusion": np.nan,
                    "n_voxels": counts.raw_mesencephalic,
                    "path": str(raw_path),
                },
                {
                    "kind": "supplementary_target",
                    "segment": "mesencephalic_outside_tha_dp",
                    "hemisphere": "bilateral",
                    "pamy_exclusion": np.nan,
                    "n_voxels": counts.final_target,
                    "path": str(target_path),
                },
            ]
        )
        pd.DataFrame(
            [
                {
                    "key": "hcp_bvr_mesencephalic_no_thadp",
                    "label": "Mesencephalic BVR outside THA-DP",
                    "path": target_path.name,
                    "segment": "mesencephalic",
                    "exclusion": "Tian S2 THA-DP",
                    "bootstrap_seed_offset": 5,
                    "bootstrap_draws": 2000,
                    "primary": True,
                }
            ]
        ).to_csv(args.output_dir / "s4_targets.tsv", sep="\t", index=False)

    if args.s4_hcp_mmp1_label_image is not None:
        hcp_mmp1_img = resample_to_reference(
            load_image(args.s4_hcp_mmp1_label_image, ndim=3),
            reference,
            interpolation="nearest",
        )
        values = candidate_label_values(
            args.s4_source_candidates,
            S4_CONTROL_COMPONENT_KEYS,
        )
        control_masks = build_s4_control_masks(
            hcp_mmp1_img.get_fdata(dtype=np.float32),
            values,
        )
        filenames = {
            "glasser__v1": "s4_control_v1.nii.gz",
            "glasser__v2": "s4_control_v2.nii.gz",
            "glasser__a1": "s4_control_a1.nii.gz",
            "glasser__lbelt": "s4_control_lbelt.nii.gz",
            "glasser__mbelt": "s4_control_mbelt.nii.gz",
            "auditory_belt": "s4_control_auditory_belt_union.nii.gz",
        }
        mask_paths: dict[str, Path] = {}
        for key, mask in control_masks.items():
            path = args.output_dir / filenames[key]
            save_mask(mask, reference, path)
            mask_paths[key] = path
            rows.append(
                {
                    "kind": "supplementary_control",
                    "segment": key,
                    "hemisphere": "bilateral",
                    "pamy_exclusion": np.nan,
                    "n_voxels": int(mask.sum()),
                    "path": str(path),
                }
            )
        pd.DataFrame(
            [
                {
                    "display_order": 0,
                    "target_key": "v2",
                    "target_label": "Second visual area (V2)",
                    "target_members": "glasser__v2",
                    "target_support_mask": mask_paths["glasser__v2"].name,
                    "target_profile_rule": "weighted z-score",
                    "atlas_rank_exclusions": "glasser__v2",
                    "atlas_screen_n_candidates": 194,
                    "source_label": "Primary visual cortex (V1)",
                    "source_members": "glasser__v1",
                    "source_support_mask": mask_paths["glasser__v1"].name,
                },
                {
                    "display_order": 1,
                    "target_key": "auditory_belt",
                    "target_label": "Auditory belt (LBelt + MBelt)",
                    "target_members": "glasser__lbelt;glasser__mbelt",
                    "target_support_mask": mask_paths["auditory_belt"].name,
                    "target_profile_rule": (
                        "weighted z-score of the mean of separately weighted-"
                        "z-scored member profiles"
                    ),
                    "atlas_rank_exclusions": "glasser__lbelt;glasser__mbelt",
                    "atlas_screen_n_candidates": 193,
                    "source_label": "Primary auditory cortex (A1)",
                    "source_members": "glasser__a1",
                    "source_support_mask": mask_paths["glasser__a1"].name,
                },
            ]
        ).to_csv(args.output_dir / "s4_positive_controls.tsv", sep="\t", index=False)

    table_path = args.output_dir / "masks.tsv"
    inventory = pd.DataFrame(rows)
    inventory.to_csv(table_path, sep="\t", index=False)
    targets = inventory.loc[
        inventory["segment"].isin(BVR_LABELS)
        & inventory["hemisphere"].eq("bilateral")
        & inventory["pamy_exclusion"].notna()
    ].copy()
    targets["key"] = targets.apply(
        lambda row: f"bvr_{row['segment']}_pamy{threshold_tag(row['pamy_exclusion'])}",
        axis=1,
    )
    targets["label"] = targets["segment"].map(
        {
            "striate": "Striate BVR",
            "peduncular": "Peduncular BVR",
            "mesencephalic": "Mesencephalic BVR",
            "striate_peduncular": "Combined peri-amygdalar target",
        }
    )
    targets["exclusion"] = targets["pamy_exclusion"].map(
        lambda value: f"pAmy >= {value:.2f}"
    )
    targets["primary"] = targets["segment"].eq("striate_peduncular") & targets[
        "pamy_exclusion"
    ].eq(0.50)
    targets.loc[targets["primary"], "label"] = "Primary peri-amygdalar target"
    targets["label"] += " (" + targets["exclusion"] + " excluded)"
    targets["bootstrap_seed_offset"] = 0
    targets["path"] = targets["path"].map(lambda value: Path(value).name)
    order = {0.50: 0, 0.20: 1, 0.05: 2}
    segment_order = {
        "striate_peduncular": 0,
        "striate": 1,
        "peduncular": 2,
        "mesencephalic": 3,
    }
    targets["_segment_order"] = targets["segment"].map(segment_order)
    targets["_threshold_order"] = targets["pamy_exclusion"].map(order)
    targets = targets.sort_values(["_segment_order", "_threshold_order"])
    output_columns = [
        "key",
        "label",
        "path",
        "segment",
        "exclusion",
        "primary",
        "bootstrap_seed_offset",
    ]
    targets[output_columns].to_csv(
        args.output_dir / "source_targets.tsv",
        sep="\t",
        index=False,
    )
    figure7_targets = targets.loc[
        targets["segment"].eq("striate_peduncular")
        & targets["pamy_exclusion"].isin((0.50, 0.05))
    ].copy()
    figure7_targets.loc[
        figure7_targets["pamy_exclusion"].eq(0.05), "bootstrap_seed_offset"
    ] = 1
    figure7_targets[output_columns].to_csv(
        args.output_dir / "figure7_targets.tsv",
        sep="\t",
        index=False,
    )
    print(table_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
