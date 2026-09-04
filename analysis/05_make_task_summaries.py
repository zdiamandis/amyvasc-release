#!/usr/bin/env python3
"""Create participant-first ROI, PSTC, and spatial task summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.hcp import TASKS, load_runs, read_subjects  # noqa: E402
from amyvasc_release.masks import load_image, load_mask, same_grid  # noqa: E402
from amyvasc_release.source_profiles import (  # noqa: E402
    candidate_masks,
    load_candidates,
)
from amyvasc_release.task_summaries import (  # noqa: E402
    collect_participant_pstcs,
    directional_trim_masks,
    distance_geometry,
    extract_roi_values,
    figure8_mean_summary,
    fit_distance_gradient,
    load_roi_images,
    load_subnucleus_masks,
    mean_ci,
    roi_value,
    summarize_subnuclei,
    wm_face_load_summary,
)

SPATIAL_CONDITIONS = (
    ("emotion", "emotion_fear", "Emotion fear", "fear_canonical"),
    ("language", "language_story", "Language story", "story_canonical"),
    ("social", "social_mental", "Social mental", "mental_canonical"),
    ("wm", "wm_0bk_faces", "WM 0-back faces", "0bk_faces_canonical"),
)

# The reporting order and labels used for the 23-condition summaries. The HCP
# Emotion EV is named ``fear``; its manuscript-facing condition is ``face``.
TASK_DISPLAY_NAMES = {
    "emotion": "Emotion",
    "wm": "WM",
    "social": "Social",
    "gambling": "Gambling",
    "language": "Language",
    "relational": "Relational",
    "motor": "Motor",
}


def _condition_label(task: str, condition: str) -> str:
    if task == "emotion":
        return "Emotion faces" if condition == "fear" else "Emotion shapes"
    display_condition = condition.replace("0bk_", "0-back ").replace("2bk_", "2-back ")
    return f"{TASK_DISPLAY_NAMES[task]} {display_condition.replace('_', ' ')}"


HCP_CANONICAL_CONDITIONS = tuple(
    (
        task,
        condition,
        "face" if task == "emotion" and condition == "fear" else condition,
        _condition_label(task, condition),
    )
    for task in (
        "emotion",
        "wm",
        "social",
        "gambling",
        "language",
        "relational",
        "motor",
    )
    for condition in TASKS[task].scientific_conditions
)

ROI_ROLE_ALIASES = {
    "amygdala": (
        "anatomical_amygdala",
        "amygdala",
        "amy_proper",
        "cit168_amygdala",
    ),
    "target": (
        "peri_amygdalar_target",
        "primary_target",
        "target",
        "hcp_bvr_striatal_peduncular_pseg50",
        "bvr_striate_peduncular_pamy50",
    ),
    "ffc": ("ffc", "glasser__ffc"),
    "v1": ("v1", "glasser__v1"),
}


def roi_spec(value: str) -> tuple[str, Path]:
    """Parse NAME=PATH."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("ROI specifications must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("ROI specifications must use NAME=PATH")
    return name.strip(), Path(path)


def candidate_roi_spec(value: str) -> tuple[str, str]:
    """Parse OUTPUT_NAME=CANDIDATE_KEY."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "candidate ROI specifications must use OUTPUT_NAME=CANDIDATE_KEY"
        )
    name, key = value.split("=", 1)
    if not name.strip() or not key.strip():
        raise argparse.ArgumentTypeError(
            "candidate ROI specifications must use OUTPUT_NAME=CANDIDATE_KEY"
        )
    return name.strip(), key.strip()


def condition_spec(value: str) -> tuple[tuple[str, str], str]:
    """Parse TASK:TRIAL_TYPE[:LABEL]."""
    parts = value.split(":", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            "condition specifications must use TASK:TRIAL_TYPE[:LABEL]"
        )
    label = parts[2] if len(parts) == 3 and parts[2] else parts[1].replace("_", " ")
    return (parts[0], parts[1]), label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    roi = commands.add_parser("roi", help="Extract ROI beta summaries.")
    roi_source = roi.add_mutually_exclusive_group(required=True)
    roi_source.add_argument("--manifest", type=Path)
    roi_source.add_argument(
        "--fixed-effects-root",
        type=Path,
        help="Root written by 02_fit_hcp_tasks.py.",
    )
    roi.add_argument("--subjects-file", type=Path)
    roi.add_argument("--subject", action="append", default=[])
    roi.add_argument("--output-dir", type=Path, required=True)
    roi.add_argument("--roi", type=roi_spec, action="append", required=True)
    roi.add_argument(
        "--candidate-table",
        type=Path,
        help="Candidate table written by 01_prepare_candidates.py.",
    )
    roi.add_argument(
        "--candidate-roi",
        type=candidate_roi_spec,
        action="append",
        default=[],
        metavar="NAME=KEY",
        help="Add a labeled atlas ROI from --candidate-table (for example ffc=glasser__ffc).",
    )
    roi.add_argument("--effect-column", default="effect_path")
    roi.add_argument("--support-column", default="support_path")
    roi.add_argument("--statistic", choices=("mean", "median"), default="median")
    roi.add_argument(
        "--amygdala-roi",
        help="ROI name for the mean-based Figure 8 and WM summaries.",
    )
    roi.add_argument(
        "--target-roi",
        help="ROI name for the mean-based Figure 8 and WM summaries.",
    )
    roi.add_argument("--ffc-roi", help="FFC ROI name for the WM load summary.")
    roi.add_argument("--v1-roi", help="V1 ROI name for the WM load summary.")

    pstc = commands.add_parser("pstc", help="Make participant-first event-locked PSC.")
    source = pstc.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--hcp-data-root", type=Path)
    pstc.add_argument("--subjects-file", type=Path)
    pstc.add_argument("--subject", action="append", default=[])
    pstc.add_argument("--output-dir", type=Path, required=True)
    pstc.add_argument("--roi", type=roi_spec, action="append", required=True)
    pstc.add_argument("--condition", type=condition_spec, action="append")
    pstc.add_argument("--tr", type=float, default=0.72)
    pstc.add_argument("--slice-time-ref", type=float, default=0.0)
    pstc.add_argument("--window-start", type=float, default=-4.0)
    pstc.add_argument("--window-end", type=float, default=32.0)

    spatial = commands.add_parser(
        "spatial",
        help="Run directional trimming, voxel-distance, and subnuclear summaries.",
    )
    spatial_source = spatial.add_mutually_exclusive_group(required=True)
    spatial_source.add_argument("--manifest", type=Path)
    spatial_source.add_argument(
        "--fixed-effects-root",
        type=Path,
        help="Root written by 02_fit_hcp_tasks.py.",
    )
    spatial.add_argument("--subjects-file", type=Path)
    spatial.add_argument("--subject", action="append", default=[])
    spatial.add_argument("--output-dir", type=Path, required=True)
    spatial.add_argument("--amygdala-mask", type=Path, required=True)
    spatial.add_argument("--bvr-mask", type=Path, required=True)
    spatial.add_argument("--cit168-pseg", type=Path, required=True)
    spatial.add_argument("--cit168-labels", type=Path, required=True)
    spatial.add_argument("--trim-step-mm", type=float, default=2.0)
    spatial.add_argument("--max-trim-mm", type=float)
    spatial.add_argument("--min-voxels", type=int, default=3)
    return parser


def load_manifest(path: Path, path_columns: tuple[str, ...]) -> pd.DataFrame:
    """Load a TSV manifest and resolve relative input paths beside it."""
    table = pd.read_csv(path, sep="\t", dtype={"subject_id": str})
    for column in path_columns:
        if column not in table.columns:
            continue
        table[column] = table[column].map(
            lambda value: str(
                (path.parent / Path(value)).resolve()
                if not Path(value).is_absolute()
                else Path(value)
            )
        )
    return table


def _metadata(row: object) -> dict[str, object]:
    names = (
        "subject_id",
        "task",
        "condition",
        "condition_key",
        "condition_label",
        "hemisphere",
    )
    return {name: getattr(row, name) for name in names if hasattr(row, name)}


def _group_summary(
    table: pd.DataFrame,
    value_column: str,
    grouping: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, frame in table.groupby(grouping, sort=False):
        summary = mean_ci(frame[value_column].to_numpy(dtype=np.float64))
        row = dict(zip(grouping, keys, strict=True))
        row.update(
            n_subjects=summary["n"],
            mean=summary["mean"],
            sem=summary["sem"],
            ci95_low=summary["ci95_low"],
            ci95_high=summary["ci95_high"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _requested_subjects(args: argparse.Namespace) -> list[str]:
    subjects = split_subject_values(args.subject)
    if args.subjects_file is not None:
        subjects.extend(read_subjects(args.subjects_file))
    return list(dict.fromkeys(item.removeprefix("sub-") for item in subjects))


def discover_fixed_effect_subjects(root: Path) -> list[str]:
    """Find participants with fixed-effects directories for all seven tasks."""
    task_subjects = []
    for task in TASKS:
        directory = root / task / "fixed_effects"
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing fixed-effects task directory: {directory}"
            )
        task_subjects.append(
            {
                path.name.removeprefix("sub-")
                for path in directory.glob("sub-*")
                if path.is_dir()
            }
        )
    common = set.intersection(*task_subjects)
    if not common:
        raise ValueError(f"No participants have all seven task outputs under {root}")
    return sorted(common)


def fixed_effect_manifest(root: Path, subjects: list[str]) -> pd.DataFrame:
    """Describe every canonical HCP effect for the requested participants."""
    rows = []
    missing_paths = []
    for subject in subjects:
        for task, effect_condition, condition, label in HCP_CANONICAL_CONDITIONS:
            fixed_dir = root / task / "fixed_effects" / f"sub-{subject}"
            effect_path = (
                fixed_dir / "contrasts" / f"{effect_condition}_canonical_effect.nii.gz"
            )
            support_path = fixed_dir / "support_mask.nii.gz"
            missing_paths.extend(
                path for path in (effect_path, support_path) if not path.exists()
            )
            rows.append(
                {
                    "subject_id": subject,
                    "task": task,
                    "condition": condition,
                    "condition_key": f"{task}__{condition}",
                    "condition_label": label,
                    "effect_path": effect_path,
                    "support_path": support_path,
                }
            )
    if missing_paths:
        preview = "\n".join(str(path) for path in missing_paths[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} fixed-effect inputs:\n{preview}"
        )
    return pd.DataFrame(rows)


def _resolve_roi_role(
    role: str,
    requested: str | None,
    available: set[str],
) -> str | None:
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Requested {role} ROI is not present: {requested}")
        return requested
    return next((name for name in ROI_ROLE_ALIASES[role] if name in available), None)


def _write_reported_mean_summaries(
    participant: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    available = set(participant["roi"])
    roles = {
        role: _resolve_roi_role(role, getattr(args, f"{role}_roi"), available)
        for role in ("amygdala", "target", "ffc", "v1")
    }
    if roles["amygdala"] is not None and roles["target"] is not None:
        profile, statistics_table = figure8_mean_summary(
            participant,
            amygdala_roi=roles["amygdala"],
            target_roi=roles["target"],
        )
        profile.to_csv(
            args.output_dir / "figure8_condition_profile.tsv", sep="\t", index=False
        )
        statistics_table.to_csv(
            args.output_dir / "figure8_statistics.tsv", sep="\t", index=False
        )

    if all(roles.values()):
        wm_participant, wm_summary, wm_concordance = wm_face_load_summary(
            participant,
            roi_roles={role: name for role, name in roles.items() if name is not None},
        )
        wm_participant.to_csv(
            args.output_dir / "wm_face_load_participant.tsv", sep="\t", index=False
        )
        wm_summary.to_csv(
            args.output_dir / "wm_face_load_summary.tsv", sep="\t", index=False
        )
        wm_concordance.to_csv(
            args.output_dir / "wm_face_load_concordance.tsv", sep="\t", index=False
        )


def run_roi(args: argparse.Namespace) -> int:
    if args.fixed_effects_root is not None:
        subjects = _requested_subjects(args)
        if not subjects:
            subjects = discover_fixed_effect_subjects(args.fixed_effects_root)
        manifest = fixed_effect_manifest(args.fixed_effects_root, subjects)
    else:
        manifest = load_manifest(
            args.manifest,
            (args.effect_column, args.support_column),
        )
    if args.effect_column not in manifest:
        raise ValueError(f"Manifest is missing {args.effect_column}")
    roi_images = load_roi_images(dict(args.roi))
    reference = next(iter(roi_images.values()))
    roi_masks = {
        name: image.get_fdata(dtype=np.float32) >= 0.5
        for name, image in roi_images.items()
    }
    if args.candidate_roi:
        if args.candidate_table is None:
            raise ValueError("--candidate-roi requires --candidate-table.")
        candidates = {
            candidate.key: candidate
            for candidate in load_candidates(args.candidate_table)
        }
        unknown = {key for _, key in args.candidate_roi}.difference(candidates)
        if unknown:
            raise ValueError(f"Candidate table is missing keys: {sorted(unknown)}")
        requested = [candidates[key] for _, key in args.candidate_roi]
        atlas_masks = candidate_masks(requested, reference)
        for name, key in args.candidate_roi:
            if name in roi_masks:
                raise ValueError(f"ROI name is repeated: {name}")
            roi_masks[name] = atlas_masks[key]
    rows = []
    for row in manifest.itertuples(index=False):
        effect = load_image(Path(getattr(row, args.effect_column)), ndim=3)
        if not same_grid(effect, reference):
            path = getattr(row, args.effect_column)
            raise ValueError(f"Effect/ROI grid mismatch: {path}")
        support = None
        if args.support_column in manifest:
            _, support = load_mask(
                Path(getattr(row, args.support_column)),
                reference=reference,
            )
        extracted = extract_roi_values(
            effect,
            roi_masks,
            support=support,
            statistic=args.statistic,
        )
        for record in extracted.to_dict("records"):
            rows.append({**_metadata(row), **record})

    participant = pd.DataFrame(rows)
    grouping = [
        column
        for column in ("task", "condition_key", "condition_label", "hemisphere", "roi")
        if column in participant
    ]
    value_column = f"{args.statistic}_beta"
    group = _group_summary(participant, value_column, grouping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "medians" if args.statistic == "median" else "means"
    participant.to_csv(
        args.output_dir / f"participant_roi_{suffix}.tsv",
        sep="\t",
        index=False,
    )
    group.to_csv(args.output_dir / f"group_roi_{suffix}.tsv", sep="\t", index=False)
    explicit_reported_roles = any(
        getattr(args, f"{role}_roi") is not None
        for role in ("amygdala", "target", "ffc", "v1")
    )
    if args.statistic == "mean" and (
        args.fixed_effects_root is not None or explicit_reported_roles
    ):
        _write_reported_mean_summaries(participant, args)
    return 0


def run_pstc(args: argparse.Namespace) -> int:
    conditions = dict(args.condition) if args.condition else None
    if args.hcp_data_root is not None:
        if conditions is None:
            conditions = {
                ("emotion", "fear"): "Emotion fear",
                ("wm", "0bk_faces"): "WM faces",
                ("social", "mental"): "Social mental",
                ("language", "story"): "Language story",
            }
        subjects = _requested_subjects(args)
        if not subjects:
            raise ValueError("HCP PSTCs require --subjects-file or --subject.")
        rows = []
        for subject in subjects:
            for task_name in dict.fromkeys(task for task, _ in conditions):
                for run in load_runs(args.hcp_data_root, subject, TASKS[task_name]):
                    rows.append(
                        {
                            "subject_id": subject,
                            "task": task_name,
                            "run": run.label,
                            "bold_path": run.bold_path,
                            "events": run.events,
                        }
                    )
        manifest = pd.DataFrame(rows)
    else:
        manifest = load_manifest(args.manifest, ("bold_path", "events_path"))
    rois = load_roi_images(dict(args.roi))
    subject, group, task_mean = collect_participant_pstcs(
        manifest,
        rois,
        conditions=conditions,
        tr_seconds=args.tr,
        slice_time_ref=args.slice_time_ref,
        window=(args.window_start, args.window_end),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject.to_csv(args.output_dir / "pstc_participant.tsv", sep="\t", index=False)
    group.to_csv(args.output_dir / "pstc_group.tsv", sep="\t", index=False)
    task_mean.to_csv(args.output_dir / "pstc_task_mean.tsv", sep="\t", index=False)
    return 0


def split_subject_values(values: list[str]) -> list[str]:
    """Split repeated or comma-separated subject arguments."""
    return [
        item.strip() for value in values for item in value.split(",") if item.strip()
    ]


def run_spatial(args: argparse.Namespace) -> int:
    if args.fixed_effects_root is not None:
        subjects = _requested_subjects(args)
        if not subjects:
            raise ValueError(
                "--fixed-effects-root requires --subjects-file or --subject."
            )
        rows = []
        missing_paths = []
        for subject in subjects:
            for task, key, label, stem in SPATIAL_CONDITIONS:
                fixed_dir = (
                    args.fixed_effects_root / task / "fixed_effects" / f"sub-{subject}"
                )
                effect_path = fixed_dir / "contrasts" / f"{stem}_effect.nii.gz"
                support_path = fixed_dir / "support_mask.nii.gz"
                missing_paths.extend(
                    path for path in (effect_path, support_path) if not path.exists()
                )
                rows.append(
                    {
                        "subject_id": subject,
                        "task": task,
                        "condition_key": key,
                        "condition_label": label,
                        "effect_path": effect_path,
                        "support_path": support_path,
                    }
                )
        if missing_paths:
            preview = "\n".join(str(path) for path in missing_paths[:10])
            raise FileNotFoundError(
                f"Missing {len(missing_paths)} fixed-effect inputs:\n{preview}"
            )
        manifest = pd.DataFrame(rows)
    else:
        manifest = load_manifest(args.manifest, ("effect_path", "support_path"))
    required = {"subject_id", "condition_key", "effect_path", "support_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Spatial manifest is missing columns: {sorted(missing)}")
    if args.min_voxels < 3:
        raise ValueError("--min-voxels must be at least three.")

    reference, amygdala = load_mask(args.amygdala_mask)
    _, bvr = load_mask(args.bvr_mask, reference=reference)
    trims = directional_trim_masks(
        amygdala,
        bvr,
        reference,
        step_mm=args.trim_step_mm,
        max_depth_mm=args.max_trim_mm,
    )
    geometry = distance_geometry(reference, amygdala, bvr)

    trim_rows: list[dict[str, object]] = []
    slope_rows: list[dict[str, object]] = []
    vectors: dict[str, list[np.ndarray]] = {}
    condition_labels: dict[str, str] = {}
    for row in manifest.itertuples(index=False):
        effect = load_image(Path(row.effect_path), ndim=3)
        _, support = load_mask(Path(row.support_path), reference=reference)
        if not same_grid(effect, reference):
            raise ValueError(f"Effect/mask grid mismatch: {row.effect_path}")
        data = effect.get_fdata(dtype=np.float32)
        metadata = _metadata(row)
        condition = str(row.condition_key)
        condition_labels[condition] = str(
            getattr(row, "condition_label", condition.replace("_", " "))
        )
        for trim in trims:
            beta, count = roi_value(
                data,
                trim.mask,
                support=support,
                statistic="mean",
            )
            if count >= args.min_voxels:
                trim_rows.append(
                    {
                        **metadata,
                        "profile": "amy_bvr_directional_trim",
                        "direction": trim.direction,
                        "depth_mm": trim.depth_mm,
                        "beta": beta,
                        "n_supported_voxels": count,
                    }
                )

        index = tuple(geometry.indices.T)
        vector = np.asarray(data[index], dtype=np.float64)
        vector[~support[index]] = np.nan
        vectors.setdefault(condition, []).append(vector)
        if np.count_nonzero(np.isfinite(vector)) >= args.min_voxels:
            fit = fit_distance_gradient(geometry.distance_mm, vector)
            slope_rows.append({**metadata, **fit})

    trim_values = pd.DataFrame(trim_rows)
    trim_summary = summarize_trims(trim_values)
    participant_slopes = pd.DataFrame(slope_rows)
    slope_grouping = [
        column
        for column in ("task", "condition_key", "condition_label")
        if column in participant_slopes
    ]
    slope_summary = _group_summary(
        participant_slopes,
        "slope_beta_per_mm",
        slope_grouping,
    )

    voxel_rows = []
    gradient_rows = []
    condition_medians = []
    for condition, arrays in vectors.items():
        matrix = np.stack(arrays)
        support_count = np.count_nonzero(np.isfinite(matrix), axis=0)
        group_median = np.full(matrix.shape[1], np.nan)
        valid = support_count > 0
        group_median[valid] = np.nanmedian(matrix[:, valid], axis=0)
        condition_medians.append(group_median)
        fit = fit_distance_gradient(geometry.distance_mm, group_median)
        gradient_rows.append(
            {
                "condition_key": condition,
                "condition_label": condition_labels[condition],
                **fit,
            }
        )
        for index, xyz, distance, count, beta in zip(
            geometry.indices,
            geometry.coordinates_mm,
            geometry.distance_mm,
            support_count,
            group_median,
            strict=True,
        ):
            voxel_rows.append(
                {
                    "condition_key": condition,
                    "condition_label": condition_labels[condition],
                    "voxel_i": int(index[0]),
                    "voxel_j": int(index[1]),
                    "voxel_k": int(index[2]),
                    "mni_x_mm": float(xyz[0]),
                    "mni_y_mm": float(xyz[1]),
                    "mni_z_mm": float(xyz[2]),
                    "distance_to_bvr_mm": float(distance),
                    "n_subjects_support": int(count),
                    "group_median_effect": float(beta),
                }
            )

    task_mean_beta = np.nanmean(np.stack(condition_medians), axis=0)
    subnucleus_masks = load_subnucleus_masks(
        args.cit168_pseg,
        args.cit168_labels,
        reference,
    )
    subnuclei = summarize_subnuclei(geometry, task_mean_beta, subnucleus_masks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "directional_participant.tsv": trim_values,
        "directional_summary.tsv": trim_summary,
        "distance_participant_slopes.tsv": participant_slopes,
        "distance_slope_summary.tsv": slope_summary,
        "distance_voxels.tsv": pd.DataFrame(voxel_rows),
        "distance_group_regressions.tsv": pd.DataFrame(gradient_rows),
        "subnucleus_summary.tsv": subnuclei,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, sep="\t", index=False)
    return 0


def summarize_trims(values: pd.DataFrame) -> pd.DataFrame:
    """Summarize each condition and the participant-first task mean."""
    grouping = [
        column
        for column in (
            "condition_key",
            "condition_label",
            "profile",
            "direction",
            "depth_mm",
        )
        if column in values
    ]
    condition = _group_summary(values, "beta", grouping).rename(
        columns={
            "n_subjects": "n_observations",
            "mean": "mean_beta",
            "sem": "sem_beta",
            "ci95_low": "ci95_low_beta",
            "ci95_high": "ci95_high_beta",
        }
    )
    task_mean = (
        values.groupby(
            ["subject_id", "profile", "direction", "depth_mm"],
            as_index=False,
        )["beta"]
        .mean()
        .assign(condition_key="task_mean", condition_label="Task mean")
    )
    task_grouping = [
        "condition_key",
        "condition_label",
        "profile",
        "direction",
        "depth_mm",
    ]
    task_summary = _group_summary(task_mean, "beta", task_grouping).rename(
        columns={
            "n_subjects": "n_observations",
            "mean": "mean_beta",
            "sem": "sem_beta",
            "ci95_low": "ci95_low_beta",
            "ci95_high": "ci95_high_beta",
        }
    )
    return pd.concat([condition, task_summary], ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"roi": run_roi, "pstc": run_pstc, "spatial": run_spatial}[args.command](
        args
    )


if __name__ == "__main__":
    raise SystemExit(main())
