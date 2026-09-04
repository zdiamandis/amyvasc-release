#!/usr/bin/env python3
"""Refit Emotion GLMs after smoothing the BOLD time series."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.glm import (  # noqa: E402
    GLMSettings,
    RunResult,
    combine_runs,
    common_epi_mask,
    fit_run,
)
from amyvasc_release.hcp import HCP_TR, TASKS, load_runs, read_subjects  # noqa: E402
from amyvasc_release.internal import (  # noqa: E402
    EMOTION_CONTRASTS,
    INTERNAL_TR,
    available_subjects,
    discover_emotion_runs,
    emotion_block_events,
)
from amyvasc_release.masks import load_mask, same_grid  # noqa: E402
from amyvasc_release.task_summaries import mean_ci  # noqa: E402

CONTRAST_STEM = "fear_minus_shape"


def parse_levels(value: str) -> tuple[float, ...]:
    """Parse ordered, unique, nonnegative FWHM values."""
    levels = tuple(
        dict.fromkeys(float(item) for item in value.split(",") if item.strip())
    )
    if not levels or any(not np.isfinite(item) or item < 0 for item in levels):
        raise argparse.ArgumentTypeError("levels must be finite nonnegative values")
    return levels


def level_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format(value, "g").replace(".", "p")


def split_subjects(values: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip().removeprefix("sub-")
            for value in values
            for item in value.split(",")
            if item.strip()
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit each smoothing level through Nilearn's FirstLevelModel. "
            "Smoothing is applied to 4D BOLD before GLM fitting; statistic maps "
            "are never smoothed."
        )
    )
    commands = parser.add_subparsers(dest="dataset", required=True)

    hcp = commands.add_parser("hcp", help="HCP-YA Emotion smoothing analysis.")
    hcp.add_argument("--data-root", type=Path, required=True)
    hcp.add_argument("--subjects-file", type=Path)
    hcp.add_argument("--subject", action="append", default=[])
    hcp.add_argument("--amygdala-mask", type=Path, required=True)
    hcp.add_argument("--levels", type=parse_levels, default=(0.0, 2.0, 4.0, 6.0))
    hcp.add_argument("--z-threshold", type=float, default=3.0)
    hcp.add_argument("--output-dir", type=Path, required=True)

    internal = commands.add_parser(
        "internal",
        help="Dense Amygdala Emotion smoothing maps.",
    )
    internal.add_argument("--data-root", type=Path, required=True)
    internal.add_argument("--subject", action="append", default=[])
    internal.add_argument("--levels", type=parse_levels, default=(0.0, 2.0, 4.0, 6.0))
    internal.add_argument("--output-dir", type=Path, required=True)
    return parser


def _fit_fixed_effects(
    *,
    bold_paths: list[Path],
    events: list[pd.DataFrame],
    run_labels: list[str],
    contrasts: tuple,
    tr: float,
    fwhm: float,
    output_dir: Path,
    mask: nib.spatialimages.SpatialImage,
) -> Path:
    """Fit all runs at one FWHM and retain only fixed-effects maps."""
    if not (len(bold_paths) == len(events) == len(run_labels)):
        raise ValueError("BOLD paths, events, and run labels must align.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run_level_", dir=output_dir) as temporary:
        run_results: list[RunResult] = []
        settings = GLMSettings(
            tr=tr,
            smoothing_fwhm=None if np.isclose(fwhm, 0.0) else fwhm,
        )
        for index, (bold_path, run_events, label) in enumerate(
            zip(bold_paths, events, run_labels, strict=True),
            start=1,
        ):
            run_results.append(
                fit_run(
                    bold_path=bold_path,
                    mask_img=mask,
                    output_dir=Path(temporary) / f"run-{index}",
                    contrasts=contrasts,
                    settings=settings,
                    events=run_events,
                    run_label=label,
                )
            )
        combine_runs(
            run_results=run_results,
            mask_img=mask,
            output_dir=output_dir,
            contrast_stems=[contrast.stem for contrast in contrasts],
        )
    return output_dir


def resolve_hcp_subjects(args: argparse.Namespace) -> list[str]:
    explicit = split_subjects(args.subject)
    if explicit:
        return explicit
    if args.subjects_file is None:
        raise ValueError("Provide --subjects-file or at least one --subject.")
    return [
        subject.removeprefix("sub-") for subject in read_subjects(args.subjects_file)
    ]


def run_hcp(args: argparse.Namespace) -> int:
    if not np.isfinite(args.z_threshold):
        raise ValueError("--z-threshold must be finite.")
    task = TASKS["emotion"]
    contrast = tuple(item for item in task.contrasts if item.stem == CONTRAST_STEM)
    subjects = resolve_hcp_subjects(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in subjects:
        runs = load_runs(args.data_root, subject, task)
        bold_paths = [run.bold_path for run in runs]
        events = [run.events for run in runs]
        labels = [run.label for run in runs]
        mask = common_epi_mask(bold_paths)
        for fwhm in args.levels:
            fixed_dir = (
                args.output_dir
                / f"fwhm-{level_tag(fwhm)}mm"
                / "fixed_effects"
                / f"sub-{subject}"
            )
            _fit_fixed_effects(
                bold_paths=bold_paths,
                events=events,
                run_labels=labels,
                contrasts=contrast,
                tr=HCP_TR,
                fwhm=fwhm,
                output_dir=fixed_dir,
                mask=mask,
            )
            effect_img = nib.load(
                fixed_dir / "contrasts" / f"{CONTRAST_STEM}_effect.nii.gz"
            )
            z_img = nib.load(fixed_dir / "contrasts" / f"{CONTRAST_STEM}_z.nii.gz")
            support_img, support = load_mask(fixed_dir / "support_mask.nii.gz")
            _, amygdala = load_mask(args.amygdala_mask, reference=support_img)
            if not same_grid(effect_img, support_img) or not same_grid(
                z_img, support_img
            ):
                raise ValueError(f"Fixed-effects grid mismatch for sub-{subject}")
            effect = effect_img.get_fdata(dtype=np.float32)
            z = z_img.get_fdata(dtype=np.float32)
            valid = amygdala & support & np.isfinite(effect) & np.isfinite(z)
            if not np.any(valid):
                raise ValueError(f"No supported amygdala voxels for sub-{subject}")
            rows.append(
                {
                    "subject_id": subject,
                    "smoothing_fwhm_mm": fwhm,
                    "n_mask_voxels": int(amygdala.sum()),
                    "n_supported_voxels": int(valid.sum()),
                    "percent_z_ge_threshold": float(
                        100 * np.mean(z[valid] >= args.z_threshold)
                    ),
                    "mean_effect": float(np.mean(effect[valid])),
                }
            )
            print(f"sub-{subject} fwhm-{level_tag(fwhm)}mm", flush=True)

    subject_table = pd.DataFrame(rows)
    group_rows = []
    for fwhm, frame in subject_table.groupby("smoothing_fwhm_mm", sort=True):
        extent = mean_ci(frame["percent_z_ge_threshold"])
        effect = mean_ci(frame["mean_effect"])
        group_rows.append(
            {
                "smoothing_fwhm_mm": fwhm,
                "n_subjects": extent["n"],
                "mean_percent_z_ge_threshold": extent["mean"],
                "sem_percent_z_ge_threshold": extent["sem"],
                "ci95_low_percent_z_ge_threshold": extent["ci95_low"],
                "ci95_high_percent_z_ge_threshold": extent["ci95_high"],
                "mean_effect": effect["mean"],
                "sem_effect": effect["sem"],
            }
        )
    subject_table.to_csv(
        args.output_dir / "participant_extent.tsv", sep="\t", index=False
    )
    pd.DataFrame(group_rows).to_csv(
        args.output_dir / "group_extent.tsv",
        sep="\t",
        index=False,
    )
    return 0


def run_internal(args: argparse.Namespace) -> int:
    subjects = split_subjects(args.subject) or [
        subject.removeprefix("sub-") for subject in available_subjects(args.data_root)
    ]
    contrast = tuple(item for item in EMOTION_CONTRASTS if item.stem == CONTRAST_STEM)
    for subject in subjects:
        runs = discover_emotion_runs(args.data_root, subject)
        bold_paths = [run.bold_path for run in runs]
        events = [emotion_block_events(run.events_path) for run in runs]
        labels = [run.label for run in runs]
        mask = common_epi_mask(bold_paths)
        for fwhm in args.levels:
            fixed_dir = (
                args.output_dir
                / f"fwhm-{level_tag(fwhm)}mm"
                / "fixed_effects"
                / f"sub-{subject.removeprefix('sub-')}"
            )
            _fit_fixed_effects(
                bold_paths=bold_paths,
                events=events,
                run_labels=labels,
                contrasts=contrast,
                tr=INTERNAL_TR,
                fwhm=fwhm,
                output_dir=fixed_dir,
                mask=mask,
            )
            label = subject.removeprefix("sub-")
            print(f"sub-{label} fwhm-{level_tag(fwhm)}mm", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_hcp(args) if args.dataset == "hcp" else run_internal(args)


if __name__ == "__main__":
    raise SystemExit(main())
