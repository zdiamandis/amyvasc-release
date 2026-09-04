#!/usr/bin/env python3
"""Fit movie face presence and combine 29 runs within each participant."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amyvasc_release.glm import GLMSettings, write_json, write_tsv
from amyvasc_release.internal import (
    INTERNAL_TR,
    MOVIES,
    available_subjects,
    fit_movie_subject,
    normalize_subject,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Fit a binary regressor derived from framewise face annotations to "
            "unsmoothed ICA-FIX-cleaned movie runs, then combine runs within "
            "participant by precision-weighted fixed effects."
        )
    )
    result.add_argument("--data-root", type=Path, default=Path("data/dense_amygdala"))
    result.add_argument(
        "--output-root", type=Path, default=Path("outputs/analysis/movie_faces")
    )
    result.add_argument(
        "--subject",
        action="append",
        default=[],
        help="BIDS subject label; repeat or separate labels with commas.",
    )
    result.add_argument("--n-jobs", type=int, default=1)
    return result


def subjects(args: argparse.Namespace) -> list[str]:
    requested = [
        normalize_subject(part.strip())
        for value in args.subject
        for part in value.split(",")
        if part.strip()
    ]
    return list(dict.fromkeys(requested or available_subjects(args.data_root)))


def fit_or_error(
    data_root: Path,
    output_root: Path,
    subject: str,
) -> dict[str, object]:
    try:
        return fit_movie_subject(
            data_root=data_root,
            output_root=output_root,
            subject=subject,
        )
    except Exception as error:
        return {
            "status": "error",
            "subject_id": subject.removeprefix("sub-"),
            "n_runs": 0,
            "error": str(error),
        }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive.")
    selected = subjects(args)
    if not selected:
        raise FileNotFoundError(f"No participants found below {args.data_root}.")
    settings = GLMSettings(tr=INTERNAL_TR)
    write_json(
        args.output_root / "analysis_config.json",
        {
            "dataset": "Dense Amygdala ICA-FIX-cleaned movie runs",
            "subjects": selected,
            "movies": list(MOVIES),
            "expected_runs_per_participant": 29,
            "regressor": "binary face presence from framewise face_area_total",
            "tr_seconds": settings.tr,
            "hrf_model": settings.hrf_model,
            "drift_model": settings.drift_model,
            "high_pass_hz": settings.high_pass,
            "noise_model": settings.noise_model,
            "signal_scaling": settings.signal_scaling,
            "smoothing_fwhm": settings.smoothing_fwhm,
            "additional_confounds": None,
            "fixed_effects": "inverse-variance weighted across movie runs",
        },
    )
    arguments = (
        [args.data_root] * len(selected),
        [args.output_root] * len(selected),
        selected,
    )
    if args.n_jobs == 1:
        rows = [fit_or_error(*items) for items in zip(*arguments, strict=True)]
    else:
        with ProcessPoolExecutor(max_workers=min(args.n_jobs, len(selected))) as pool:
            rows = list(pool.map(fit_or_error, *arguments))
    write_tsv(args.output_root / "completed_subjects.tsv", rows)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
