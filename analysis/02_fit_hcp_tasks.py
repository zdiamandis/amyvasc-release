#!/usr/bin/env python3
"""Fit the seven HCP-YA task models and participant fixed effects."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amyvasc_release.glm import group_mean_maps, write_tsv
from amyvasc_release.hcp import (
    TASK_ORDER,
    TASKS,
    fit_subject_task,
    read_subjects,
    write_task_config,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Fit unsmoothed HCP-YA task GLMs, combine LR/RL runs within "
            "participant, and optionally create descriptive group-mean maps."
        )
    )
    result.add_argument("--data-root", type=Path, default=Path("data/HCP_1200"))
    result.add_argument(
        "--output-root", type=Path, default=Path("outputs/analysis/hcp_tasks")
    )
    result.add_argument(
        "--subjects-file",
        type=Path,
        help="Text, CSV, or TSV file containing the authorized HCP subject IDs.",
    )
    result.add_argument(
        "--subject",
        action="append",
        default=[],
        help="Subject ID; repeat the option or separate IDs with commas.",
    )
    result.add_argument(
        "--task",
        action="append",
        choices=TASK_ORDER,
        default=[],
        help="Task to fit; repeat as needed. The default is all seven tasks.",
    )
    result.add_argument("--n-jobs", type=int, default=1)
    result.add_argument(
        "--group-maps",
        action="store_true",
        help="Also write coverage-aware participant-mean effect maps.",
    )
    result.add_argument("--group-coverage", type=float, default=0.50)
    return result


def subject_ids(args: argparse.Namespace) -> list[str]:
    values = [part.strip() for value in args.subject for part in value.split(",")]
    values = [value.removeprefix("sub-") for value in values if value]
    if args.subjects_file is not None:
        values.extend(
            value.removeprefix("sub-") for value in read_subjects(args.subjects_file)
        )
    resolved = list(dict.fromkeys(values))
    if not resolved:
        raise ValueError("Provide --subjects-file or at least one --subject.")
    return resolved


def fit_or_error(
    data_root: Path,
    output_root: Path,
    subject_id: str,
    task_name: str,
) -> dict[str, object]:
    try:
        return fit_subject_task(
            data_root=data_root,
            output_root=output_root,
            subject_id=subject_id,
            task_name=task_name,
        )
    except Exception as error:
        return {
            "status": "error",
            "subject_id": subject_id,
            "task": task_name,
            "n_runs": 0,
            "error": str(error),
        }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least one.")
    subjects = subject_ids(args)
    tasks = [name for name in TASK_ORDER if not args.task or name in args.task]
    any_errors = False

    for task_name in tasks:
        task = TASKS[task_name]
        write_task_config(args.output_root, task, subjects)
        if args.n_jobs == 1:
            rows = [
                fit_or_error(args.data_root, args.output_root, subject, task_name)
                for subject in subjects
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=min(args.n_jobs, len(subjects))
            ) as pool:
                rows = list(
                    pool.map(
                        fit_or_error,
                        [args.data_root] * len(subjects),
                        [args.output_root] * len(subjects),
                        subjects,
                        [task_name] * len(subjects),
                    )
                )
        write_tsv(
            args.output_root / task_name / "fixed_effects" / "completed_subjects.tsv",
            rows,
        )
        successful = [row["subject_id"] for row in rows if row["status"] == "ok"]
        task_complete = len(successful) == len(subjects)
        if args.group_maps and task_complete:
            group_mean_maps(
                fixed_effect_dirs=[
                    args.output_root / task_name / "fixed_effects" / f"sub-{subject}"
                    for subject in successful
                ],
                output_dir=args.output_root / task_name / "group_display",
                contrast_stems=[contrast.stem for contrast in task.contrasts],
                minimum_coverage=args.group_coverage,
            )
        elif args.group_maps and successful:
            print(
                f"Skipped {task_name} group maps because "
                f"{len(subjects) - len(successful)} requested fits failed.",
                flush=True,
            )
        any_errors |= any(row["status"] != "ok" for row in rows)
    return int(any_errors)


if __name__ == "__main__":
    raise SystemExit(main())
