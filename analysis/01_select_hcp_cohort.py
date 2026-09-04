#!/usr/bin/env python3
"""Select the HCP-YA task cohort from behavioral and imaging inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amyvasc_release.hcp import TASK_ORDER, TASKS, load_runs  # noqa: E402

DEFAULT_ACCURACY_COLUMNS = (
    "Emotion_Task_Acc",
    "Language_Task_Acc",
    "Relational_Task_Acc",
    "WM_Task_Acc",
)
REQUIRED_SCREENING_COLUMNS = ("3T_Full_Task_fMRI", "QC_Issue")


def read_table(path: Path) -> pd.DataFrame:
    """Read a comma- or tab-delimited table without coercing subject IDs."""
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, dtype=str)


def read_subjects(path: Path | None) -> set[str]:
    """Read one subject identifier per row, with or without a header."""
    if path is None:
        return set()
    if path.suffix.lower() == ".txt":
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    frame = read_table(path)
    column = "subject_id" if "subject_id" in frame.columns else frame.columns[0]
    return set(frame[column].dropna().astype(str).str.strip())


def eligible_subjects(
    frame: pd.DataFrame,
    *,
    subject_column: str = "Subject",
    accuracy_columns: tuple[str, ...] | list[str] = DEFAULT_ACCURACY_COLUMNS,
    minimum_accuracy: float = 80.0,
) -> pd.Series:
    """Apply HCP percent-accuracy, complete-task, and quality-control criteria."""
    required = [subject_column, *accuracy_columns, *REQUIRED_SCREENING_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Behavioral table is missing columns: {', '.join(missing)}")
    accuracy = frame[list(accuracy_columns)].apply(pd.to_numeric, errors="coerce")
    selected = accuracy.notna().all(axis=1) & accuracy.ge(minimum_accuracy).all(axis=1)
    selected &= (
        frame["3T_Full_Task_fMRI"].astype(str).str.strip().str.lower().eq("true")
    )
    selected &= frame["QC_Issue"].isna()
    return frame.loc[selected, subject_column].astype(str).str.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-column", default="Subject")
    parser.add_argument(
        "--accuracy-columns",
        default=",".join(DEFAULT_ACCURACY_COLUMNS),
        help="Comma-separated accuracy columns required to meet the threshold.",
    )
    parser.add_argument(
        "--minimum-accuracy",
        type=float,
        default=80.0,
        help="Minimum accuracy in HCP percentage units (default: 80, meaning 80%%).",
    )
    parser.add_argument(
        "--exclude-subjects",
        type=Path,
        help="Optional additional exclusions after the HCP quality-control filter.",
    )
    parser.add_argument(
        "--required-path",
        action="append",
        default=[],
        help=(
            "Required imaging path template; repeat as needed. Use {subject} "
            "where the subject identifier belongs."
        ),
    )
    parser.add_argument(
        "--hcp-data-root",
        type=Path,
        help="Require valid Task3T inputs for all seven HCP-YA tasks.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    frame = read_table(args.behavior)
    columns = [value.strip() for value in args.accuracy_columns.split(",") if value]
    subjects = eligible_subjects(
        frame,
        subject_column=args.subject_column,
        accuracy_columns=columns,
        minimum_accuracy=args.minimum_accuracy,
    )

    excluded = read_subjects(args.exclude_subjects)
    subjects = subjects.loc[~subjects.isin(excluded)]

    for template in args.required_path:
        required_paths = [
            Path(template.format(subject=subject)).expanduser() for subject in subjects
        ]
        present = pd.Series(
            [path.exists() for path in required_paths],
            index=subjects.index,
        )
        subjects = subjects.loc[present]

    if args.hcp_data_root is not None:
        complete = []
        for subject in subjects:
            try:
                for task_name in TASK_ORDER:
                    load_runs(args.hcp_data_root, subject, TASKS[task_name])
            except (FileNotFoundError, ValueError):
                complete.append(False)
            else:
                complete.append(True)
        subjects = subjects.loc[complete]

    output = pd.DataFrame({"subject_id": sorted(subjects.unique())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    print(f"Selected {len(output)} participants; wrote {args.output}")


if __name__ == "__main__":
    main()
