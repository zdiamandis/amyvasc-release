"""HCP-YA task definitions, input loading, and participant-level fitting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .glm import (
    Contrast,
    GLMSettings,
    RunResult,
    combine_runs,
    common_epi_mask,
    fit_run,
    save_image,
    write_json,
)

HCP_TR = 0.72
BOLD_SUFFIX = "_hp0_clean_rclean_tclean.nii.gz"


@dataclass(frozen=True)
class Task:
    """Run names, event labels, and output contrasts for one HCP task."""

    name: str
    runs: tuple[str, str]
    events: tuple[tuple[str, str], ...]
    conditions: tuple[str, ...]
    n_scans: int
    nuisance: tuple[str, ...] = ()
    include_derivatives: bool = True
    include_pairwise: bool = True
    single_event_per_condition: bool = False

    @property
    def scientific_conditions(self) -> tuple[str, ...]:
        return tuple(name for name in self.conditions if name not in self.nuisance)

    @property
    def contrasts(self) -> tuple[Contrast, ...]:
        main_effects: list[Contrast] = []
        for condition in self.scientific_conditions:
            main_effects.append(Contrast(f"{condition}_canonical", {condition: 1.0}))
            if self.include_derivatives:
                main_effects.append(
                    Contrast(
                        f"{condition}_derivative",
                        {f"{condition}_derivative": 1.0},
                    )
                )
        pairwise = (
            [
                Contrast(
                    f"{positive}_minus_{negative}",
                    {positive: 1.0, negative: -1.0},
                )
                for positive, negative in combinations(self.scientific_conditions, 2)
            ]
            if self.include_pairwise
            else []
        )
        return tuple([*main_effects, *pairwise])


TASKS: dict[str, Task] = {
    "emotion": Task(
        "emotion",
        ("tfMRI_EMOTION_LR", "tfMRI_EMOTION_RL"),
        (("fear", "fear"), ("neut", "shape")),
        ("fear", "shape"),
        176,
    ),
    "gambling": Task(
        "gambling",
        ("tfMRI_GAMBLING_LR", "tfMRI_GAMBLING_RL"),
        (("win", "win"), ("loss", "loss")),
        ("win", "loss"),
        253,
    ),
    "language": Task(
        "language",
        ("tfMRI_LANGUAGE_LR", "tfMRI_LANGUAGE_RL"),
        (("story", "story"), ("math", "math")),
        ("story", "math"),
        316,
    ),
    "motor": Task(
        "motor",
        ("tfMRI_MOTOR_LR", "tfMRI_MOTOR_RL"),
        (
            ("lh", "left_hand"),
            ("rh", "right_hand"),
            ("lf", "left_foot"),
            ("rf", "right_foot"),
            ("t", "tongue"),
            ("cue", "cue"),
        ),
        ("left_hand", "right_hand", "left_foot", "right_foot", "tongue", "cue"),
        284,
        nuisance=("cue",),
        include_derivatives=False,
        include_pairwise=False,
    ),
    "relational": Task(
        "relational",
        ("tfMRI_RELATIONAL_LR", "tfMRI_RELATIONAL_RL"),
        (("relation", "relational"), ("match", "matching")),
        ("relational", "matching"),
        232,
    ),
    "social": Task(
        "social",
        ("tfMRI_SOCIAL_LR", "tfMRI_SOCIAL_RL"),
        (("mental", "mental"), ("rnd", "random")),
        ("mental", "random"),
        274,
    ),
    "wm": Task(
        "wm",
        ("tfMRI_WM_LR", "tfMRI_WM_RL"),
        tuple(
            (name, name)
            for name in (
                "0bk_faces",
                "0bk_places",
                "0bk_tools",
                "0bk_body",
                "2bk_faces",
                "2bk_places",
                "2bk_tools",
                "2bk_body",
            )
        ),
        (
            "0bk_faces",
            "0bk_places",
            "0bk_tools",
            "0bk_body",
            "2bk_faces",
            "2bk_places",
            "2bk_tools",
            "2bk_body",
        ),
        405,
        single_event_per_condition=True,
    ),
}
TASK_ORDER = tuple(TASKS)


@dataclass(frozen=True)
class HCPRun:
    """Validated files and events for one HCP-YA task run."""

    name: str
    label: str
    bold_path: Path
    events: pd.DataFrame
    n_scans: int


def read_subjects(path: Path) -> list[str]:
    """Read HCP subject IDs from a text, CSV, or TSV file."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        table = pd.read_csv(path, sep=separator)
        for column in ("Subject", "subject_id", "subject"):
            if column in table:
                return list(dict.fromkeys(table[column].astype(str)))
        raise ValueError(f"No subject-ID column found in {path}.")
    return list(
        dict.fromkeys(
            line.strip() for line in path.read_text().splitlines() if line.strip()
        )
    )


def _load_ev(path: Path, condition: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing HCP event file: {path}")
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite three-column HCP event file: {path}")
    events = pd.DataFrame(values, columns=("onset", "duration", "modulation"))
    if (events["duration"] <= 0).any():
        raise ValueError(f"Event durations must be positive: {path}")
    events["trial_type"] = condition
    return events[["onset", "duration", "trial_type", "modulation"]]


def load_runs(data_root: Path, subject_id: str, task: Task) -> tuple[HCPRun, ...]:
    """Load both phase-encoded runs and their official EV files."""
    results_root = data_root / subject_id / "MNINonLinear" / "Results"
    runs: list[HCPRun] = []
    reference_grid: tuple[tuple[int, ...], np.ndarray] | None = None
    for run_name in task.runs:
        run_dir = results_root / run_name
        bold_path = run_dir / f"{run_name}{BOLD_SUFFIX}"
        if not bold_path.exists():
            raise FileNotFoundError(f"Missing HCP task image: {bold_path}")
        image = nib.load(bold_path)
        if image.ndim != 4 or image.shape[3] != task.n_scans:
            raise ValueError(
                f"Expected {task.n_scans} volumes for {run_name}, found {image.shape}."
            )
        grid = (image.shape[:3], image.affine)
        if reference_grid is not None and (
            grid[0] != reference_grid[0] or not np.allclose(grid[1], reference_grid[1])
        ):
            raise ValueError(f"HCP run grids differ for subject {subject_id}.")
        reference_grid = grid
        frames = []
        for source, label in task.events:
            frame = _load_ev(run_dir / "EVs" / f"{source}.txt", label)
            if task.single_event_per_condition and len(frame) != 1:
                raise ValueError(
                    f"Expected one event row for {label} in {run_dir / 'EVs'}."
                )
            frames.append(frame)
        events = (
            pd.concat(frames, ignore_index=True)
            .sort_values("onset")
            .reset_index(drop=True)
        )
        if set(events["trial_type"]) != set(task.conditions):
            raise ValueError(f"Incomplete HCP event conditions in {run_dir}.")
        runs.append(
            HCPRun(
                run_name,
                run_name.rsplit("_", 1)[-1],
                bold_path,
                events,
                int(image.shape[3]),
            )
        )
    return tuple(runs)


def fit_subject_task(
    *,
    data_root: Path,
    output_root: Path,
    subject_id: str,
    task_name: str,
    smoothing_fwhm: float | None = None,
) -> dict[str, object]:
    """Fit both HCP task runs and combine them within one participant."""
    task = TASKS[task_name]
    runs = load_runs(data_root, subject_id, task)
    subject_run_dir = output_root / task.name / "run_level" / f"sub-{subject_id}"
    subject_fixed_dir = output_root / task.name / "fixed_effects" / f"sub-{subject_id}"
    mask = common_epi_mask([run.bold_path for run in runs])
    save_image(mask, subject_run_dir / "subject_mask.nii.gz")
    settings = GLMSettings(tr=HCP_TR, smoothing_fwhm=smoothing_fwhm)
    results: list[RunResult] = []
    for run in runs:
        results.append(
            fit_run(
                bold_path=run.bold_path,
                mask_img=mask,
                output_dir=subject_run_dir / f"run-{run.label}",
                contrasts=task.contrasts,
                settings=settings,
                events=run.events,
                run_label=run.label,
            )
        )
    combine_runs(
        run_results=results,
        mask_img=mask,
        output_dir=subject_fixed_dir,
        contrast_stems=[contrast.stem for contrast in task.contrasts],
    )
    return {
        "status": "ok",
        "subject_id": subject_id,
        "task": task.name,
        "n_runs": len(results),
        "error": "",
    }


def task_config(task: Task, subject_ids: list[str]) -> dict[str, object]:
    """Return a concise record of the fixed manuscript model."""
    settings = GLMSettings(tr=HCP_TR)
    return {
        "dataset": "HCP-YA 2025 Task3T Recommended volumetric derivatives",
        "task": task.name,
        "subjects": subject_ids,
        "runs": list(task.runs),
        "bold_suffix": BOLD_SUFFIX,
        "conditions": list(task.conditions),
        "nuisance_conditions": list(task.nuisance),
        "tr_seconds": settings.tr,
        "hrf_model": settings.hrf_model,
        "drift_model": settings.drift_model,
        "high_pass_hz": settings.high_pass,
        "noise_model": settings.noise_model,
        "signal_scaling": settings.signal_scaling,
        "slice_time_ref": settings.slice_time_ref,
        "smoothing_fwhm": None,
        "additional_confounds": None,
        "fixed_effects": "inverse-variance weighted across LR and RL runs",
        "contrasts": [
            {"stem": contrast.stem, "weights": contrast.weights}
            for contrast in task.contrasts
        ],
    }


def write_task_config(output_root: Path, task: Task, subject_ids: list[str]) -> None:
    """Write the task configuration beside its run-level outputs."""
    write_json(
        output_root / task.name / "run_level" / "analysis_config.json",
        task_config(task, subject_ids),
    )
