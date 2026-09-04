"""Dense Amygdala Emotion and movie face-presence input and fitting code."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

from .glm import (
    Contrast,
    GLMSettings,
    RunResult,
    combine_runs,
    common_epi_mask,
    fit_run,
    intersect_masks,
    save_image,
)

INTERNAL_TR = 0.556
EMOTION_RUNS = (1, 2, 3, 4)
# Figure S2 combines nine Budapest and twenty Gump runs per participant.
MOVIES = ("budapest", "gump")
EMOTION_CONTRASTS = (
    Contrast("fear_canonical", {"fear": 1.0}),
    Contrast("fear_derivative", {"fear_derivative": 1.0}),
    Contrast("shape_canonical", {"shape": 1.0}),
    Contrast("shape_derivative", {"shape_derivative": 1.0}),
    Contrast("fear_minus_shape", {"fear": 1.0, "shape": -1.0}),
)
FACE_CONTRAST = (Contrast("face_presence", {"face_presence": 1.0}),)

_CLEAN_BOLD_RE = re.compile(
    r"(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)"
    r"_run-(?P<run>\d+)_recon-clean_part-mag_bold\.nii\.gz$"
)
_EVENT_RE = re.compile(
    r"(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)"
    r"(?:_dir-[^_]+)?(?:_run-(?P<run>\d+))?_events\.tsv$"
)


@dataclass(frozen=True)
class InternalRun:
    """One subject-native cleaned BOLD run and its events."""

    label: str
    bold_path: Path
    events_path: Path


@dataclass(frozen=True)
class MovieRun(InternalRun):
    """One annotated movie run and its run-specific signal mask."""

    movie: str
    mask_path: Path


def normalize_subject(subject: str) -> str:
    """Return a BIDS-formatted subject label."""
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def available_subjects(data_root: Path) -> list[str]:
    """List participants with slab-preprocessed data."""
    root = data_root / "derivatives" / "slabpreproc"
    return [path.name for path in sorted(root.glob("sub-*")) if path.is_dir()]


def _event_path(
    data_root: Path, subject: str, session: str, task: str, run: int
) -> Path:
    return (
        data_root
        / subject
        / session
        / "func"
        / f"{subject}_{session}_task-{task}_dir-AP_run-{run}_events.tsv"
    )


def discover_emotion_runs(data_root: Path, subject: str) -> tuple[InternalRun, ...]:
    """Find the four ICA-FIX-cleaned Dense Amygdala Emotion runs."""
    subject = normalize_subject(subject)
    preproc = data_root / "derivatives" / "slabpreproc" / subject
    found: dict[int, InternalRun] = {}
    for bold_path in sorted(
        preproc.glob("ses-*/preproc/*_recon-clean_part-mag_bold.nii.gz")
    ):
        match = _CLEAN_BOLD_RE.fullmatch(bold_path.name)
        if match is None or match["subject"] != subject:
            continue
        run = int(match["run"])
        if run not in EMOTION_RUNS:
            continue
        events_path = _event_path(
            data_root, subject, match["session"], match["task"], run
        )
        if not events_path.exists():
            continue
        if run in found:
            raise ValueError(f"Duplicate Emotion run {run} for {subject}.")
        found[run] = InternalRun(str(run), bold_path, events_path)
    if sorted(found) != list(EMOTION_RUNS):
        raise FileNotFoundError(
            f"Expected Emotion runs {list(EMOTION_RUNS)} for {subject}; "
            f"found {sorted(found)}."
        )
    return tuple(found[run] for run in EMOTION_RUNS)


def emotion_block_events(path: Path) -> pd.DataFrame:
    """Collapse 36 trial rows into six same-condition Emotion blocks."""
    events = pd.read_csv(path, sep="\t")
    required = {"onset", "duration", "trial_type"}
    if missing := required.difference(events):
        raise ValueError(f"Missing event columns {sorted(missing)} in {path}.")
    events = events[["onset", "duration", "trial_type"]].copy()
    events["onset"] = pd.to_numeric(events["onset"], errors="raise")
    events["duration"] = pd.to_numeric(events["duration"], errors="raise")
    events["trial_type"] = events["trial_type"].astype(str).str.lower()
    events = events.sort_values("onset").reset_index(drop=True)
    numeric = events[["onset", "duration"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (events["onset"] < 0).any():
        raise ValueError(f"Event timings must be finite and nonnegative in {path}.")
    if (events["duration"] <= 0).any():
        raise ValueError(f"Event durations must be positive in {path}.")
    if events["trial_type"].value_counts().to_dict() != {"face": 18, "shape": 18}:
        raise ValueError(f"Expected 18 face and 18 shape trials in {path}.")

    blocks: list[dict[str, object]] = []
    current: list[pd.Series] = []

    def finish_block() -> None:
        if not current:
            return
        first, last = current[0], current[-1]
        blocks.append(
            {
                "onset": float(first["onset"]),
                "duration": float(last["onset"] + last["duration"] - first["onset"]),
                "trial_type": "fear" if str(first["trial_type"]) == "face" else "shape",
                "n_trials": len(current),
            }
        )

    for _, row in events.iterrows():
        if current:
            previous = current[-1]
            gap = float(row["onset"] - previous["onset"] - previous["duration"])
            if row["trial_type"] != previous["trial_type"] or gap > 4.0:
                finish_block()
                current = []
        current.append(row)
    finish_block()
    block_events = pd.DataFrame(blocks)
    if len(block_events) != 6 or set(block_events["n_trials"]) != {6}:
        raise ValueError(f"Expected six blocks of six trials in {path}.")
    if block_events["trial_type"].value_counts().to_dict() != {"fear": 3, "shape": 3}:
        raise ValueError(f"Expected three fear and three shape blocks in {path}.")
    return block_events[["onset", "duration", "trial_type"]]


def fit_emotion_subject(
    *, data_root: Path, output_root: Path, subject: str
) -> dict[str, object]:
    """Fit and combine four Dense Amygdala Emotion runs for one participant."""
    subject = normalize_subject(subject)
    runs = discover_emotion_runs(data_root, subject)
    run_root = output_root / "run_level" / subject
    fixed_root = output_root / "fixed_effects" / subject
    mask = common_epi_mask([run.bold_path for run in runs])
    save_image(mask, run_root / "subject_mask.nii.gz")
    results: list[RunResult] = []
    settings = GLMSettings(tr=INTERNAL_TR)
    for run in runs:
        events = emotion_block_events(run.events_path)
        image = nib.load(run.bold_path)
        final_event = float((events["onset"] + events["duration"]).max())
        if image.ndim != 4 or final_event > image.shape[3] * INTERNAL_TR + 1e-3:
            raise ValueError(f"Emotion events fall outside the run: {run.bold_path}")
        results.append(
            fit_run(
                bold_path=run.bold_path,
                mask_img=mask,
                output_dir=run_root / f"run-{run.label}",
                contrasts=EMOTION_CONTRASTS,
                settings=settings,
                events=events,
                run_label=run.label,
            )
        )
    combine_runs(
        run_results=results,
        mask_img=mask,
        output_dir=fixed_root,
        contrast_stems=[contrast.stem for contrast in EMOTION_CONTRASTS],
    )
    return {
        "status": "ok",
        "subject_id": subject.removeprefix("sub-"),
        "n_runs": len(results),
        "error": "",
    }


def _movie_name(task: str) -> str | None:
    return next((movie for movie in MOVIES if task.startswith(movie)), None)


def _movie_mask(data_root: Path, subject: str, session: str, task: str) -> Path | None:
    base = data_root / "derivatives" / "slabpreproc" / subject / session
    candidates = [
        base / "qc" / f"{subject}_{session}_task-{task}_desc-signal_mask.nii.gz",
        base
        / "qc"
        / f"{subject}_{session}_task-{task}_recon-tmean_part-mag_bold.nii.gz",
        *sorted((base / "atlas").glob("*_desc-brain_mask.nii.gz")),
    ]
    return next((path for path in candidates if path.exists()), None)


def discover_movie_runs(data_root: Path, subject: str) -> tuple[MovieRun, ...]:
    """Find annotated movie runs with cleaned BOLD and a valid slab mask."""
    subject = normalize_subject(subject)
    runs: list[MovieRun] = []
    for events_path in sorted((data_root / subject).glob("ses-*/func/*_events.tsv")):
        match = _EVENT_RE.fullmatch(events_path.name)
        if match is None or match["subject"] != subject:
            continue
        session, task = match["session"], match["task"]
        movie = _movie_name(task)
        if movie is None:
            continue
        columns = pd.read_csv(events_path, sep="\t", nrows=1).columns
        if "face_area_total" not in columns:
            continue
        bold_path = (
            data_root
            / "derivatives"
            / "slabpreproc"
            / subject
            / session
            / "preproc"
            / f"{subject}_{session}_task-{task}_recon-clean_part-mag_bold.nii.gz"
        )
        mask_path = _movie_mask(data_root, subject, session, task)
        if not bold_path.exists() or mask_path is None:
            continue
        annotations = pd.read_csv(events_path, sep="\t", na_values=["n/a"])
        face_area = pd.to_numeric(
            annotations["face_area_total"], errors="coerce"
        ).fillna(0.0)
        if not (face_area > 0).any():
            continue
        runs.append(
            MovieRun(f"{session}_{task}", bold_path, events_path, movie, mask_path)
        )
    if not runs:
        raise FileNotFoundError(f"No annotated movie runs found for {subject}.")
    return tuple(runs)


def face_presence_design(events_path: Path, n_scans: int) -> pd.DataFrame:
    """Build the binary framewise face-presence design used for Figure S2."""
    annotations = pd.read_csv(events_path, sep="\t", na_values=["n/a"])
    if "onset" not in annotations or "face_area_total" not in annotations:
        raise ValueError(f"Missing face-annotation columns in {events_path}.")
    onsets = pd.to_numeric(annotations["onset"], errors="raise").to_numpy(float)
    differences = np.diff(onsets)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    frame_duration = float(np.median(differences)) if differences.size else 1.0 / 30.0
    area = pd.to_numeric(annotations["face_area_total"], errors="coerce").fillna(0.0)
    present = area.to_numpy(float) > 0
    if not np.any(present):
        raise ValueError(f"No face-present frames in {events_path}.")
    events = pd.DataFrame(
        {
            "onset": onsets[present],
            "duration": frame_duration,
            "trial_type": "face_presence",
            "modulation": 1.0,
        }
    )
    return make_first_level_design_matrix(
        frame_times=np.arange(n_scans, dtype=float) * INTERNAL_TR,
        events=events,
        hrf_model="spm + derivative",
        drift_model="cosine",
        high_pass=1.0 / 128.0,
    )


def fit_movie_subject(
    *,
    data_root: Path,
    output_root: Path,
    subject: str,
) -> dict[str, object]:
    """Fit framewise face presence and combine movie runs within participant."""
    subject = normalize_subject(subject)
    runs = discover_movie_runs(data_root, subject)
    if len(runs) != 29:
        raise ValueError(
            f"Expected 29 annotated movie runs for {subject}; found {len(runs)}."
        )
    composition = Counter(run.movie for run in runs)
    if composition != {"budapest": 9, "gump": 20}:
        raise ValueError(
            f"Expected 9 Budapest and 20 Gump runs for {subject}; "
            f"found {dict(composition)}."
        )
    run_root = output_root / subject / "run_level"
    fixed_root = output_root / subject / "fixed_effects"
    mask = intersect_masks([run.mask_path for run in runs])
    save_image(mask, run_root / "subject_mask.nii.gz")
    settings = GLMSettings(tr=INTERNAL_TR)
    results: list[RunResult] = []
    for run in runs:
        image = nib.load(run.bold_path)
        if image.ndim != 4:
            raise ValueError(f"Expected a 4D movie BOLD image: {run.bold_path}")
        results.append(
            fit_run(
                bold_path=run.bold_path,
                mask_img=mask,
                output_dir=run_root / run.label,
                contrasts=FACE_CONTRAST,
                settings=settings,
                design_matrix=face_presence_design(run.events_path, image.shape[3]),
                run_label=run.label,
            )
        )
    combine_runs(
        run_results=results,
        mask_img=mask,
        output_dir=fixed_root,
        contrast_stems=["face_presence"],
    )
    return {
        "status": "ok",
        "subject_id": subject.removeprefix("sub-"),
        "n_runs": len(results),
        "n_budapest_runs": composition.get("budapest", 0),
        "n_gump_runs": composition.get("gump", 0),
        "error": "",
    }
