"""Run and summarize the resting-state Rapidtide analysis.

The Rapidtide configuration matches the manuscript analysis. Four HCP-YA
resting-state runs are combined within participant by a voxelwise median over
valid delay estimates. ROI delays and anterior-to-posterior target gradients
are then calculated from the participant consensus maps.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline

from .masks import require_binary_mask

RUNS = (
    "rfMRI_REST1_LR",
    "rfMRI_REST1_RL",
    "rfMRI_REST2_LR",
    "rfMRI_REST2_RL",
)


@dataclass(frozen=True)
class LagTarget:
    key: str
    label: str
    path: Path
    segment: str = ""
    exclusion: str = ""


@dataclass(frozen=True)
class RapidtideRun:
    participant: int
    subject: str
    run: str
    bold: Path
    motion: Path
    motion_six: Path
    output_prefix: Path


@dataclass(frozen=True)
class Consensus:
    subject: str
    delay: np.ndarray
    weighted_delay: np.ndarray
    coverage: np.ndarray
    mean_correlation: np.ndarray
    runs: tuple[str, ...]
    reference: nib.Nifti1Image


def read_subjects(path: Path) -> list[str]:
    """Read subject IDs from text, CSV, or TSV."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        table = pd.read_csv(
            path,
            sep="\t" if path.suffix.lower() == ".tsv" else ",",
            dtype=str,
        )
        for column in ("subject_id", "subject", "Subject"):
            if column in table:
                subjects = table[column].dropna().astype(str).tolist()
                break
        else:
            if table.shape[1] != 1:
                raise ValueError(f"No subject column in {path}")
            subjects = table.iloc[:, 0].dropna().astype(str).tolist()
    else:
        subjects = [
            line.strip() for line in path.read_text().splitlines() if line.strip()
        ]
    subjects = list(dict.fromkeys(subjects))
    if not subjects:
        raise ValueError(f"No participants found in {path}")
    return subjects


def load_targets(path: Path) -> list[LagTarget]:
    """Read ``key, label, path`` target definitions from CSV or TSV."""
    table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    missing = {"key", "label", "path"}.difference(table.columns)
    if missing:
        raise ValueError(f"Target table is missing {sorted(missing)}")
    targets = []
    for row in table.to_dict("records"):
        key = str(row["key"])
        raw_segment = row.get("segment", "")
        raw_exclusion = row.get("exclusion", "")
        segment = "" if pd.isna(raw_segment) else str(raw_segment).strip()
        exclusion = "" if pd.isna(raw_exclusion) else str(raw_exclusion).strip()
        if segment in {"striate_peduncular", "striatal_peduncular"}:
            segment = "combined"
        lower = key.lower()
        if not segment:
            if ("striate" in lower or "striatal" in lower) and "peduncular" in lower:
                segment = "combined"
            elif "peduncular" in lower:
                segment = "peduncular"
            elif "striate" in lower or "striatal" in lower:
                segment = "striate"
        if not exclusion and "pseg" in lower:
            exclusion = "pseg" + lower.rsplit("pseg", 1)[1].split("_", 1)[0]
        target_path = Path(str(row["path"])).expanduser()
        if not target_path.is_absolute():
            target_path = path.parent / target_path
        targets.append(
            LagTarget(key, str(row["label"]), target_path, segment, exclusion)
        )
    return targets


def hcp_rest_paths(hcp_root: Path, subject: str, run: str) -> tuple[Path, Path]:
    base = hcp_root / subject / "MNINonLinear" / "Results" / run
    return base / f"{run}.nii.gz", base / "Movement_Regressors.txt"


def select_rest_cohort(subjects: list[str], hcp_root: Path) -> pd.DataFrame:
    """Audit BOLD and motion availability for the four-run resting-state subset."""
    rows = []
    for subject in subjects:
        row = {"subject_id": subject}
        complete_runs = 0
        for run in RUNS:
            bold, motion = hcp_rest_paths(hcp_root, subject, run)
            row[f"{run}_bold"] = bold.is_file()
            row[f"{run}_motion"] = motion.is_file()
            complete_runs += int(bold.is_file() and motion.is_file())
        row["n_complete_runs"] = complete_runs
        row["included"] = complete_runs == len(RUNS)
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_runs(
    subjects: list[str], hcp_root: Path, output_root: Path
) -> list[RapidtideRun]:
    """Audit four-run inputs and define one Rapidtide output per run."""
    jobs: list[RapidtideRun] = []
    missing: list[Path] = []
    for participant, subject in enumerate(subjects, start=1):
        for run in RUNS:
            bold, motion = hcp_rest_paths(hcp_root, subject, run)
            if not bold.exists():
                missing.append(bold)
            if not motion.exists():
                missing.append(motion)
            run_dir = output_root / subject / run
            jobs.append(
                RapidtideRun(
                    participant=participant,
                    subject=subject,
                    run=run,
                    bold=bold,
                    motion=motion,
                    motion_six=run_dir / "motion_6col.txt",
                    output_prefix=run_dir / f"amyvasc_{subject}_{run}",
                )
            )
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} HCP rest inputs:\n{preview}")
    return jobs


def rapidtide_command(
    job: RapidtideRun,
    *,
    executable: str = "rapidtide",
    n_processes: int = 4,
) -> list[str]:
    """Return the exact command-line configuration used in the manuscript."""
    return [
        executable,
        str(job.bold),
        str(job.output_prefix),
        "--delaymapping",
        "--filterband",
        "lfo",
        "--searchrange",
        "-10",
        "10",
        "--numnull",
        "10000",
        "--timerange",
        "72",
        "-1",
        "--nprocs",
        str(n_processes),
        "--motionfile",
        str(job.motion_six),
        "--nomotderiv",
    ]


def write_run_plan(
    jobs: list[RapidtideRun],
    path: Path,
    *,
    executable: str = "rapidtide",
    n_processes: int = 4,
) -> None:
    """Write a portable shell script containing all per-run commands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for job in jobs:
        lines.append(f"mkdir -p {shlex.quote(str(job.output_prefix.parent))}")
        lines.append(
            "awk '{print $1,$2,$3,$4,$5,$6}' "
            f"{shlex.quote(str(job.motion))} > {shlex.quote(str(job.motion_six))}"
        )
        command = rapidtide_command(job, executable=executable, n_processes=n_processes)
        lines.append(shlex.join(command))
        lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(path.stat().st_mode | 0o111)


def run_rapidtide_jobs(
    jobs: list[RapidtideRun],
    *,
    executable: str = "rapidtide",
    n_processes: int = 4,
    overwrite: bool = False,
) -> None:
    """Run the audited jobs sequentially; external parallelization is optional."""
    for job in jobs:
        expected = Path(f"{job.output_prefix}_desc-maxtimerefined_map.nii.gz")
        if expected.exists() and not overwrite:
            continue
        job.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        motion = np.loadtxt(job.motion, dtype=float)
        if motion.ndim != 2 or motion.shape[1] < 6:
            raise ValueError(f"Motion file has fewer than six columns: {job.motion}")
        np.savetxt(job.motion_six, motion[:, :6], fmt="%.10g")
        subprocess.run(
            rapidtide_command(job, executable=executable, n_processes=n_processes),
            check=True,
        )


def _same_grid(
    a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage
) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4)


def _find_map(directory: Path, prefix: str, description: str) -> Path | None:
    for suffix in ("_map.nii.gz", "_mask.nii.gz"):
        path = directory / f"{prefix}_desc-{description}{suffix}"
        if path.exists():
            return path
    return None


def _load_3d(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    if image.ndim != 3:
        raise ValueError(f"Expected 3D Rapidtide output: {path}")
    return image, image.get_fdata(dtype=np.float32)


def _load_on_grid(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image, data = _load_3d(path)
    if not _same_grid(image, reference):
        raise ValueError(f"Rapidtide map grid differs from delay map: {path}")
    return data


def _run_maps(
    rapidtide_root: Path,
    subject: str,
    run: str,
    *,
    strict: bool = True,
) -> tuple[nib.Nifti1Image, np.ndarray, np.ndarray, np.ndarray] | None:
    directory = rapidtide_root / subject / run
    prefix = f"amyvasc_{subject}_{run}"
    delay_path = _find_map(directory, prefix, "maxtimerefined")
    if delay_path is None and not strict:
        delay_path = _find_map(directory, prefix, "maxtime")
    if delay_path is None:
        if strict:
            raise FileNotFoundError(
                f"Missing refined Rapidtide delay map for {subject}/{run}"
            )
        return None
    image, delay = _load_3d(delay_path)
    correlation_path = _find_map(directory, prefix, "maxcorr")
    if strict and correlation_path is None:
        raise FileNotFoundError(
            f"Missing Rapidtide max-correlation map for {subject}/{run}"
        )
    correlation = (
        _load_on_grid(correlation_path, image)
        if correlation_path
        else np.ones_like(delay)
    )
    valid_path = _find_map(directory, prefix, "corrfit")
    if strict and valid_path is None:
        raise FileNotFoundError(f"Missing Rapidtide corrfit mask for {subject}/{run}")
    valid = (
        _load_on_grid(valid_path, image) > 0
        if valid_path
        else np.isfinite(delay) & (delay != 0)
    )
    return image, delay, correlation, valid & np.isfinite(delay)


def build_consensus(
    rapidtide_root: Path, subject: str, *, strict: bool = True
) -> Consensus | None:
    """Combine valid run-level delay maps within participant."""
    delays: list[np.ndarray] = []
    correlations: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    runs: list[str] = []
    reference: nib.Nifti1Image | None = None
    for run in RUNS:
        maps = _run_maps(rapidtide_root, subject, run, strict=strict)
        if maps is None:
            continue
        image, delay, correlation, valid = maps
        if reference is None:
            reference = image
        elif not _same_grid(image, reference):
            raise ValueError(f"Rapidtide grids differ for {subject}/{run}")
        delays.append(delay)
        correlations.append(correlation)
        valids.append(valid)
        runs.append(run)
    if reference is None:
        return None
    if strict and tuple(runs) != RUNS:
        raise ValueError(f"Expected all four resting-state runs for {subject}")
    delay_stack = np.stack(delays)
    correlation_stack = np.stack(correlations)
    valid_stack = np.stack(valids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        delay = np.nanmedian(np.where(valid_stack, delay_stack, np.nan), axis=0)
        mean_correlation = np.nanmean(
            np.where(valid_stack, correlation_stack, np.nan), axis=0
        )
    weights = np.where(
        valid_stack & np.isfinite(correlation_stack),
        np.clip(correlation_stack, 0, None),
        0,
    )
    weighted_delay = np.divide(
        np.sum(weights * np.where(valid_stack, delay_stack, 0), axis=0),
        weights.sum(axis=0),
        out=np.full(delay.shape, np.nan),
        where=weights.sum(axis=0) > 0,
    )
    return Consensus(
        subject=subject,
        delay=delay.astype(np.float32),
        weighted_delay=weighted_delay.astype(np.float32),
        coverage=valid_stack.sum(axis=0).astype(np.int16),
        mean_correlation=mean_correlation.astype(np.float32),
        runs=tuple(runs),
        reference=reference,
    )


def _like(
    data: np.ndarray, reference: nib.Nifti1Image, dtype: np.dtype
) -> nib.Nifti1Image:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    return image


def save_consensus(rapidtide_root: Path, consensus: Consensus) -> None:
    """Save the four compact maps required for later summary."""
    directory = rapidtide_root / consensus.subject / "consensus"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"amyvasc_{consensus.subject}"
    outputs = {
        "consensusDelay": (consensus.delay, np.float32),
        "consensusDelayWmean": (consensus.weighted_delay, np.float32),
        "consensusCoverage": (consensus.coverage, np.int16),
        "consensusMeanCorr": (consensus.mean_correlation, np.float32),
    }
    for description, (data, dtype) in outputs.items():
        nib.save(
            _like(data, consensus.reference, dtype),
            directory / f"{prefix}_desc-{description}_map.nii.gz",
        )
    (directory / f"{prefix}_consensus_info.json").write_text(
        json.dumps(
            {
                "subject_id": consensus.subject,
                "runs_used": list(consensus.runs),
                "n_runs_found": len(consensus.runs),
                "delay_consensus": "voxelwise median over valid run estimates",
                "validity_criterion": "corrfit > 0, or finite nonzero delay when corrfit is absent",
            },
            indent=2,
        )
        + "\n"
    )


def load_consensus(rapidtide_root: Path, subject: str) -> Consensus | None:
    directory = rapidtide_root / subject / "consensus"
    prefix = f"amyvasc_{subject}"
    paths = {
        key: directory / f"{prefix}_desc-{description}_map.nii.gz"
        for key, description in (
            ("delay", "consensusDelay"),
            ("weighted", "consensusDelayWmean"),
            ("coverage", "consensusCoverage"),
            ("correlation", "consensusMeanCorr"),
        )
    }
    if not paths["delay"].exists() or not paths["coverage"].exists():
        return None
    reference, delay = _load_3d(paths["delay"])
    weighted = (
        _load_on_grid(paths["weighted"], reference)
        if paths["weighted"].exists()
        else np.full(delay.shape, np.nan)
    )
    coverage = _load_on_grid(paths["coverage"], reference).astype(np.int16)
    correlation = (
        _load_on_grid(paths["correlation"], reference)
        if paths["correlation"].exists()
        else np.full(delay.shape, np.nan)
    )
    info_path = directory / f"{prefix}_consensus_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    runs = tuple(info.get("runs_used", info.get("runs", ())))
    return Consensus(subject, delay, weighted, coverage, correlation, runs, reference)


def _load_mask(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = require_binary_mask(path)
    if not _same_grid(image, reference):
        raise ValueError(f"Mask must be a 3D image on the Rapidtide grid: {path}")
    return np.asarray(image.dataobj) > 0.5


def _roi_delay(
    consensus: Consensus, mask: np.ndarray, min_coverage: int
) -> dict[str, float | int]:
    valid = mask & np.isfinite(consensus.delay) & (consensus.coverage >= min_coverage)
    values = consensus.delay[valid]
    correlations = consensus.mean_correlation[valid]
    correlations = correlations[np.isfinite(correlations)]
    return {
        "n_total": int(mask.sum()),
        "n_valid": int(valid.sum()),
        "pct_valid": 100.0 * valid.sum() / mask.sum(),
        "delay_median_s": float(np.median(values)) if values.size else np.nan,
        "delay_mean_s": float(np.mean(values)) if values.size else np.nan,
        "correlation_median": float(np.median(correlations))
        if correlations.size
        else np.nan,
    }


def _centerline(
    mask: np.ndarray, reference: nib.Nifti1Image
) -> tuple[np.ndarray, np.ndarray]:
    """Return target voxel indices and anterior-to-posterior arc coordinates."""
    ijk = np.argwhere(mask)
    xyz = nib.affines.apply_affine(reference.affine, ijk)
    distances = np.full(len(ijk), np.nan)
    for side in (xyz[:, 0] < 0, xyz[:, 0] > 0):
        local = np.where(side)[0]
        if local.size == 0:
            continue
        coords = xyz[local]
        y_values = np.sort(np.unique(coords[:, 1]))[::-1]
        centroids = []
        counts = []
        for y in y_values:
            selected = np.isclose(coords[:, 1], y)
            centroids.append(
                [coords[selected, 0].mean(), y, coords[selected, 2].mean()]
            )
            counts.append(selected.sum())
        curve = np.asarray(centroids, float)
        if len(curve) >= 5:
            axis = np.arange(len(curve), dtype=float)
            weight = np.sqrt(counts)
            curve[:, 0] = UnivariateSpline(axis, curve[:, 0], w=weight)(axis)
            curve[:, 2] = UnivariateSpline(axis, curve[:, 2], w=weight)(axis)
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        )
        nearest = np.linalg.norm(coords[:, None, :] - curve[None, :, :], axis=2).argmin(
            axis=1
        )
        distances[local] = arc[nearest]
    if not np.isfinite(distances).any():
        raise ValueError("Target has no left or right hemisphere voxels")
    return ijk, distances


def _bootstrap_ci(
    values: np.ndarray, seed: int, draws: int = 5000
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _one_sample(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan, np.nan
    test = stats.ttest_1samp(values, 0)
    try:
        wilcoxon = float(stats.wilcoxon(values).pvalue)
    except ValueError:
        wilcoxon = np.nan
    return float(test.statistic), float(test.pvalue), wilcoxon


def summarize_lags(rows: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    summaries = []
    for target_key, frame in rows.groupby("target_key", sort=False):
        values = frame["target_minus_amy_ms"].to_numpy(float)
        values = values[np.isfinite(values)]
        low, high = _bootstrap_ci(values, seed)
        statistic, p_value, wilcoxon = _one_sample(values)
        summaries.append(
            {
                "target_key": target_key,
                "target_label": frame["target_label"].iloc[0],
                "segment": frame["segment"].iloc[0],
                "exclusion": frame["exclusion"].iloc[0],
                "n_participants": len(values),
                "mean_target_minus_amy_ms": np.mean(values),
                "median_target_minus_amy_ms": np.median(values),
                "ci95_low_ms": low,
                "ci95_high_ms": high,
                "t_statistic": statistic,
                "t_p_value": p_value,
                "wilcoxon_p_value": wilcoxon,
                "pct_target_later_than_amy": 100 * np.mean(values > 0),
            }
        )
    return pd.DataFrame(summaries)


def summarize_gradients(rows: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    summaries = []
    for target_key, frame in rows.groupby("target_key", sort=False):
        values = frame["slope_ms_per_mm"].to_numpy(float)
        values = values[np.isfinite(values)]
        low, high = _bootstrap_ci(values, seed)
        statistic, p_value, wilcoxon = _one_sample(values)
        summaries.append(
            {
                "target_key": target_key,
                "target_label": frame["target_label"].iloc[0],
                "segment": frame["segment"].iloc[0],
                "exclusion": frame["exclusion"].iloc[0],
                "n_participants": len(values),
                "mean_slope_ms_per_mm": np.mean(values),
                "median_slope_ms_per_mm": np.median(values),
                "ci95_low_ms_per_mm": low,
                "ci95_high_ms_per_mm": high,
                "t_statistic": statistic,
                "t_p_value": p_value,
                "wilcoxon_p_value": wilcoxon,
                "pct_positive": 100 * np.mean(values > 0),
            }
        )
    return pd.DataFrame(summaries)


def summarize_segment_ordering(
    lag_rows: pd.DataFrame, *, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate participantwise peduncular-minus-striate lag ordering."""
    participant_rows: list[dict[str, object]] = []
    for exclusion, frame in lag_rows.groupby("exclusion", sort=False):
        pivot = frame.pivot_table(
            index="participant",
            columns="segment",
            values="target_delay_median_s",
            aggfunc="first",
        )
        striate = next(
            (column for column in ("striate", "striatal") if column in pivot),
            None,
        )
        if striate is None or "peduncular" not in pivot:
            continue
        difference = 1000.0 * (pivot["peduncular"] - pivot[striate])
        for participant, value in difference.items():
            if not np.isfinite(value):
                continue
            participant_rows.append(
                {
                    "participant": participant,
                    "exclusion": exclusion,
                    "peduncular_minus_striate_ms": value,
                    "peduncular_later_than_striate": bool(value > 0),
                }
            )
    participant_table = pd.DataFrame(participant_rows)
    summary_rows = []
    if not participant_table.empty:
        for exclusion, frame in participant_table.groupby("exclusion", sort=False):
            values = frame["peduncular_minus_striate_ms"].to_numpy(float)
            low, high = _bootstrap_ci(values, seed)
            summary_rows.append(
                {
                    "exclusion": exclusion,
                    "n_participants": len(values),
                    "mean_peduncular_minus_striate_ms": np.mean(values),
                    "median_peduncular_minus_striate_ms": np.median(values),
                    "ci95_low_ms": low,
                    "ci95_high_ms": high,
                    "pct_peduncular_later_than_striate": 100 * np.mean(values > 0),
                }
            )
    return participant_table, pd.DataFrame(summary_rows)


def reported_timing_summary(
    lag_summary: pd.DataFrame,
    gradient_summary: pd.DataFrame,
    ordering_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the lag ordering and combined-target slope for each exclusion."""
    rows: list[dict[str, object]] = []
    for exclusion in lag_summary["exclusion"].drop_duplicates():
        lags = lag_summary.loc[lag_summary["exclusion"].eq(exclusion)].set_index(
            "segment"
        )
        gradients = gradient_summary.loc[
            gradient_summary["exclusion"].eq(exclusion)
        ].set_index("segment")
        striate = next(
            (segment for segment in ("striate", "striatal") if segment in lags.index),
            None,
        )
        if striate is None or "peduncular" not in lags.index:
            continue
        ordering = (
            ordering_summary.loc[ordering_summary["exclusion"].eq(exclusion)]
            if not ordering_summary.empty
            else ordering_summary
        )
        row: dict[str, object] = {
            "exclusion": exclusion,
            "median_striate_minus_amygdala_ms": lags.loc[
                striate, "median_target_minus_amy_ms"
            ],
            "median_peduncular_minus_amygdala_ms": lags.loc[
                "peduncular", "median_target_minus_amy_ms"
            ],
            "pct_striate_later_than_amygdala": lags.loc[
                striate, "pct_target_later_than_amy"
            ],
            "pct_peduncular_later_than_amygdala": lags.loc[
                "peduncular", "pct_target_later_than_amy"
            ],
        }
        if not ordering.empty:
            row["pct_peduncular_later_than_striate"] = ordering.iloc[0][
                "pct_peduncular_later_than_striate"
            ]
        if "combined" in gradients.index:
            row.update(
                {
                    "combined_slope_mean_ms_per_mm": gradients.loc[
                        "combined", "mean_slope_ms_per_mm"
                    ],
                    "combined_slope_median_ms_per_mm": gradients.loc[
                        "combined", "median_slope_ms_per_mm"
                    ],
                    "pct_positive_combined_slopes": gradients.loc[
                        "combined", "pct_positive"
                    ],
                    "n_participants_combined_slope": gradients.loc[
                        "combined", "n_participants"
                    ],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_rapidtide(
    *,
    rapidtide_root: Path,
    subjects: list[str],
    amygdala_mask_path: Path,
    targets: list[LagTarget],
    output_dir: Path,
    build_missing_consensus: bool = True,
    include_run_reproducibility: bool = True,
    min_coverage: int = 2,
    n_bins: int = 6,
    min_bin_voxels: int = 2,
    min_valid_bins: int = 3,
    strict: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build consensus maps and write ROI, QC, reproducibility, and gradient tables."""
    first = None
    first_subject = ""
    first_was_built = False
    for subject in subjects:
        first = load_consensus(rapidtide_root, subject)
        if first is None and build_missing_consensus:
            first = build_consensus(rapidtide_root, subject, strict=strict)
            first_was_built = first is not None
        if first is not None:
            if strict and tuple(first.runs) != RUNS:
                raise ValueError(
                    f"Consensus metadata does not list all four runs for {subject}"
                )
            first_subject = subject
            break
    if first is None:
        raise FileNotFoundError(f"No Rapidtide delay maps under {rapidtide_root}")
    amy = _load_mask(amygdala_mask_path, first.reference)
    target_masks = {}
    for target in targets:
        target_mask = _load_mask(target.path, first.reference) & ~amy
        if not np.any(target_mask):
            raise ValueError(f"Target is empty after amygdala exclusion: {target.key}")
        target_masks[target.key] = target_mask
    geometries = {
        key: _centerline(mask, first.reference) for key, mask in target_masks.items()
    }

    qc_rows = []
    lag_rows = []
    gradient_rows = []
    bin_rows = []
    reproducibility_rows = []
    for participant, subject in enumerate(subjects, start=1):
        consensus = (
            first
            if subject == first_subject
            else load_consensus(rapidtide_root, subject)
        )
        if subject == first_subject and first_was_built:
            save_consensus(rapidtide_root, consensus)
        if consensus is None and build_missing_consensus:
            consensus = build_consensus(rapidtide_root, subject, strict=strict)
            if consensus is not None:
                save_consensus(rapidtide_root, consensus)
        if consensus is None:
            if strict:
                raise FileNotFoundError(f"No consensus delay map for {subject}")
            continue
        if strict and tuple(consensus.runs) != RUNS:
            raise ValueError(
                f"Consensus metadata does not list all four runs for {subject}"
            )
        if not _same_grid(consensus.reference, first.reference):
            raise ValueError(f"Consensus grid differs for participant {participant}")
        finite = np.isfinite(consensus.delay)
        qc_rows.append(
            {
                "participant": participant,
                "n_runs": len(consensus.runs) or int(consensus.coverage.max()),
                "n_voxels_any_valid": int((consensus.coverage > 0).sum()),
                "n_voxels_min_coverage": int(
                    (consensus.coverage >= min_coverage).sum()
                ),
                "median_coverage": float(np.median(consensus.coverage[finite])),
                "median_correlation": float(
                    np.nanmedian(consensus.mean_correlation[finite])
                ),
            }
        )
        run_maps = []
        if include_run_reproducibility:
            for run in RUNS:
                maps = _run_maps(rapidtide_root, subject, run, strict=strict)
                if maps is not None:
                    if not _same_grid(maps[0], consensus.reference):
                        raise ValueError(
                            f"Run map grid differs from consensus for {subject}/{run}"
                        )
                    run_maps.append(maps)
        for target in targets:
            target_mask = target_masks[target.key]
            amy_mask = amy & ~target_masks[target.key]
            amy_stats = _roi_delay(consensus, amy_mask, min_coverage)
            target_stats = _roi_delay(consensus, target_mask, min_coverage)
            difference = target_stats["delay_median_s"] - amy_stats["delay_median_s"]
            lag_rows.append(
                {
                    "participant": participant,
                    "target_key": target.key,
                    "target_label": target.label,
                    "segment": target.segment,
                    "exclusion": target.exclusion,
                    "amy_n_valid": amy_stats["n_valid"],
                    "target_n_valid": target_stats["n_valid"],
                    "amy_delay_median_s": amy_stats["delay_median_s"],
                    "target_delay_median_s": target_stats["delay_median_s"],
                    "target_minus_amy_ms": 1000 * difference,
                    "target_later_than_amy": difference > 0,
                }
            )

            run_differences = []
            for _image, delay, _correlation, valid in run_maps:
                amy_values = delay[amy_mask & valid]
                target_values = delay[target_mask & valid]
                if amy_values.size and target_values.size:
                    run_differences.append(
                        np.median(target_values) - np.median(amy_values)
                    )
            if include_run_reproducibility:
                reproducibility_rows.append(
                    {
                        "participant": participant,
                        "target_key": target.key,
                        "segment": target.segment,
                        "exclusion": target.exclusion,
                        "n_valid_runs": len(run_differences),
                        "run_difference_sd_s": np.std(run_differences, ddof=1)
                        if len(run_differences) > 1
                        else np.nan,
                        "run_difference_range_s": np.ptp(run_differences)
                        if run_differences
                        else np.nan,
                    }
                )

            ijk, distance = geometries[target.key]
            tuple_ijk = tuple(ijk[:, axis] for axis in range(3))
            delay = consensus.delay[tuple_ijk]
            coverage = consensus.coverage[tuple_ijk]
            valid = (
                np.isfinite(delay) & np.isfinite(distance) & (coverage >= min_coverage)
            )
            local_distance = distance[valid]
            local_delay = delay[valid]
            edges = np.linspace(
                np.nanmin(distance), np.nanmax(distance) + 1e-10, n_bins + 1
            )
            centers = (edges[:-1] + edges[1:]) / 2
            assigned = np.clip(np.digitize(local_distance, edges) - 1, 0, n_bins - 1)
            fit_x, fit_y = [], []
            for bin_index, center in enumerate(centers):
                selected = assigned == bin_index
                median = (
                    np.median(local_delay[selected])
                    if selected.sum() >= min_bin_voxels
                    else np.nan
                )
                if np.isfinite(median):
                    fit_x.append(center)
                    fit_y.append(median)
                bin_rows.append(
                    {
                        "participant": participant,
                        "target_key": target.key,
                        "target_label": target.label,
                        "segment": target.segment,
                        "exclusion": target.exclusion,
                        "bin": bin_index,
                        "distance_mm": center,
                        "n_voxels": int(selected.sum()),
                        "median_delay_ms": 1000 * median,
                    }
                )
            slope = r_squared = p_value = np.nan
            if len(fit_x) >= min_valid_bins and len(np.unique(fit_x)) > 1:
                regression = stats.linregress(fit_x, fit_y)
                slope = 1000 * regression.slope
                r_squared = regression.rvalue**2
                p_value = regression.pvalue
            gradient_rows.append(
                {
                    "participant": participant,
                    "target_key": target.key,
                    "target_label": target.label,
                    "segment": target.segment,
                    "exclusion": target.exclusion,
                    "n_valid_voxels": int(valid.sum()),
                    "n_valid_bins": len(fit_x),
                    "slope_ms_per_mm": slope,
                    "r_squared": r_squared,
                    "p_value": p_value,
                }
            )

    participant_lags = pd.DataFrame(lag_rows)
    participant_gradients = pd.DataFrame(gradient_rows)
    ordering, ordering_summary = summarize_segment_ordering(participant_lags)
    lag_summary = summarize_lags(participant_lags)
    gradient_summary = summarize_gradients(participant_gradients)
    tables = {
        "consensus_qc": pd.DataFrame(qc_rows),
        "participant_lags": participant_lags,
        "lag_summary": lag_summary,
        "segment_ordering_participant": ordering,
        "segment_ordering_summary": ordering_summary,
        "participant_gradients": participant_gradients,
        "gradient_bins": pd.DataFrame(bin_rows),
        "gradient_summary": gradient_summary,
        "reported_timing_summary": reported_timing_summary(
            lag_summary, gradient_summary, ordering_summary
        ),
    }
    if include_run_reproducibility:
        tables["run_reproducibility"] = pd.DataFrame(reproducibility_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.tsv", sep="\t", index=False)
    return tables
