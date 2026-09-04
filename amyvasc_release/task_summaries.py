"""Participant-first ROI, time-course, and spatial task summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from amyvasc_release.masks import (
    load_image,
    load_mask,
    resample_to_reference,
    same_grid,
)

SUBNUCLEI = (
    "AMY_BLN_La",
    "AMY_BLN_BL_BLD+BLI",
    "AMY_BLN_BM",
    "AMY_CEN",
    "AMY_CMN",
    "AMY_BL_BLV",
    "AMY_ATA",
    "AMY_ATA_ASTA",
    "AMY_AAA",
)
SUBNUCLEUS_METADATA = {
    "AMY_BLN_La": ("Lateral (La)", "La", "basolateral"),
    "AMY_BLN_BL_BLD+BLI": ("Basolateral (BL d+i)", "BL", "basolateral"),
    "AMY_BLN_BM": ("Basomedial (BM)", "BM", "basolateral"),
    "AMY_CEN": ("Central (CeN)", "CeN", "centromedial"),
    "AMY_CMN": ("Corticomedial (CMN)", "CMN", "centromedial"),
    "AMY_BL_BLV": ("Basolateral ventral (BLv)", "BLv", "basolateral"),
    "AMY_ATA": ("Transition area (ATA)", "ATA", "other"),
    "AMY_ATA_ASTA": ("Amygdalo-striatal (ASTA)", "ASTA", "other"),
    "AMY_AAA": ("Anterior (AAA)", "AAA", "other"),
}


@dataclass(frozen=True)
class TrimMask:
    """One cumulative directional trimming stage."""

    direction: str
    depth_mm: float
    mask: np.ndarray


@dataclass(frozen=True)
class DistanceGeometry:
    """Amygdala voxel coordinates and distance to the nearest BVR voxel."""

    indices: np.ndarray
    coordinates_mm: np.ndarray
    distance_mm: np.ndarray


def _finite_values(
    data: np.ndarray,
    mask: np.ndarray,
    support: np.ndarray | None = None,
) -> np.ndarray:
    eligible = np.asarray(mask, dtype=bool) & np.isfinite(data)
    if support is not None:
        eligible &= np.asarray(support, dtype=bool)
    return np.asarray(data[eligible], dtype=np.float64)


def roi_value(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    support: np.ndarray | None = None,
    statistic: str = "median",
) -> tuple[float, int]:
    """Summarize finite supported voxels in one ROI."""
    values = _finite_values(data, mask, support)
    if values.size == 0:
        return np.nan, 0
    functions = {"median": np.median, "mean": np.mean}
    try:
        function = functions[statistic]
    except KeyError as exc:
        raise ValueError("statistic must be 'median' or 'mean'") from exc
    return float(function(values)), int(values.size)


def extract_roi_values(
    image: nib.spatialimages.SpatialImage,
    rois: Mapping[str, np.ndarray],
    *,
    support: np.ndarray | None = None,
    statistic: str = "median",
) -> pd.DataFrame:
    """Extract one voxelwise mean or median and supported count per ROI."""
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    data = image.get_fdata(dtype=np.float32)
    rows = []
    for name, mask in rois.items():
        if mask.shape != image.shape[:3]:
            raise ValueError(f"ROI grid shape mismatch: {name}")
        value, count = roi_value(data, mask, support=support, statistic=statistic)
        rows.append(
            {
                "roi": name,
                f"{statistic}_beta": value,
                "n_voxels": count,
            }
        )
    return pd.DataFrame(rows)


def figure8_mean_summary(
    participant_values: pd.DataFrame,
    *,
    amygdala_roi: str,
    target_roi: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make the reported 23-condition sensitivity/specificity summaries.

    ``participant_values`` contains one bilateral voxelwise mean per participant,
    condition, and ROI. Group condition values are participant means, matching
    the estimand used for Figure 8.
    """
    required = {
        "subject_id",
        "task",
        "condition",
        "condition_key",
        "condition_label",
        "roi",
        "mean_beta",
    }
    missing = required.difference(participant_values.columns)
    if missing:
        raise ValueError(f"Figure 8 input is missing columns: {sorted(missing)}")
    selected = participant_values.loc[
        participant_values["roi"].isin((amygdala_roi, target_roi))
    ].copy()
    observed_rois = set(selected["roi"])
    if observed_rois != {amygdala_roi, target_roi}:
        missing_rois = {amygdala_roi, target_roi}.difference(observed_rois)
        raise ValueError(f"Figure 8 input is missing ROIs: {sorted(missing_rois)}")

    keys = ["task", "condition", "condition_label", "condition_key"]
    condition_order = {
        key: index
        for index, key in enumerate(
            selected["condition_key"].drop_duplicates().astype(str)
        )
    }
    group = (
        selected.groupby([*keys, "roi"], as_index=False, sort=False)["mean_beta"]
        .mean()
        .pivot(index=keys, columns="roi", values="mean_beta")
        .reset_index()
    )
    group["_condition_order"] = group["condition_key"].map(condition_order)
    group = group.sort_values("_condition_order").drop(columns="_condition_order")
    if len(group) != 23 or group["condition_key"].nunique() != 23:
        raise ValueError(
            "Figure 8 requires exactly 23 unique HCP-YA canonical conditions."
        )
    if group[[amygdala_roi, target_roi]].isna().any().any():
        raise ValueError("Figure 8 group condition means contain missing values.")

    amygdala_column = "anatomical_amygdala_mean_canonical_beta_pct_signal"
    target_column = "peri_amygdalar_target_mean_canonical_beta_pct_signal"
    group = group.rename(
        columns={amygdala_roi: amygdala_column, target_roi: target_column}
    )
    group["included_in_emotion_excluded_correlation"] = group["task"].ne("emotion")
    group["display_marker"] = np.where(
        group["condition_key"].eq("motor__tongue"), "diamond", "circle"
    )
    group["annotated"] = group["condition_key"].eq("motor__tongue")

    x = group[amygdala_column].to_numpy(dtype=np.float64)
    y = group[target_column].to_numpy(dtype=np.float64)
    keep = group["included_in_emotion_excluded_correlation"].to_numpy(dtype=bool)
    tongue = group.loc[group["condition_key"].eq("motor__tongue")]
    if len(tongue) != 1:
        raise ValueError("Figure 8 requires exactly one Motor tongue condition.")
    slope, intercept = np.polyfit(x, y, 1)
    statistics_table = pd.DataFrame(
        [
            {
                "measure": "all_conditions_pearson_r",
                "value": float(stats.pearsonr(x, y).statistic),
                "n_conditions": len(group),
            },
            {
                "measure": "emotion_excluded_pearson_r",
                "value": float(stats.pearsonr(x[keep], y[keep]).statistic),
                "n_conditions": int(keep.sum()),
            },
            {
                "measure": "ols_slope",
                "value": float(slope),
                "n_conditions": len(group),
            },
            {
                "measure": "ols_intercept",
                "value": float(intercept),
                "n_conditions": len(group),
            },
            {
                "measure": "tongue_anatomical_amygdala_mean",
                "value": float(tongue.iloc[0][amygdala_column]),
                "n_conditions": 1,
            },
            {
                "measure": "tongue_peri_amygdalar_target_mean",
                "value": float(tongue.iloc[0][target_column]),
                "n_conditions": 1,
            },
        ]
    )
    return group, statistics_table


def wm_face_load_summary(
    participant_values: pd.DataFrame,
    *,
    roi_roles: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize 0-back versus 2-back faces and Amy-target concordance.

    ``roi_roles`` maps the output roles ``amygdala``, ``target``, ``ffc``, and
    ``v1`` to ROI names in ``participant_values``.
    """
    expected_roles = {"amygdala", "target", "ffc", "v1"}
    if set(roi_roles) != expected_roles:
        raise ValueError(f"WM ROI roles must be {sorted(expected_roles)}")
    required = {"subject_id", "task", "condition_key", "roi", "mean_beta"}
    missing = required.difference(participant_values.columns)
    if missing:
        raise ValueError(f"WM input is missing columns: {sorted(missing)}")

    reverse_roles = {roi: role for role, roi in roi_roles.items()}
    if len(reverse_roles) != len(roi_roles):
        raise ValueError("WM ROI roles must refer to four distinct ROIs.")
    selected = participant_values.loc[
        participant_values["task"].eq("wm")
        & participant_values["condition_key"].isin(("wm__0bk_faces", "wm__2bk_faces"))
        & participant_values["roi"].isin(reverse_roles)
    ].copy()
    selected["roi_role"] = selected["roi"].map(reverse_roles)
    selected["load"] = selected["condition_key"].map(
        {"wm__0bk_faces": "0bk", "wm__2bk_faces": "2bk"}
    )
    wide = selected.pivot_table(
        index="subject_id", columns=["roi_role", "load"], values="mean_beta"
    )
    required_cells = pd.MultiIndex.from_product(
        [sorted(expected_roles), ("0bk", "2bk")], names=["roi_role", "load"]
    )
    wide = wide.reindex(columns=required_cells).dropna(how="any")
    if wide.empty:
        raise ValueError("No participants have complete WM face ROI estimates.")

    participant = pd.DataFrame({"subject_id": wide.index.astype(str)})
    summary_rows = []
    for role in ("amygdala", "target", "ffc", "v1"):
        low = wide[(role, "0bk")].to_numpy(dtype=np.float64)
        high = wide[(role, "2bk")].to_numpy(dtype=np.float64)
        suppression = low - high
        participant[f"{role}_0bk_faces"] = low
        participant[f"{role}_2bk_faces"] = high
        participant[f"{role}_suppression"] = suppression
        test = stats.ttest_1samp(suppression, 0.0)
        summary_rows.append(
            {
                "roi": role,
                "mean_0bk": float(low.mean()),
                "mean_2bk": float(high.mean()),
                "suppression": float(suppression.mean()),
                "t": float(test.statistic),
                "p": float(test.pvalue),
                "n_subjects": len(suppression),
            }
        )

    concordance = stats.pearsonr(
        participant["amygdala_suppression"], participant["target_suppression"]
    )
    concordance_table = pd.DataFrame(
        [
            {
                "measure": "amygdala_target_suppression_concordance",
                "pearson_r": float(concordance.statistic),
                "p": float(concordance.pvalue),
                "n_subjects": len(participant),
            }
        ]
    )
    return participant, pd.DataFrame(summary_rows), concordance_table


def mean_ci(values: Sequence[float]) -> dict[str, float | int]:
    """Return a mean, SEM, and two-sided t-based 95% confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    result: dict[str, float | int] = {
        "n": int(array.size),
        "mean": np.nan,
        "sem": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
    }
    if array.size == 0:
        return result
    result["mean"] = float(array.mean())
    if array.size == 1:
        result["sem"] = 0.0
        return result
    sem = float(array.std(ddof=1) / np.sqrt(array.size))
    critical = float(stats.t.ppf(0.975, array.size - 1))
    result.update(
        sem=sem,
        ci95_low=float(array.mean() - critical * sem),
        ci95_high=float(array.mean() + critical * sem),
    )
    return result


def percent_signal_change(series: np.ndarray) -> np.ndarray:
    """Convert a time series to run-mean percent signal change."""
    values = np.asarray(series, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, np.nan)
    baseline = float(values[finite].mean())
    if not np.isfinite(baseline) or abs(baseline) < 1e-8:
        return np.full(values.shape, np.nan)
    output = np.full(values.shape, np.nan)
    output[finite] = 100.0 * (values[finite] - baseline) / baseline
    return output


def collect_participant_pstcs(
    run_manifest: pd.DataFrame,
    roi_images: Mapping[str, nib.spatialimages.SpatialImage],
    *,
    conditions: Mapping[tuple[str, str], str] | None = None,
    tr_seconds: float = 0.72,
    slice_time_ref: float = 0.0,
    window: tuple[float, float] = (-4.0, 32.0),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract event-locked PSC and average events before participants.

    The manifest requires ``subject_id``, ``task``, ``run``, and ``bold_path``.
    Events can be supplied either as an ``events`` DataFrame column or through
    ``events_path``; tables require ``onset`` and ``trial_type``. A common
    positive-mean support mask is applied across every retained run within each
    participant, matching the Figure 4 analysis.
    """
    required = {"subject_id", "task", "run", "bold_path"}
    missing = required - set(run_manifest.columns)
    if missing:
        raise ValueError(f"PSTC manifest is missing columns: {sorted(missing)}")
    if not {"events", "events_path"}.intersection(run_manifest.columns):
        raise ValueError("PSTC manifest requires events or events_path.")
    if tr_seconds <= 0 or window[1] <= window[0]:
        raise ValueError("TR must be positive and the PSTC window must increase.")

    roi_items = list(roi_images.items())
    if not roi_items:
        raise ValueError("At least one ROI is required.")
    reference = roi_items[0][1]
    roi_masks: dict[str, np.ndarray] = {}
    union = np.zeros(reference.shape[:3], dtype=bool)
    for name, image in roi_items:
        if not same_grid(image, reference):
            raise ValueError(f"ROI grid mismatch: {name}")
        mask = image.get_fdata(dtype=np.float32) >= 0.5
        if not np.any(mask):
            raise ValueError(f"ROI is empty: {name}")
        roi_masks[name] = mask
        union |= mask

    relative_times = np.arange(window[0], window[1] + tr_seconds / 2, tr_seconds)
    subject_rows: list[dict[str, object]] = []
    for subject_id, subject_runs in run_manifest.groupby("subject_id", sort=False):
        loaded: list[tuple[object, np.ndarray, pd.DataFrame]] = []
        common_support = np.ones(int(union.sum()), dtype=bool)
        for row in subject_runs.itertuples(index=False):
            bold = load_image(Path(row.bold_path))
            if bold.ndim != 4 or not same_grid(bold, reference):
                raise ValueError(f"BOLD/ROI grid mismatch: {row.bold_path}")
            union_data = np.asarray(bold.dataobj, dtype=np.float32)[union]
            temporal_mean = union_data.mean(axis=1, dtype=np.float64)
            common_support &= np.isfinite(temporal_mean) & (temporal_mean > 0)
            if hasattr(row, "events") and isinstance(row.events, pd.DataFrame):
                events = row.events.copy()
            elif hasattr(row, "events_path"):
                events = pd.read_csv(Path(row.events_path), sep="\t")
            else:
                raise ValueError(f"No events supplied for {row.bold_path}")
            if not {"onset", "trial_type"}.issubset(events.columns):
                raise ValueError(
                    f"Events table lacks onset/trial_type: {row.bold_path}"
                )
            loaded.append((row, union_data, events))
        if not np.any(common_support):
            raise ValueError(f"No common ROI support for participant {subject_id}")

        curves: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
        for row, union_data, events in loaded:
            frame_times = (
                np.arange(union_data.shape[1], dtype=np.float64) + slice_time_ref
            ) * tr_seconds
            for roi_name, roi_mask in roi_masks.items():
                supported = common_support & roi_mask[union]
                if not np.any(supported):
                    continue
                series = percent_signal_change(union_data[supported].mean(axis=0))
                for event in events.itertuples(index=False):
                    key = (str(row.task), str(event.trial_type))
                    if conditions is not None and key not in conditions:
                        continue
                    label = conditions[key] if conditions is not None else key[1]
                    curve = np.interp(
                        float(event.onset) + relative_times,
                        frame_times,
                        series,
                        left=np.nan,
                        right=np.nan,
                    )
                    curves.setdefault((key[0], key[1], label, roi_name), []).append(
                        curve
                    )
        for (task, condition, label, roi_name), event_curves in curves.items():
            subject_curve = np.nanmean(np.stack(event_curves), axis=0)
            for time, psc in zip(relative_times, subject_curve, strict=True):
                subject_rows.append(
                    {
                        "subject_id": str(subject_id),
                        "task": task,
                        "condition": condition,
                        "condition_label": label,
                        "roi": roi_name,
                        "relative_time_sec": float(time),
                        "psc": float(psc),
                    }
                )

    subject = pd.DataFrame(subject_rows)
    if subject.empty:
        raise ValueError("No PSTC observations matched the requested conditions.")
    group = _summarize_curves(
        subject,
        ["task", "condition", "condition_label", "roi", "relative_time_sec"],
    )
    participant_task_mean = (
        subject.groupby(["subject_id", "roi", "relative_time_sec"], as_index=False)[
            "psc"
        ]
        .mean()
        .assign(task="task_mean", condition="task_mean", condition_label="Task mean")
    )
    task_mean = _summarize_curves(
        participant_task_mean,
        ["task", "condition", "condition_label", "roi", "relative_time_sec"],
    )
    return subject, group, task_mean


def _summarize_curves(table: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    rows = []
    for keys, frame in table.groupby(grouping, sort=False):
        summary = mean_ci(frame["psc"].to_numpy(dtype=np.float64))
        row = dict(zip(grouping, keys, strict=True))
        row.update(
            n_subjects=summary["n"],
            mean_psc=summary["mean"],
            sem_psc=summary["sem"],
            ci95_low_psc=summary["ci95_low"],
            ci95_high_psc=summary["ci95_high"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def directional_trim_masks(
    amygdala: np.ndarray,
    bvr: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    *,
    step_mm: float = 2.0,
    max_depth_mm: float | None = None,
) -> list[TrimMask]:
    """Trim the amygdala+BVR union from its medial or lateral boundary."""
    if step_mm <= 0:
        raise ValueError("step_mm must be positive.")
    base = np.asarray(amygdala, dtype=bool) | np.asarray(bvr, dtype=bool)
    indices = np.argwhere(base)
    if indices.size == 0:
        raise ValueError("Amygdala+BVR mask is empty.")
    xyz = nib.affines.apply_affine(reference.affine, indices)
    x = xyz[:, 0]
    absolute_x = np.abs(x)
    selectors = {"left": x < 0, "right": x > 0}
    bounds = {
        name: (float(absolute_x[select].min()), float(absolute_x[select].max()))
        for name, select in selectors.items()
        if np.any(select)
    }
    observed_max = max(high - low for low, high in bounds.values())
    maximum = observed_max if max_depth_mm is None else min(max_depth_mm, observed_max)
    depths = np.arange(0.0, maximum + step_mm / 2, step_mm)
    if not np.isclose(depths[-1], maximum):
        depths = np.append(depths, maximum)

    records = []
    for direction in ("medial", "lateral"):
        for depth in np.unique(np.round(depths, 6)):
            keep = np.zeros(len(indices), dtype=bool)
            for hemisphere, select in selectors.items():
                if hemisphere not in bounds:
                    continue
                medial, lateral = bounds[hemisphere]
                if direction == "medial":
                    keep[select] = absolute_x[select] - medial >= depth - 1e-6
                else:
                    keep[select] = lateral - absolute_x[select] >= depth - 1e-6
            mask = np.zeros(base.shape, dtype=bool)
            mask[tuple(indices[keep].T)] = True
            records.append(TrimMask(direction, float(depth), mask))
    return records


def distance_geometry(
    reference: nib.spatialimages.SpatialImage,
    amygdala: np.ndarray,
    bvr: np.ndarray,
    *,
    exclude_overlap: bool = True,
) -> DistanceGeometry:
    """Calculate millimeter distance from amygdala to the nearest BVR voxel."""
    amygdala = np.asarray(amygdala, dtype=bool)
    bvr = np.asarray(bvr, dtype=bool)
    source = amygdala & ~bvr if exclude_overlap else amygdala
    source_indices = np.argwhere(source)
    bvr_indices = np.argwhere(bvr)
    if source_indices.size == 0 or bvr_indices.size == 0:
        raise ValueError("Distance analysis requires nonempty amygdala and BVR masks.")
    source_xyz = nib.affines.apply_affine(reference.affine, source_indices)
    bvr_xyz = nib.affines.apply_affine(reference.affine, bvr_indices)
    distances, _ = cKDTree(bvr_xyz).query(source_xyz, k=1)
    return DistanceGeometry(source_indices, source_xyz, distances)


def fit_distance_gradient(
    distance_mm: np.ndarray, beta: np.ndarray
) -> dict[str, float]:
    """Fit the descriptive beta-by-distance ordinary least-squares line."""
    valid = np.isfinite(distance_mm) & np.isfinite(beta)
    if valid.sum() < 3 or np.unique(distance_mm[valid]).size < 2:
        raise ValueError("Distance regression has insufficient supported voxels.")
    fit = stats.linregress(distance_mm[valid], beta[valid])
    return {
        "n_voxels": int(valid.sum()),
        "slope_beta_per_mm": float(fit.slope),
        "intercept": float(fit.intercept),
        "pearson_r": float(fit.rvalue),
        "p_value": float(fit.pvalue),
        "stderr": float(fit.stderr),
    }


def load_subnucleus_masks(
    pseg_path: Path,
    labels_path: Path,
    reference: nib.spatialimages.SpatialImage,
    *,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Load the nine CIT168 subnuclei as thresholded masks on a target grid."""
    pseg = load_image(pseg_path, ndim=4)
    labels = [
        line.strip() for line in labels_path.read_text().splitlines() if line.strip()
    ]
    if pseg.shape[3] != len(labels):
        raise ValueError("CIT168 pseg volume count does not match its labels.")
    masks = {}
    for name in SUBNUCLEI:
        if name not in labels:
            raise ValueError(f"CIT168 pseg is missing {name}")
        probability = nib.Nifti1Image(
            np.asarray(pseg.dataobj[..., labels.index(name)], dtype=np.float32),
            pseg.affine,
            pseg.header,
        )
        resampled = resample_to_reference(
            probability,
            reference,
            interpolation="continuous",
        )
        masks[name] = resampled.get_fdata(dtype=np.float32) > threshold
    return masks


def summarize_subnuclei(
    geometry: DistanceGeometry,
    beta: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Summarize task-mean beta and BVR distance within CIT168 subnuclei."""
    rows = []
    voxel_index = tuple(geometry.indices.T)
    for name in SUBNUCLEI:
        selected = masks[name][voxel_index] & np.isfinite(beta)
        if not np.any(selected):
            continue
        long_label, label, group = SUBNUCLEUS_METADATA[name]
        rows.append(
            {
                "subnucleus": name,
                "long_label": long_label,
                "label": label,
                "group": group,
                "median_distance_mm": float(np.median(geometry.distance_mm[selected])),
                "mean_distance_mm": float(np.mean(geometry.distance_mm[selected])),
                "n_voxels_in_amygdala": int(selected.sum()),
                "task_mean_effect": float(np.mean(beta[selected])),
            }
        )
    return pd.DataFrame(rows).sort_values("median_distance_mm").reset_index(drop=True)


def load_roi_images(specs: Mapping[str, Path]) -> dict[str, nib.Nifti1Image]:
    """Load named ROI masks for command-line workflows."""
    loaded = {}
    reference = None
    for name, path in specs.items():
        image, _ = load_mask(path, reference=reference)
        loaded[name] = image
        reference = image if reference is None else reference
    return loaded
