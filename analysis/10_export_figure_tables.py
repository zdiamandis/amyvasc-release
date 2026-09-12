#!/usr/bin/env python3
"""Translate analysis summaries into the tables read by figure renderers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FIGURE1 = (
    ("emotion__face", "emotion", "fear", "Emotion\nfear", "#4C78A8"),
    ("emotion__shape", "emotion", "shape", "Emotion\nshape", "#4C78A8"),
    ("wm__0bk_faces", "wm", "0bk_faces", "WM\nfaces", "#F28E2B"),
    ("wm__0bk_places", "wm", "0bk_places", "WM\nplaces", "#F28E2B"),
    ("wm__0bk_tools", "wm", "0bk_tools", "WM\ntools", "#F28E2B"),
    ("social__mental", "social", "mental", "Social\nmental", "#59A14F"),
    ("social__random", "social", "random", "Social\nrandom", "#59A14F"),
    ("language__story", "language", "story", "Language\nstory", "#7A5EA8"),
    ("language__math", "language", "math", "Language\nmath", "#7A5EA8"),
)

CONDITION_LABELS = {
    "emotion__face": "fear",
    "emotion__shape": "shape",
    "wm__0bk_faces": "0bk\nface",
    "wm__0bk_places": "0bk\nplace",
    "wm__0bk_tools": "0bk\ntool",
    "wm__0bk_body": "0bk\nbody",
    "wm__2bk_faces": "2bk\nface",
    "wm__2bk_places": "2bk\nplace",
    "wm__2bk_tools": "2bk\ntool",
    "wm__2bk_body": "2bk\nbody",
    "social__mental": "mental",
    "social__random": "random",
    "language__story": "story",
    "language__math": "math",
    "gambling__win": "win",
    "gambling__loss": "loss",
    "relational__relational": "rel.",
    "relational__matching": "match",
    "motor__left_hand": "L hand",
    "motor__right_hand": "R hand",
    "motor__left_foot": "L foot",
    "motor__right_foot": "R foot",
    "motor__tongue": "tongue",
}

# Analysis key, displayed key, correlation label, fingerprint label.
CANDIDATES = (
    ("amy_proper", "amy", "Amygdala", "Amygdala"),
    ("glasser__pir", "pir", "Piriform cortex (Pir)", "Piriform cortex (Pir)"),
    (
        "glasser__47m",
        "ofc_47m",
        "Orbitofrontal area 47m",
        "Orbitofrontal area 47m",
    ),
    (
        "anterior_rhinal",
        "anterior_rhinal",
        "Rhinal cortex",
        "Rhinal cortex",
    ),
    (
        "tian_s2__ahip",
        "ahip",
        "Anterior hippocampus",
        "Anterior hippocampus",
    ),
    (
        "posterior_insula",
        "posterior_insula",
        "Posterior insula",
        "Posterior insula",
    ),
    (
        "glasser__ffc",
        "ffc",
        "Fusiform face complex (FFC)",
        "Fusiform face complex (FFC)",
    ),
)

CANDIDATE_ROUTES = {
    "amy_proper": "Peduncular / medial temporal",
    "glasser__pir": "Peduncular / medial temporal",
    "glasser__47m": "First / striate",
    "anterior_rhinal": "Peduncular / medial temporal",
    "tian_s2__ahip": "Peduncular / medial temporal",
    "posterior_insula": "First / striate",
    "glasser__ffc": "Distal alternative",
}

CANDIDATE_ROLES = {
    "amy_proper": "Primary hypothesized source",
    "glasser__pir": "Adjacent medial-temporal candidate",
    "glasser__47m": "Better-covered orbitofrontal representative",
    "anterior_rhinal": "Rhinal-cortex composite",
    "tian_s2__ahip": "Documented medial-temporal tributary territory",
    "posterior_insula": "Deep-middle-cerebral-vein tributary territory",
    "glasser__ffc": "Historical distal-visual hypothesis",
}

CANDIDATE_MEMBERS = {
    "glasser__pir": "glasser__pir",
    "glasser__47m": "glasser__47m",
    "anterior_rhinal": "glasser__ec;glasser__peec",
    "tian_s2__ahip": "tian_s2__ahip",
    "posterior_insula": "glasser__ig;glasser__pi;glasser__poi1;glasser__poi2",
    "glasser__ffc": "glasser__ffc",
}

SPATIAL = {
    "emotion_fear": ("Emotion fear", "#C44E52"),
    "wm_0bk_faces": ("WM faces", "#DD8452"),
    "social_mental": ("Social mental", "#55A868"),
    "language_story": ("Language story", "#8172B2"),
}

SOURCE_SUPPORT_COLUMNS = (
    "n_cohort",
    "n_paired_cell_participants_min",
    "n_paired_cell_participants_median",
    "n_paired_cell_participants_max",
)


def read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def emit(
    frame: pd.DataFrame,
    output_dir: Path,
    name: str,
    columns: tuple[str, ...] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    (frame if columns is None else frame.loc[:, columns]).to_csv(
        path, sep="\t", index=False
    )
    print(path)
    return path


def unique_row(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"Expected one {description} row, found {len(frame)}")
    return frame.iloc[0]


def require_profile_statistic(root: Path, statistic: str) -> None:
    """Prevent a sensitivity run from silently replacing the primary statistic."""
    metadata = json.loads((root / "run_metadata.json").read_text())
    if metadata.get("voxel_statistic") != statistic:
        raise ValueError(f"{root} must contain a {statistic}-based source-profile run")


def pseg(value: object) -> str | None:
    match = re.search(r"(?:0[.]|p(?:amy|seg)?\s*)(50|20|05)\b", str(value).lower())
    return f"pseg{match.group(1)}" if match else None


def canonical_target_key(value: object) -> str:
    """Translate generated target keys to the manuscript table convention."""
    key = str(value)
    if not key.startswith("hcp_"):
        key = "hcp_" + key
    return key.replace("bvr_striate_", "bvr_striatal_").replace("_pamy", "_pseg")


def export_roi(root: Path, out: Path, amygdala_roi: str) -> list[Path]:
    group = read(
        root / "group_roi_means.tsv",
        ("condition_key", "roi", "mean", "sem", "n_subjects"),
    )
    group = group.loc[group["roi"].eq(amygdala_roi)].set_index("condition_key")
    rows = []
    for index, (key, task, condition, label, color) in enumerate(FIGURE1):
        row = unique_row(group.loc[group.index == key], f"Figure 1 {key}")
        rows.append(
            {
                "plot_index": index,
                "task": task,
                "condition": condition,
                "x_label": label,
                "color": color,
                "group_mean": row["mean"],
                "group_sem": row["sem"],
                "n_subjects": int(row["n_subjects"]),
            }
        )
    outputs = [emit(pd.DataFrame(rows), out, "figure1_panel_a_summary.tsv")]

    figure8 = read(
        root / "figure8_condition_profile.tsv",
        (
            "condition_label",
            "condition_key",
            "anatomical_amygdala_mean_canonical_beta_pct_signal",
            "peri_amygdalar_target_mean_canonical_beta_pct_signal",
            "included_in_emotion_excluded_correlation",
            "annotated",
        ),
    )
    if len(figure8) != 23 or figure8["condition_key"].nunique() != 23:
        raise ValueError("Figure 8 requires exactly 23 canonical conditions")
    outputs.append(emit(figure8, out, "figure8_source_data.tsv"))

    s1 = figure8.rename(
        columns={
            "peri_amygdalar_target_mean_canonical_beta_pct_signal": (
                "bvr_striate_peduncular_pamy50"
            ),
            "anatomical_amygdala_mean_canonical_beta_pct_signal": "amy_proper",
        }
    )
    outputs.append(
        emit(
            s1,
            out,
            "figureS1_condition_amygdala_trace_profile.tsv",
            (
                "task",
                "condition",
                "condition_label",
                "condition_key",
                "bvr_striate_peduncular_pamy50",
                "amy_proper",
            ),
        )
    )
    outputs.append(
        emit(
            read(
                root / "wm_face_load_summary.tsv",
                ("roi", "mean_0bk", "mean_2bk", "suppression", "t", "p"),
            ),
            out,
            "figure7_wm_load.tsv",
            ("roi", "mean_0bk", "mean_2bk", "suppression", "t", "p"),
        )
    )
    outputs.append(
        emit(
            read(
                root / "wm_face_load_concordance.tsv",
                ("measure", "pearson_r", "p", "n_subjects"),
            ),
            out,
            "figure7_wm_concordance.tsv",
        )
    )
    return outputs


def export_pstc(root: Path, out: Path, args: argparse.Namespace) -> list[Path]:
    table = read(root / "pstc_task_mean.tsv", ("roi", "relative_time_sec", "mean_psc"))
    support = read(
        root / "pstc_group.tsv",
        ("condition", "roi", "relative_time_sec", "n_subjects"),
    )
    support = support.groupby(["roi", "relative_time_sec"], as_index=False).agg(
        n_task_conditions=("condition", "nunique"),
        source_n_subjects_min=("n_subjects", "min"),
        source_n_subjects_max=("n_subjects", "max"),
    )
    specs = (
        (args.pstc_amygdala_roi, "CIT168 amygdala", "#00E676"),
        (args.pstc_peduncular_roi, "Peduncular BVR trace", "#7A4EA3"),
        (args.pstc_striate_roi, "Striate BVR trace", "#E45756"),
    )
    rows = []
    for roi, label, color in specs:
        selected = table.loc[table["roi"].eq(roi)].copy()
        if selected.empty:
            raise ValueError(f"PSTC summary is missing ROI: {roi}")
        selected["roi_label"], selected["color"] = label, color
        selected = selected.merge(
            support.loc[support["roi"].eq(roi)],
            on=("roi", "relative_time_sec"),
            how="left",
            validate="one_to_one",
        )
        if (
            selected[
                ["n_task_conditions", "source_n_subjects_min", "source_n_subjects_max"]
            ]
            .isna()
            .any()
            .any()
        ):
            raise ValueError(f"PSTC support summary is incomplete for ROI: {roi}")
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    return [
        emit(
            result,
            out,
            "figure4_panel_b_task_mean_pstc.tsv",
            (
                "roi",
                "roi_label",
                "color",
                "relative_time_sec",
                "mean_psc",
                "n_task_conditions",
                "source_n_subjects_min",
                "source_n_subjects_max",
            ),
        )
    ]


def export_spatial(root: Path, out: Path) -> list[Path]:
    trimming = read(
        root / "directional_summary.tsv",
        (
            "condition_key",
            "direction",
            "depth_mm",
            "mean_beta",
            "ci95_low_beta",
            "ci95_high_beta",
        ),
    )
    trimming = trimming.loc[trimming["condition_key"].eq("task_mean")]
    if trimming.empty:
        raise ValueError("Directional summary has no participant-first task mean")
    participant = read(
        root / "directional_participant.tsv",
        ("subject_id", "profile", "direction", "depth_mm", "beta"),
    )
    participant_task_mean = participant.groupby(
        ["subject_id", "profile", "direction", "depth_mm"], as_index=False
    )["beta"].mean()
    medians = (
        participant_task_mean.groupby(
            ["profile", "direction", "depth_mm"], as_index=False
        )["beta"]
        .median()
        .rename(columns={"beta": "median_beta"})
    )
    trimming = trimming.merge(
        medians,
        on=("profile", "direction", "depth_mm"),
        how="left",
        validate="one_to_one",
    )
    if trimming["median_beta"].isna().any():
        raise ValueError("Directional participant medians are incomplete")
    trimming["profile"] = trimming["profile"].replace(
        {"amy_bvr_directional_trim": "amy_ros_directional_erosion"}
    )
    outputs = [
        emit(
            trimming,
            out,
            "figure4_panel_c_directional_trimming.tsv",
            (
                "condition_key",
                "condition_label",
                "profile",
                "direction",
                "depth_mm",
                "n_observations",
                "mean_beta",
                "median_beta",
                "ci95_low_beta",
                "ci95_high_beta",
            ),
        )
    ]

    voxels = read(
        root / "distance_voxels.tsv",
        ("condition_key", "distance_to_bvr_mm", "group_median_effect"),
    ).rename(columns={"distance_to_bvr_mm": "distance_to_ros_mm"})
    outputs.append(
        emit(
            voxels,
            out,
            "figure4_panel_d_voxel_observations.tsv",
            (
                "condition_key",
                "condition_label",
                "voxel_i",
                "voxel_j",
                "voxel_k",
                "mni_x_mm",
                "mni_y_mm",
                "mni_z_mm",
                "distance_to_ros_mm",
                "n_subjects_support",
                "group_median_effect",
            ),
        )
    )

    regressions = read(
        root / "distance_group_regressions.tsv",
        ("condition_key", "slope_beta_per_mm", "intercept"),
    ).copy()
    unknown = set(regressions["condition_key"]).difference(SPATIAL)
    if unknown:
        raise ValueError(f"Unexpected spatial conditions: {sorted(unknown)}")
    regressions["condition_label"] = regressions["condition_key"].map(
        lambda key: SPATIAL[key][0]
    )
    regressions["color"] = regressions["condition_key"].map(lambda key: SPATIAL[key][1])
    outputs.append(
        emit(
            regressions,
            out,
            "figure4_panel_d_voxel_gradient_summary.tsv",
            (
                "condition_key",
                "condition_label",
                "color",
                "n_voxels",
                "slope_beta_per_mm",
                "intercept",
                "pearson_r",
                "p_value",
                "stderr",
            ),
        )
    )

    subnuclei = read(
        root / "subnucleus_summary.tsv",
        (
            "label",
            "group",
            "median_distance_mm",
            "n_voxels_in_amygdala",
            "task_mean_effect",
        ),
    )
    outputs.append(emit(subnuclei, out, "figure4_panel_e_subnucleus_summary.tsv"))
    slopes = read(
        root / "distance_slope_summary.tsv",
        ("condition_key", "n_subjects", "mean", "sem", "ci95_low", "ci95_high"),
    ).rename(
        columns={
            "mean": "mean_slope_beta_per_mm",
            "sem": "sem_slope_beta_per_mm",
            "ci95_low": "ci95_low_slope_beta_per_mm",
            "ci95_high": "ci95_high_slope_beta_per_mm",
        }
    )
    outputs.append(emit(slopes, out, "figure4_participant_distance_slope_summary.tsv"))
    return outputs


def export_smoothing(root: Path, out: Path, threshold: float) -> list[Path]:
    if not np.isclose(threshold, 3.0):
        raise ValueError("Figure 5B is defined at z >= 3")
    table = read(
        root / "group_extent.tsv",
        (
            "smoothing_fwhm_mm",
            "mean_percent_z_ge_threshold",
            "ci95_low_percent_z_ge_threshold",
            "ci95_high_percent_z_ge_threshold",
            "mean_effect",
        ),
    ).rename(
        columns={
            "mean_percent_z_ge_threshold": "mean_percent_z_ge_3",
            "sem_percent_z_ge_threshold": "sem_percent_z_ge_3",
            "ci95_low_percent_z_ge_threshold": "ci95_low_percent_z_ge_3",
            "ci95_high_percent_z_ge_threshold": "ci95_high_percent_z_ge_3",
        }
    )
    return [
        emit(
            table,
            out,
            "figure5_panel_b_hcp_emotion_extent.tsv",
            (
                "smoothing_fwhm_mm",
                "n_subjects",
                "mean_percent_z_ge_3",
                "sem_percent_z_ge_3",
                "ci95_low_percent_z_ge_3",
                "ci95_high_percent_z_ge_3",
                "mean_effect",
            ),
        )
    ]


def at_threshold(table: pd.DataFrame, value: str) -> pd.DataFrame:
    return table.loc[table["exclusion"].map(pseg).eq(value)]


def figure6_participant_tables(
    lags: pd.DataFrame, gradients: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve within-participant pairing while omitting controlled identifiers."""
    primary_lags = at_threshold(lags, "pseg50").copy()
    primary_lags["segment"] = primary_lags["segment"].replace({"striatal": "striate"})
    primary_lags = primary_lags.loc[
        primary_lags["segment"].isin(("striate", "peduncular"))
    ]
    if primary_lags.duplicated(["participant", "segment"]).any():
        raise ValueError("Figure 6 requires one lag per participant and segment")
    paired = primary_lags.pivot(
        index="participant", columns="segment", values="target_minus_amy_ms"
    ).reindex(columns=("striate", "peduncular"))
    if paired.empty or not np.isfinite(paired.to_numpy(dtype=float)).all():
        raise ValueError("Figure 6 requires finite paired striate/peduncular lags")

    slopes = at_threshold(gradients, "pseg50")
    slopes = slopes.loc[
        slopes["segment"].isin(("combined", "striate_peduncular"))
        & np.isfinite(slopes["slope_ms_per_mm"])
    ].copy()
    if slopes.empty or slopes["participant"].duplicated().any():
        raise ValueError("Figure 6 requires one finite combined slope per participant")
    participants = sorted(set(paired.index) | set(slopes["participant"]))
    display_ids = {participant: index + 1 for index, participant in enumerate(participants)}
    paired = paired.rename(
        columns={"striate": "striate_lag_ms", "peduncular": "peduncular_lag_ms"}
    )
    paired.insert(0, "amygdala_lag_ms", 0.0)
    paired.insert(0, "display_id", paired.index.map(display_ids))
    slopes["display_id"] = slopes["participant"].map(display_ids)
    return (
        paired.reset_index(drop=True).sort_values("display_id"),
        slopes[["display_id", "slope_ms_per_mm"]].sort_values("display_id"),
    )


def export_lags(root: Path, out: Path) -> list[Path]:
    paired, slopes = figure6_participant_tables(
        read(
            root / "participant_lags.tsv",
            ("participant", "segment", "exclusion", "target_minus_amy_ms"),
        ),
        read(
            root / "participant_gradients.tsv",
            ("participant", "segment", "exclusion", "slope_ms_per_mm"),
        ),
    )
    reported = read(
        root / "reported_timing_summary.tsv",
        (
            "exclusion",
            "median_striate_minus_amygdala_ms",
            "median_peduncular_minus_amygdala_ms",
            "pct_striate_later_than_amygdala",
            "pct_peduncular_later_than_amygdala",
            "pct_peduncular_later_than_striate",
            "combined_slope_mean_ms_per_mm",
            "combined_slope_median_ms_per_mm",
            "pct_positive_combined_slopes",
            "n_participants_combined_slope",
        ),
    )
    primary = unique_row(at_threshold(reported, "pseg50"), "primary timing")
    ordering = read(
        root / "segment_ordering_summary.tsv", ("exclusion", "n_participants")
    )
    ordering = unique_row(at_threshold(ordering, "pseg50"), "primary ordering")
    values = (
        (
            "median_lag_striatal_minus_amygdala_ms",
            primary["median_striate_minus_amygdala_ms"],
        ),
        (
            "median_lag_peduncular_minus_amygdala_ms",
            primary["median_peduncular_minus_amygdala_ms"],
        ),
        (
            "fraction_striatal_after_amygdala",
            primary["pct_striate_later_than_amygdala"] / 100,
        ),
        (
            "fraction_peduncular_after_amygdala",
            primary["pct_peduncular_later_than_amygdala"] / 100,
        ),
        (
            "fraction_peduncular_after_striatal",
            primary["pct_peduncular_later_than_striate"] / 100,
        ),
        ("combined_slope_mean_ms_per_mm", primary["combined_slope_mean_ms_per_mm"]),
        ("combined_slope_median_ms_per_mm", primary["combined_slope_median_ms_per_mm"]),
        ("fraction_positive_slopes", primary["pct_positive_combined_slopes"] / 100),
        ("n_subjects_ordering", ordering["n_participants"]),
        ("n_subjects_slopes", primary["n_participants_combined_slope"]),
    )
    outputs = [
        emit(
            pd.DataFrame(values, columns=("measure", "value")),
            out,
            "figure6_panel_bc_group_summary.tsv",
        )
    ]

    outputs.extend(
        [
            emit(paired, out, "figure6_panel_b_participant_ordering.tsv"),
            emit(slopes, out, "figure6_panel_c_participant_slopes.tsv"),
        ]
    )

    lags = read(
        root / "lag_summary.tsv",
        (
            "target_key",
            "segment",
            "exclusion",
            "median_target_minus_amy_ms",
            "pct_target_later_than_amy",
        ),
    ).copy()
    lags["code"] = lags["exclusion"].map(pseg)
    lags = lags.loc[
        lags["code"].isin(("pseg50", "pseg20", "pseg05"))
        & lags["segment"].isin(("striate", "striatal", "peduncular"))
    ]
    if len(lags) != 6:
        raise ValueError(f"Figure 6D requires six rows, found {len(lags)}")
    lags["segment"] = lags["segment"].replace({"striatal": "striate"})
    lags["mask_definition"] = lags["code"].map(
        {"pseg50": "pAmy >= 0.50", "pseg20": "pAmy >= 0.20", "pseg05": "pAmy >= 0.05"}
    )
    lags["target_key"] = lags["target_key"].map(canonical_target_key)
    baseline = (
        lags.loc[lags["code"].eq("pseg50")]
        .set_index("segment")["median_target_minus_amy_ms"]
        .to_dict()
    )
    if set(baseline) != {"striate", "peduncular"}:
        raise ValueError("Figure 6D is missing a pAmy >= 0.50 segment baseline")
    lags["change_from_pAmy_ge_0p50_ms"] = lags.apply(
        lambda row: row["median_target_minus_amy_ms"] - baseline[row["segment"]],
        axis=1,
    )
    lags["segment"] = lags["segment"].map(
        {
            "striate": "Striate BVR",
            "striatal": "Striate BVR",
            "peduncular": "Peduncular BVR",
        }
    )
    lags = lags.rename(
        columns={
            "median_target_minus_amy_ms": "median_lag_ms",
            "pct_target_later_than_amy": "pct_later_than_amygdala",
        }
    )
    order = {"pseg50": 0, "pseg20": 1, "pseg05": 2}
    lags["order"] = lags["code"].map(order)
    lags["segment_order"] = lags["segment"].map({"Striate BVR": 0, "Peduncular BVR": 1})
    lags = lags.sort_values(["order", "segment_order"])
    outputs.append(
        emit(
            lags,
            out,
            "figure6_panel_d_amygdala_probability_sensitivity.tsv",
            (
                "mask_definition",
                "segment",
                "target_key",
                "median_lag_ms",
                "pct_later_than_amygdala",
                "change_from_pAmy_ge_0p50_ms",
            ),
        )
    )
    return outputs


def select_primary(table: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in set(table["target_key"]):
            raise ValueError(f"Unknown primary target: {requested}")
        return requested
    selected = table.loc[
        table["segment"]
        .astype(str)
        .str.lower()
        .isin(("combined", "striate_peduncular"))
        & table["exclusion"].map(pseg).eq("pseg50")
    ]
    keys = selected["target_key"].drop_duplicates().tolist()
    if len(keys) != 1:
        raise ValueError("Pass --primary-target-key to identify the Figure 7 target")
    return str(keys[0])


def published_targets(root: Path) -> dict[str, dict[str, object]]:
    """Map analysis target keys to the compact manuscript table metadata."""
    table = read(
        root / "target_inventory.tsv",
        ("target_key", "segment", "exclusion", "bootstrap_seed_offset"),
    )
    result = {}
    for row in table.itertuples(index=False):
        segment = (
            "combined"
            if str(row.segment).lower() in {"combined", "striate_peduncular"}
            else str(row.segment).lower()
        )
        key = canonical_target_key(row.target_key)
        code = pseg(row.exclusion)
        label = (
            "Combined primary"
            if segment == "combined" and code == "pseg50"
            else "Combined strict"
            if segment == "combined"
            else str(row.segment)
        )
        result[str(row.target_key)] = {
            "key": key,
            "label": label,
            "segment": segment,
            "threshold": f"{str(row.exclusion).removesuffix(' excluded')} excluded",
            "bootstrap_seed_offset": int(row.bootstrap_seed_offset),
        }
    return result


def combined_targets(
    metadata: dict[str, dict[str, object]], available: set[str]
) -> list[str]:
    """Return the primary and strict combined targets in manuscript order."""
    selected = [
        raw
        for raw, values in metadata.items()
        if raw in available
        and values["segment"] == "combined"
        and values["label"] in {"Combined primary", "Combined strict"}
    ]
    selected.sort(key=lambda raw: metadata[raw]["label"] != "Combined primary")
    if len(selected) != 2:
        raise ValueError("Figure 7 requires primary and strict combined targets")
    return selected


def canonical_inventory(
    table: pd.DataFrame, metadata: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """Translate the analysis inventory to its manuscript-facing schema."""
    table = table.copy()
    unknown = set(table["target_key"]).difference(metadata)
    if unknown:
        raise ValueError(f"Inventory contains unknown targets: {sorted(unknown)}")
    table["target_roi"] = table["target_key"].map(lambda raw: metadata[raw]["key"])
    table = table.rename(
        columns={
            "candidate_key": "source_roi",
            "candidate_label": "source_label",
            "overlap_removed": "overlap_with_target_voxels",
            "retained_voxels": "prepared_voxels",
        }
    )
    table["source_label"] = table.apply(canonical_source_label, axis=1)
    table["family"] = table["source_roi"].map(canonical_family)
    columns = (
        "target_roi",
        "target_label",
        "hemisphere",
        "source_roi",
        "source_label",
        "source_short_label",
        "source_long_label",
        "source_cortical_division",
        "family",
        "gray_matter_support_source",
        "target_voxels",
        "raw_voxels",
        "pre_support_voxels",
        "gray_matter_supported_voxels",
        "gray_matter_excluded_voxels",
        "overlap_with_target_voxels",
        "prepared_voxels",
        "status",
        "source",
    )
    missing = set(columns).difference(table.columns)
    if missing:
        raise ValueError(f"Candidate inventory is missing columns: {sorted(missing)}")
    return table.loc[:, columns]


def canonical_family(key: object) -> str:
    key = str(key)
    if key == "amy_proper":
        return "cit168"
    if key.startswith("glasser__"):
        return "glasser"
    if key.startswith("tian_s2__"):
        return "tian_s2"
    return "composite"


def canonical_source_label(row: pd.Series) -> str:
    def text(value: object) -> str:
        return "" if pd.isna(value) else str(value).strip()

    key = str(row.get("source_roi", row.get("candidate_key", "")))
    if key == "amy_proper":
        return "Amy proper"
    if key.startswith("glasser__"):
        short = text(row.get("source_short_label", ""))
        long = text(row.get("source_long_label", ""))
        division = text(row.get("source_cortical_division", ""))
        detail = f"{short} - {long}" if long else short
        return f"Glasser: {detail} ({division})" if division else f"Glasser: {detail}"
    if key.startswith("tian_s2__"):
        label = text(row.get("source_label", row.get("candidate_label", "")))
        return label if label.startswith("Tian S2: ") else f"Tian S2: {label}"
    return text(row.get("source_label", row.get("candidate_label", key)))


def export_figure7(root: Path, out: Path, requested_target: str | None) -> list[Path]:
    require_profile_statistic(root, "median")
    correlations = read(
        root / "source_profile_correlations.tsv",
        (
            "target_key",
            "target_label",
            "segment",
            "exclusion",
            "candidate_key",
            "ranked",
            "rank",
            "group_r",
            "bootstrap_q025",
            "bootstrap_q975",
            "n_conditions",
            "n_tasks",
            *SOURCE_SUPPORT_COLUMNS,
        ),
    )
    target = select_primary(correlations, requested_target)
    target_metadata = published_targets(root)
    targets = combined_targets(
        target_metadata, set(correlations["target_key"].astype(str))
    )
    rows = []
    for target_key in targets:
        selected = correlations.loc[
            correlations["target_key"].eq(target_key)
        ].set_index("candidate_key")
        for raw, displayed, label, _fingerprint in CANDIDATES:
            row = unique_row(
                selected.loc[selected.index == raw], f"candidate {target_key}/{raw}"
            )
            rows.append(
                {
                    **target_metadata[target_key],
                    "candidate_key": displayed,
                    "candidate_label": label,
                    "route": CANDIDATE_ROUTES[raw],
                    "role": CANDIDATE_ROLES[raw],
                    "main_figure": True,
                    "group_r": row["group_r"],
                    "bootstrap_median": row["bootstrap_median"],
                    "bootstrap_q025": row["bootstrap_q025"],
                    "bootstrap_q975": row["bootstrap_q975"],
                    **{column: row[column] for column in SOURCE_SUPPORT_COLUMNS},
                }
            )
    outputs = [emit(pd.DataFrame(rows), out, "figure7_candidate_correlations.tsv")]

    emotion_all = read(
        root / "emotion_exclusion_sensitivity.tsv",
        (
            "target_key",
            "scenario",
            "candidate_key",
            "ranked",
            "group_r",
            "rank",
            *SOURCE_SUPPORT_COLUMNS,
            "n_conditions",
            "n_tasks",
        ),
    )
    atlas_rows = []
    screens = (
        ("all_tasks", correlations.loc[correlations["target_key"].eq(target)]),
        ("exclude_emotion", emotion_all),
        (
            "all_tasks",
            correlations.loc[correlations["target_key"].eq(targets[1])],
        ),
    )
    for scenario, table in screens:
        table = table.loc[table["ranked"].astype(str).str.lower().eq("true")]
        for row in table.itertuples(index=False):
            atlas_rows.append(
                {
                    **target_metadata[row.target_key],
                    "scenario": scenario,
                    "candidate_column": row.candidate_key,
                    "group_r": row.group_r,
                    "n_conditions": int(row.n_conditions),
                    "n_tasks": int(row.n_tasks),
                    "rank": int(row.rank),
                    **{column: getattr(row, column) for column in SOURCE_SUPPORT_COLUMNS},
                }
            )
    outputs.append(
        emit(
            pd.DataFrame(atlas_rows),
            out,
            "figure7_atlas_correlations.tsv",
        )
    )

    profiles = read(
        root / "group_condition_profiles.tsv",
        ("target_key", "profile_key", "task", "condition_key", "standardized_beta"),
    )
    rows = []
    for target_key in targets:
        target_profiles = profiles.loc[profiles["target_key"].eq(target_key)]
        target_correlations = correlations.loc[
            correlations["target_key"].eq(target_key)
        ].set_index("candidate_key")
        specs = [(target_key, "trace", "Peri-amygdalar target")]
        specs += [
            (raw, displayed, fingerprint)
            for raw, displayed, _label, fingerprint in CANDIDATES
        ]
        for raw, displayed, label in specs:
            frame = target_profiles.loc[target_profiles["profile_key"].eq(raw)].copy()
            if set(frame["condition_key"]) != set(CONDITION_LABELS):
                raise ValueError(
                    f"Fingerprint profile is incomplete: {target_key}/{raw}"
                )
            group_r = (
                1.0
                if raw == target_key
                else unique_row(
                    target_correlations.loc[target_correlations.index == raw],
                    f"profile correlation {target_key}/{raw}",
                )["group_r"]
            )
            frame["profile_key"] = displayed
            frame["profile_label"] = label
            frame["condition_label"] = frame["condition_key"].map(CONDITION_LABELS)
            frame["group_r"] = group_r
            for field, value in target_metadata[target_key].items():
                frame[field] = value
            rows.append(frame)
    fingerprint = pd.concat(rows, ignore_index=True)
    outputs.append(
        emit(
            fingerprint,
            out,
            "figure7_group_condition_profiles.tsv",
            (
                "key",
                "label",
                "segment",
                "threshold",
                "bootstrap_seed_offset",
                "profile_key",
                "profile_label",
                "task",
                "condition_key",
                "condition_label",
                "standardized_beta",
                "group_r",
            ),
        )
    )

    commonality = read(
        root / "pairwise_commonality.tsv",
        (
            "target_key",
            "rival_key",
            "r2_reference_alone",
            "r2_rival_alone",
            "r2_both",
            "unique_reference",
            "unique_rival",
            "unique_reference_bootstrap_median",
            "unique_reference_q025",
            "unique_reference_q975",
            "unique_rival_bootstrap_median",
            "unique_rival_q025",
            "unique_rival_q975",
        ),
    )
    rows = []
    for target_key in targets:
        selected = commonality.loc[commonality["target_key"].eq(target_key)].set_index(
            "rival_key"
        )
        for raw, displayed, label, _fingerprint in CANDIDATES[1:]:
            row = unique_row(
                selected.loc[selected.index == raw],
                f"commonality {target_key}/{raw}",
            )
            rows.append(
                {
                    **target_metadata[target_key],
                    "candidate_key": displayed,
                    "candidate_label": label,
                    "members": CANDIDATE_MEMBERS[raw],
                    "main_figure": True,
                    "r2_amy_alone": row["r2_reference_alone"],
                    "r2_rival_alone": row["r2_rival_alone"],
                    "r2_both": row["r2_both"],
                    "unique_amy": row["unique_reference"],
                    "unique_rival": row["unique_rival"],
                    "unique_amy_bootstrap_median": row[
                        "unique_reference_bootstrap_median"
                    ],
                    "unique_amy_q025": row["unique_reference_q025"],
                    "unique_amy_q975": row["unique_reference_q975"],
                    "unique_rival_bootstrap_median": row[
                        "unique_rival_bootstrap_median"
                    ],
                    "unique_rival_q025": row["unique_rival_q025"],
                    "unique_rival_q975": row["unique_rival_q975"],
                }
            )
    outputs.append(emit(pd.DataFrame(rows), out, "figure7_pairwise_commonality.tsv"))

    spread = read(
        root / "modeled_blur.tsv",
        (
            "shell",
            "observed_amygdala_profile_amplitude",
            "maximum_modeled_blur",
        ),
    ).rename(
        columns={
            "n_voxels": "n_vox",
            "observed_amygdala_profile_amplitude": "observed",
            "predicted_2mm": "pred_2mm",
            "predicted_3mm": "pred_3mm",
            "predicted_4mm": "pred_4mm",
            "predicted_5mm": "pred_5mm",
            "predicted_6mm": "pred_6mm",
            "predicted_8mm": "pred_8mm",
            "maximum_modeled_blur": "modeled",
        }
    )
    outputs.append(
        emit(
            spread,
            out,
            "figure7_modeled_blur.tsv",
            (
                "shell",
                "n_vox",
                "observed",
                "pred_2mm",
                "pred_3mm",
                "pred_4mm",
                "pred_5mm",
                "pred_6mm",
                "pred_8mm",
                "modeled",
            ),
        )
    )

    paired = read(
        root / "paired_candidate_differences.tsv",
        (
            "target_key",
            "reference_key",
            "rival_key",
            "reference_r",
            "rival_r",
            "r_difference",
            "bootstrap_median_difference",
            "bootstrap_q025",
            "bootstrap_q975",
            "probability_reference_greater",
            "n_cohort",
            "n_bootstrap",
        ),
    )
    displayed_rivals = {
        raw: (displayed, label)
        for raw, displayed, label, _fingerprint in CANDIDATES[1:]
    }
    paired = paired.loc[
        paired["reference_key"].eq("amy_proper")
        & paired["rival_key"].isin(displayed_rivals)
    ]
    paired_rows = []
    for row in paired.itertuples(index=False):
        rival_key, rival_label = displayed_rivals[row.rival_key]
        paired_rows.append(
            {
                **target_metadata[row.target_key],
                "reference_key": "amy",
                "reference_label": "Amygdala",
                "rival_key": rival_key,
                "rival_label": rival_label,
                "interval_scope": "candidate-wise, unadjusted",
                "reference_r": row.reference_r,
                "rival_r": row.rival_r,
                "r_difference": row.r_difference,
                "paired_difference_median": row.bootstrap_median_difference,
                "paired_difference_q025": row.bootstrap_q025,
                "paired_difference_q975": row.bootstrap_q975,
                "paired_interval_excludes_zero": (
                    row.bootstrap_q025 > 0 or row.bootstrap_q975 < 0
                ),
                "probability_reference_greater": row.probability_reference_greater,
                "n_cohort": int(row.n_cohort),
                "n_bootstrap": int(row.n_bootstrap),
            }
        )
    outputs.append(
        emit(
            pd.DataFrame(paired_rows),
            out,
            "figure7_paired_candidate_differences.tsv",
        )
    )

    emotion = emotion_all.loc[emotion_all["target_key"].eq(target)].set_index(
        "candidate_key"
    )
    amy = unique_row(emotion.loc[emotion.index == "amy_proper"], "Emotion amygdala")
    pir = unique_row(emotion.loc[emotion.index == "glasser__pir"], "Emotion piriform")
    emotion_paired = read(
        root / "emotion_exclusion_paired_candidate_differences.tsv",
        (
            "target_key",
            "reference_key",
            "rival_key",
            "reference_r",
            "rival_r",
            "r_difference",
            "bootstrap_median_difference",
            "bootstrap_q025",
            "bootstrap_q975",
            "probability_reference_greater",
            "n_cohort",
            "n_bootstrap",
        ),
    )
    comparison = unique_row(
        emotion_paired.loc[
            emotion_paired["target_key"].eq(target)
            & emotion_paired["reference_key"].eq("amy_proper")
            & emotion_paired["rival_key"].eq("glasser__pir")
        ],
        "Emotion amygdala-piriform comparison",
    )
    outputs.append(
        emit(
            pd.DataFrame(
                [
                    {
                        **target_metadata[target],
                        "scenario": amy["scenario"],
                        "reference_column": "amy_proper",
                        "reference_rank": int(amy["rank"]),
                        "rival_column": "glasser__pir",
                        "rival_rank": int(pir["rank"]),
                        "interval_scope": "candidate-wise, unadjusted",
                        "reference_r": comparison["reference_r"],
                        "rival_r": comparison["rival_r"],
                        "r_difference": comparison["r_difference"],
                        "paired_difference_median": comparison[
                            "bootstrap_median_difference"
                        ],
                        "paired_difference_q025": comparison["bootstrap_q025"],
                        "paired_difference_q975": comparison["bootstrap_q975"],
                        "paired_interval_excludes_zero": (
                            comparison["bootstrap_q025"] > 0
                            or comparison["bootstrap_q975"] < 0
                        ),
                        "probability_reference_greater": comparison[
                            "probability_reference_greater"
                        ],
                        "n_conditions": int(amy["n_conditions"]),
                        "n_tasks": int(amy["n_tasks"]),
                        "n_cohort": int(comparison["n_cohort"]),
                        "n_bootstrap": int(comparison["n_bootstrap"]),
                    }
                ]
            ),
            out,
            "figure7_emotion_exclusion_sensitivity.tsv",
        )
    )

    outputs.append(
        emit(
            read(
                root / "held_out_task_prediction.tsv",
                ("scenario", "candidate_key", "q2_global_mean", "n_cohort", "rank"),
            ),
            out,
            "figure7_held_out_task_prediction.tsv",
        )
    )
    return outputs


def export_mean_sensitivity(root: Path, out: Path) -> list[Path]:
    require_profile_statistic(root, "mean")
    table = read(
        root / "source_profile_correlations.tsv",
        (
            "target_key", "segment", "exclusion", "candidate_key", "group_r",
            "bootstrap_q025", "bootstrap_q975", *SOURCE_SUPPORT_COLUMNS,
        ),
    )
    primary = select_primary(table, None)
    table = table.loc[table["target_key"].eq(primary)].copy()
    table["voxel_statistic"] = "mean"
    return [
        emit(
            table, out, "figure7_voxelwise_mean_sensitivity.tsv",
            (
                "target_key", "target_label", "segment", "exclusion",
                "candidate_key", "candidate_label", "group_r",
                "bootstrap_q025", "bootstrap_q975", *SOURCE_SUPPORT_COLUMNS,
                "n_conditions", "n_tasks", "voxel_statistic", "voxel_statistic_role",
            ),
        )
    ]


def export_s5(root: Path, out: Path) -> list[Path]:
    require_profile_statistic(root, "median")
    grid = read(
        root / "segment_exclusion_grid.tsv",
        ("segment", "exclusion", "r_amy", "r_amy_lo", "r_amy_hi", "rank_amy", "r_pir"),
    ).copy()
    grid["exclusion"] = grid["exclusion"].map(pseg)
    grid["segment"] = grid["segment"].replace(
        {"striate": "striatal", "striate_peduncular": "combined"}
    )
    grid = grid.loc[
        grid["segment"].isin(("combined", "striatal", "peduncular"))
        & grid["exclusion"].isin(("pseg50", "pseg20", "pseg05"))
    ]
    if len(grid) != 9:
        raise ValueError(f"Figure S5 requires nine rows, found {len(grid)}")
    outputs = [
        emit(
            grid,
            out,
            "figureS5_segment_grid.tsv",
            (
                "segment",
                "exclusion",
                "n_vox",
                "r_amy",
                "r_amy_lo",
                "r_amy_hi",
                "rank_amy",
                "r_pir",
                "p_amy_first",
                "p_pir_first",
            ),
        )
    ]

    metadata = published_targets(root)
    allowed = {
        raw
        for raw, values in metadata.items()
        if values["segment"] in {"combined", "striate", "striatal", "peduncular"}
        and (
            str(values["key"]).endswith("pseg50")
            or str(values["key"]).endswith("pseg05")
        )
    }
    inventory = read(
        root / "candidate_inventory.tsv",
        (
            "target_key",
            "target_label",
            "candidate_key",
            "candidate_label",
            "hemisphere",
            "target_voxels",
            "raw_voxels",
            "pre_support_voxels",
            "gray_matter_supported_voxels",
            "gray_matter_excluded_voxels",
            "overlap_removed",
            "retained_voxels",
        ),
    )
    inventory = inventory.loc[inventory["target_key"].isin(allowed)]
    expected_rows = len(allowed) * 195 * 2
    if len(allowed) != 6 or len(inventory) != expected_rows:
        raise ValueError(
            "Figure 7 mask inventory requires six targets and "
            f"{expected_rows} rows; found {len(allowed)} targets and {len(inventory)} rows"
        )
    outputs.append(
        emit(
            canonical_inventory(inventory, metadata),
            out,
            "figure7_candidate_mask_inventory.tsv",
        )
    )
    return outputs


def export_s4_controls(root: Path, out: Path) -> list[Path]:
    require_profile_statistic(root, "median")
    controls = read(
        root / "figureS4_pairwise_positive_controls.tsv",
        (
            "display_order",
            "target_label",
            "source_label",
            "unique_amy_beyond_rival",
            "unique_rival_beyond_amy",
        ),
    )
    return [emit(controls, out, "figureS4_pairwise_positive_controls.tsv")]


def export_s4_rankings(
    root: Path, out: Path, requested_target: str | None
) -> list[Path]:
    require_profile_statistic(root, "median")
    correlations = read(
        root / "source_profile_correlations.tsv",
        (
            "target_key",
            "segment",
            "exclusion",
            "candidate_key",
            "candidate_label",
            "source_short_label",
            "source_long_label",
            "source_cortical_division",
            "family",
            "ranked",
            "rank",
            "group_r",
            "bootstrap_median",
            "bootstrap_q025",
            "bootstrap_q975",
            "participant_median_r",
            "participant_q25_r",
            "participant_q75_r",
            *SOURCE_SUPPORT_COLUMNS,
        ),
    )
    if requested_target is None:
        likely = correlations.loc[
            correlations["segment"].astype(str).str.lower().eq("mesencephalic")
            & correlations["exclusion"]
            .astype(str)
            .str.contains("tha[- ]?dp", case=False, regex=True)
        ]
        keys = likely["target_key"].drop_duplicates().tolist()
        if len(keys) != 1:
            raise ValueError("Pass --s4-target-key to identify the S4 target")
        requested_target = str(keys[0])
    selected = correlations.loc[correlations["target_key"].eq(requested_target)].copy()
    if selected["ranked"].dtype == bool:
        selected = selected.loc[selected["ranked"]]
    else:
        selected = selected.loc[selected["ranked"].astype(str).str.lower().eq("true")]
    if selected.empty:
        raise ValueError(f"No ranked candidates for S4 target: {requested_target}")
    inventory = read(
        root / "candidate_inventory.tsv",
        (
            "target_key",
            "candidate_key",
            "candidate_label",
            "family",
            "hemisphere",
            "target_voxels",
            "pre_support_voxels",
            "gray_matter_supported_voxels",
            "gray_matter_excluded_voxels",
            "overlap_removed",
            "retained_voxels",
        ),
    )
    counts = (
        inventory.loc[inventory["target_key"].eq(requested_target)]
        .groupby("candidate_key", as_index=False)
        .agg(
            pre_support_voxels=("pre_support_voxels", "sum"),
            gray_matter_supported_voxels=("gray_matter_supported_voxels", "sum"),
            gray_matter_excluded_voxels=("gray_matter_excluded_voxels", "sum"),
            prepared_voxels=("retained_voxels", "sum"),
            overlap_with_target_voxels=("overlap_removed", "sum"),
        )
    )
    selected = selected.drop(
        columns=[
            "pre_support_voxels",
            "gray_matter_supported_voxels",
            "gray_matter_excluded_voxels",
            "prepared_voxels",
            "overlap_with_target_voxels",
        ],
        errors="ignore",
    )
    selected = selected.merge(
        counts,
        on="candidate_key",
        how="left",
        validate="one_to_one",
    )
    if (
        selected[
            [
                "pre_support_voxels",
                "gray_matter_supported_voxels",
                "gray_matter_excluded_voxels",
                "prepared_voxels",
                "overlap_with_target_voxels",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("S4 candidate voxel counts are incomplete")
    selected["rank"] = selected["rank"].astype(int)
    selected["source_roi"] = selected["candidate_key"]
    selected["source_label"] = selected["candidate_label"]
    selected = selected.rename(
        columns={
            "group_r": "group_profile_r",
            "bootstrap_median": "bootstrap_median_r",
            "bootstrap_q025": "bootstrap_q025_r",
            "bootstrap_q975": "bootstrap_q975_r",
            "participant_median_r": "subject_median_r",
            "participant_q25_r": "subject_q25_r",
            "participant_q75_r": "subject_q75_r",
        }
    ).sort_values("rank")
    selected["source_label"] = selected.apply(canonical_source_label, axis=1)
    selected["family"] = selected["source_roi"].map(canonical_family)
    outputs = [
        emit(
            selected,
            out,
            "figureS4_mesencephalic_source_rankings.tsv",
            (
                "rank",
                "source_roi",
                "source_label",
                "source_short_label",
                "source_long_label",
                "source_cortical_division",
                "family",
                "group_profile_r",
                "bootstrap_median_r",
                "bootstrap_q025_r",
                "bootstrap_q975_r",
                "subject_median_r",
                "subject_q25_r",
                "subject_q75_r",
                *SOURCE_SUPPORT_COLUMNS,
                "pre_support_voxels",
                "gray_matter_supported_voxels",
                "gray_matter_excluded_voxels",
                "prepared_voxels",
                "overlap_with_target_voxels",
            ),
        )
    ]
    outputs.append(
        emit(
            canonical_inventory(inventory, published_targets(root)),
            out,
            "figureS4_candidate_mask_inventory.tsv",
        )
    )
    return outputs


def export_overlap(path: Path, out: Path) -> list[Path]:
    table = read(
        path,
        (
            "atlas_label",
            "pre_exclusion_trace_overlap_voxels",
            "pre_exclusion_trace_overlap_percent",
        ),
    )
    return [emit(table, out, "figureS7_atlas_trace_overlap.tsv")]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--roi-dir", type=Path)
    result.add_argument("--pstc-dir", type=Path)
    result.add_argument("--spatial-dir", type=Path)
    result.add_argument("--smoothing-dir", type=Path)
    result.add_argument("--rapidtide-dir", type=Path)
    result.add_argument("--figure7-source-profiles-dir", type=Path)
    result.add_argument("--mean-source-profiles-dir", type=Path)
    result.add_argument("--s5-source-profiles-dir", type=Path)
    result.add_argument("--s4-source-profiles-dir", type=Path)
    result.add_argument("--s4-positive-controls-dir", type=Path)
    result.add_argument("--atlas-overlap", type=Path)
    result.add_argument("--figure1-amygdala-roi", default="amygdala")
    result.add_argument("--pstc-amygdala-roi", default="cit168_amygdala")
    result.add_argument("--pstc-striate-roi", default="striate_bvr")
    result.add_argument("--pstc-peduncular-roi", default="peduncular_bvr")
    result.add_argument("--smoothing-z-threshold", type=float, default=3.0)
    result.add_argument("--primary-target-key")
    result.add_argument("--s4-target-key")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    inputs = (
        args.roi_dir,
        args.pstc_dir,
        args.spatial_dir,
        args.smoothing_dir,
        args.rapidtide_dir,
        args.figure7_source_profiles_dir,
        args.mean_source_profiles_dir,
        args.s5_source_profiles_dir,
        args.s4_source_profiles_dir,
        args.s4_positive_controls_dir,
        args.atlas_overlap,
    )
    if not any(value is not None for value in inputs):
        raise ValueError("Provide at least one analysis-stage input")
    outputs: list[Path] = []
    if args.roi_dir:
        outputs += export_roi(args.roi_dir, args.output_dir, args.figure1_amygdala_roi)
    if args.pstc_dir:
        outputs += export_pstc(args.pstc_dir, args.output_dir, args)
    if args.spatial_dir:
        outputs += export_spatial(args.spatial_dir, args.output_dir)
    if args.smoothing_dir:
        outputs += export_smoothing(
            args.smoothing_dir, args.output_dir, args.smoothing_z_threshold
        )
    if args.rapidtide_dir:
        outputs += export_lags(args.rapidtide_dir, args.output_dir)
    if args.figure7_source_profiles_dir:
        outputs += export_figure7(
            args.figure7_source_profiles_dir,
            args.output_dir,
            args.primary_target_key,
        )
    if args.s5_source_profiles_dir:
        outputs += export_s5(args.s5_source_profiles_dir, args.output_dir)
    if args.mean_source_profiles_dir:
        outputs += export_mean_sensitivity(args.mean_source_profiles_dir, args.output_dir)
    if args.s4_source_profiles_dir:
        outputs += export_s4_rankings(
            args.s4_source_profiles_dir, args.output_dir, args.s4_target_key
        )
    if args.s4_positive_controls_dir:
        outputs += export_s4_controls(
            args.s4_positive_controls_dir,
            args.output_dir,
        )
    if args.atlas_overlap:
        outputs += export_overlap(args.atlas_overlap, args.output_dir)
    print(f"Exported {len(outputs)} figure tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
