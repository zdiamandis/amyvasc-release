"""Reproduce the four quantitative panels of main Figure 7."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    ALTERNATIVE,
    AMYGDALA,
    TARGET,
    as_bool,
    clean_label,
    interval_bounds,
    panel_label,
    read_table,
    save_figure,
    style_axis,
)

PROFILE_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "trace",
            "amy",
            "pir",
            "ofc_47m",
            "anterior_rhinal",
            "ahip",
            "posterior_insula",
            "ffc",
        )
    )
}

CONDITION_ORDER = {
    key: index
    for index, key in enumerate(
        (
            "emotion__face",
            "emotion__shape",
            "wm__0bk_faces",
            "wm__0bk_places",
            "wm__0bk_tools",
            "wm__0bk_body",
            "wm__2bk_faces",
            "wm__2bk_places",
            "wm__2bk_tools",
            "wm__2bk_body",
            "social__mental",
            "social__random",
            "language__story",
            "language__math",
            "gambling__win",
            "gambling__loss",
            "relational__relational",
            "relational__matching",
            "motor__left_hand",
            "motor__right_hand",
            "motor__left_foot",
            "motor__right_foot",
            "motor__tongue",
        )
    )
}

CONDITION_LABELS = {
    "emotion__face": "fear",
    "emotion__shape": "shape",
    "wm__0bk_faces": "0-back face",
    "wm__0bk_places": "0-back place",
    "wm__0bk_tools": "0-back tool",
    "wm__0bk_body": "0-back body",
    "wm__2bk_faces": "2-back face",
    "wm__2bk_places": "2-back place",
    "wm__2bk_tools": "2-back tool",
    "wm__2bk_body": "2-back body",
    "social__mental": "mental",
    "social__random": "random",
    "language__story": "story",
    "language__math": "math",
    "gambling__win": "win",
    "gambling__loss": "loss",
    "relational__relational": "relational",
    "relational__matching": "matching",
    "motor__left_hand": "left hand",
    "motor__right_hand": "right hand",
    "motor__left_foot": "left foot",
    "motor__right_foot": "right foot",
    "motor__tongue": "tongue",
}

PROFILE_LABELS = {
    "trace": "Peri-amygdalar target",
    "amy": "Amygdala",
    "amy_proper": "Amygdala",
    "pir": "Piriform cortex (Pir)",
    "glasser__pir": "Piriform cortex (Pir)",
    "ofc_47m": "Orbitofrontal 47m",
    "glasser__47m": "Orbitofrontal 47m",
}


def _main_rows(table):
    if "label" in table.columns:
        primary = table["label"].astype(str).str.lower().eq("combined primary")
        if primary.any():
            table = table.loc[primary]
    elif "threshold" in table.columns:
        primary = table["threshold"].astype(str).str.contains("0.50", regex=False)
        if primary.any():
            table = table.loc[primary]
    if "main_figure" in table.columns:
        table = table.loc[as_bool(table["main_figure"])]
    if "candidate_order" in table.columns:
        table = table.sort_values("candidate_order")
    return table


def _plot_correlations(ax: plt.Axes, table) -> None:
    table = _main_rows(table).copy()
    table["candidate_label"] = table["candidate_label"].map(clean_label)
    y = np.arange(len(table))[::-1]
    lower, upper = interval_bounds(table["bootstrap_q025"], table["bootstrap_q975"])
    colors = [AMYGDALA if key == "amy" else TARGET for key in table["candidate_key"]]
    ax.hlines(y, lower, upper, color="#777777", linewidth=1)
    ax.scatter(table["group_r"], y, color=colors, s=35, zorder=3)
    ax.set_yticks(y, table["candidate_label"])
    ax.set(xlabel="Correlation with peri-amygdalar target", xlim=(0, 1))
    panel_label(ax, "A")
    style_axis(ax, grid_axis="x")


def _plot_fingerprint(ax: plt.Axes, table) -> None:
    table = _main_rows(table).copy()
    if "row_order" not in table:
        table["row_order"] = table["profile_key"].map(PROFILE_ORDER)
    if "condition_order" not in table:
        table["condition_order"] = table["condition_key"].map(CONDITION_ORDER)
    if table[["row_order", "condition_order"]].isna().any().any():
        raise ValueError("Figure 7 fingerprint has unknown profile or condition keys")
    table["profile_label"] = table.apply(
        lambda row: PROFILE_LABELS.get(row["profile_key"], row["profile_label"]),
        axis=1,
    )
    table["condition_label"] = table["condition_key"].map(CONDITION_LABELS)
    table = table.sort_values(["row_order", "condition_order"])
    matrix = table.pivot(
        index=["row_order", "profile_label"],
        columns="condition_order",
        values="standardized_beta",
    ).sort_index()
    image = ax.imshow(
        matrix.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto"
    )
    ax.set_yticks(
        np.arange(len(matrix)),
        [clean_label(label) for _, label in matrix.index],
        fontsize=7,
    )
    condition_labels = (
        table.sort_values("condition_order")
        .drop_duplicates("condition_order")["condition_label"]
        .astype(str)
    )
    ax.set_xticks(
        np.arange(len(condition_labels)),
        [label.replace("\n", " ") for label in condition_labels],
        rotation=90,
        fontsize=6,
    )
    ax.set_xlabel("HCP-YA task condition")
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        label="Standardized beta",
        orientation="horizontal",
        fraction=0.06,
        pad=0.18,
    )
    colorbar.ax.tick_params(labelsize=7)
    panel_label(ax, "B")


def _plot_commonality(ax: plt.Axes, table) -> None:
    table = table.copy()
    if "threshold" in table.columns:
        primary = table["threshold"].astype(str).str.contains("0.50", regex=False)
        if primary.any():
            table = table.loc[primary]
    if "main_figure" in table.columns:
        table = table.loc[as_bool(table["main_figure"])]
    table["shared"] = table["r2_both"] - table["unique_amy"] - table["unique_rival"]
    labels = [clean_label(label) for label in table["candidate_label"]]
    x = np.arange(len(table))
    shared = table["shared"].clip(lower=0)
    amy = table["unique_amy"].clip(lower=0)
    rival = table["unique_rival"].clip(lower=0)
    ax.bar(x, shared, color="#BDBDBD", label="Shared")
    ax.bar(x, amy, bottom=shared, color=AMYGDALA, label="Unique to amygdala")
    ax.bar(
        x,
        rival,
        bottom=shared + amy,
        color=ALTERNATIVE,
        label="Unique to candidate",
    )
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Explained variance ($R^2$)")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "C")
    style_axis(ax)


def _plot_point_spread(ax: plt.Axes, table) -> None:
    x = np.arange(len(table))
    ax.bar(
        x - 0.19,
        table["observed"],
        0.36,
        color=AMYGDALA,
        label="measured in BVR trace",
    )
    ax.bar(
        x + 0.19,
        table["modeled"],
        0.36,
        color="#777777",
        label="Maximum modeled\npartial-volume blur",
    )
    for xi, value in zip(x, table["modeled"], strict=True):
        ax.text(xi + 0.02, value + 0.06, f"{value:.2f}", fontsize=8, color="0.35")
    ax.axhline(1, color="0.6", linewidth=0.7, linestyle=":")
    ax.set_xticks(x, table["shell"])
    ax.set(
        xlabel="Distance from anatomical amygdala",
        ylabel="Amygdala-patterned amplitude\n(anatomical amygdala = 1)",
        title="Peri-amygdalar signal vs. modeled blur",
        ylim=(0, max(4.15, float(table["observed"].max()) * 1.15)),
    )
    ax.legend(frameon=False, fontsize=8.625, loc="upper right")
    panel_label(ax, "D")
    style_axis(ax)


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    correlations = read_table(
        data_dir,
        "figure7_candidate_correlations.tsv",
        {
            "candidate_key",
            "candidate_label",
            "group_r",
            "bootstrap_q025",
            "bootstrap_q975",
        },
    )
    fingerprint = read_table(
        data_dir,
        "figure7_group_condition_profiles.tsv",
        {
            "profile_key",
            "profile_label",
            "condition_key",
            "condition_label",
            "standardized_beta",
        },
    )
    commonality = read_table(
        data_dir,
        "figure7_pairwise_commonality.tsv",
        {
            "candidate_label",
            "r2_both",
            "unique_amy",
            "unique_rival",
        },
    )
    point_spread = read_table(
        data_dir,
        "figure7_modeled_blur.tsv",
        {"shell", "observed", "modeled"},
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    _plot_correlations(axes[0, 0], correlations)
    _plot_fingerprint(axes[0, 1], fingerprint)
    _plot_commonality(axes[1, 0], commonality)
    _plot_point_spread(axes[1, 1], point_spread)
    return save_figure(fig, output_dir, "figure7")
