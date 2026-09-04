"""Reproduce the quantitative panels of main Figure 4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import panel_label, read_table, save_figure, style_axis


def _plot_pstc(ax: plt.Axes, table) -> None:
    for _, group in table.groupby("roi", sort=False):
        group = group.sort_values("relative_time_sec")
        ax.plot(
            group["relative_time_sec"],
            group["mean_psc"],
            color=group["color"].iloc[0],
            linewidth=2,
            label=group["roi_label"].iloc[0],
        )
    ax.axhline(0, color="#666666", linewidth=0.7)
    ax.axvline(0, color="#AAAAAA", linewidth=0.7, linestyle="--")
    ax.set(xlabel="Time from block onset (s)", ylabel="Mean signal change (%)")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "B")
    style_axis(ax)


def _plot_trimming(ax: plt.Axes, table) -> None:
    colors = {"lateral": "#3B75AF", "medial": "#C44E52"}
    labels = {
        "lateral": "Trim from lateral side",
        "medial": "Trim from medial side",
    }
    for direction, group in table.groupby("direction", sort=False):
        group = group.sort_values("depth_mm")
        color = colors.get(direction, "#777777")
        ax.plot(
            group["depth_mm"],
            group["mean_beta"],
            marker="o",
            color=color,
            label=labels.get(direction, direction),
        )
        ax.fill_between(
            group["depth_mm"],
            group["ci95_low_beta"],
            group["ci95_high_beta"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.set(xlabel="Trim depth (mm)", ylabel="Task-mean beta")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "C")
    style_axis(ax)


def _plot_voxels(ax: plt.Axes, observations, summary) -> None:
    color_by_key = summary.set_index("condition_key")["color"].to_dict()
    for key, group in observations.groupby("condition_key", sort=False):
        row = summary.loc[summary["condition_key"] == key].iloc[0]
        color = color_by_key.get(key, "#777777")
        ax.scatter(
            group["distance_to_ros_mm"],
            group["group_median_effect"],
            s=8,
            alpha=0.20,
            color=color,
            linewidth=0,
        )
        x_line = np.array(
            [group["distance_to_ros_mm"].min(), group["distance_to_ros_mm"].max()]
        )
        y_line = row["intercept"] + row["slope_beta_per_mm"] * x_line
        ax.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2,
            label=f"{row['condition_label']} ({row['slope_beta_per_mm']:.3f}/mm)",
        )
    ax.set(
        xlabel="Distance from striate-BVR mask (mm)",
        ylabel="Group-median voxel beta",
    )
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "D")
    style_axis(ax)


def _plot_subnuclei(ax: plt.Axes, table) -> None:
    group_colors = {
        "centromedial": "#D95F5F",
        "basolateral": "#3B75AF",
        "other": "#777777",
    }
    for group_name, group in table.groupby("group", sort=False):
        ax.scatter(
            group["median_distance_mm"],
            group["task_mean_effect"],
            s=np.maximum(group["n_voxels_in_amygdala"], 5) * 2,
            color=group_colors.get(group_name, "#777777"),
            alpha=0.82,
            label=group_name.title(),
        )
    for row in table.itertuples(index=False):
        ax.annotate(
            row.label,
            (row.median_distance_mm, row.task_mean_effect),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set(
        xlabel="Median distance from striate-BVR mask (mm)",
        ylabel="Task-mean beta",
    )
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "E")
    style_axis(ax)


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render panels B--E; panel A is the manuscript's anatomical key."""

    pstc = read_table(
        data_dir,
        "figure4_panel_b_task_mean_pstc.tsv",
        {"roi", "roi_label", "color", "relative_time_sec", "mean_psc"},
    )
    trimming = read_table(
        data_dir,
        "figure4_panel_c_directional_trimming.tsv",
        {
            "direction",
            "depth_mm",
            "mean_beta",
            "ci95_low_beta",
            "ci95_high_beta",
        },
    )
    observations = read_table(
        data_dir,
        "figure4_panel_d_voxel_observations.tsv",
        {"condition_key", "distance_to_ros_mm", "group_median_effect"},
    )
    gradients = read_table(
        data_dir,
        "figure4_panel_d_voxel_gradient_summary.tsv",
        {
            "condition_key",
            "condition_label",
            "color",
            "slope_beta_per_mm",
            "intercept",
        },
    )
    subnuclei = read_table(
        data_dir,
        "figure4_panel_e_subnucleus_summary.tsv",
        {
            "label",
            "group",
            "median_distance_mm",
            "n_voxels_in_amygdala",
            "task_mean_effect",
        },
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    _plot_pstc(axes[0, 0], pstc)
    _plot_trimming(axes[0, 1], trimming)
    _plot_voxels(axes[1, 0], observations, gradients)
    _plot_subnuclei(axes[1, 1], subnuclei)
    return save_figure(fig, output_dir, "figure4")
