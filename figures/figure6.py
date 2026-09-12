"""Render participant distributions for Figure 6B/C and the 6D lag summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._common import (
    AMYGDALA,
    PEDUNCULAR,
    STRIATE,
    panel_label,
    read_table,
    save_figure,
    style_axis,
)

LAG_COLUMNS = ("amygdala_lag_ms", "striate_lag_ms", "peduncular_lag_ms")


def _validate_participant_tables(
    ordering: pd.DataFrame, slopes: pd.DataFrame, summary: pd.DataFrame
) -> None:
    """Reject incomplete pairs or participant exports that disagree with summaries."""
    for name, table, columns in (
        ("ordering", ordering, LAG_COLUMNS),
        ("slopes", slopes, ("slope_ms_per_mm",)),
    ):
        if table.empty or table["display_id"].isna().any():
            raise ValueError(f"Figure 6 {name} requires identified participant rows")
        if table["display_id"].duplicated().any():
            raise ValueError(f"Figure 6 {name} has duplicate participant rows")
        if not np.isfinite(table[list(columns)].to_numpy(float)).all():
            raise ValueError(f"Figure 6 {name} values must be finite")
    if not ordering["amygdala_lag_ms"].eq(0).all():
        raise ValueError("Figure 6 ordering must use the amygdala as its zero reference")
    if summary["measure"].duplicated().any():
        raise ValueError("Figure 6 summary has duplicate measures")

    striate = ordering["striate_lag_ms"].to_numpy(float)
    peduncular = ordering["peduncular_lag_ms"].to_numpy(float)
    gradient = slopes["slope_ms_per_mm"].to_numpy(float)
    observed = {
        "median_lag_striatal_minus_amygdala_ms": np.median(striate),
        "median_lag_peduncular_minus_amygdala_ms": np.median(peduncular),
        "fraction_striatal_after_amygdala": np.mean(striate > 0),
        "fraction_peduncular_after_amygdala": np.mean(peduncular > 0),
        "fraction_peduncular_after_striatal": np.mean(peduncular > striate),
        "combined_slope_mean_ms_per_mm": np.mean(gradient),
        "combined_slope_median_ms_per_mm": np.median(gradient),
        "fraction_positive_slopes": np.mean(gradient > 0),
        "n_subjects_ordering": len(ordering),
        "n_subjects_slopes": len(slopes),
    }
    reported = summary.set_index("measure")["value"]
    for measure, value in observed.items():
        if measure not in reported or not np.isclose(
            value, reported[measure], rtol=1e-8, atol=1e-8
        ):
            raise ValueError(f"Figure 6 participant data disagree with summary: {measure}")


def _plot_lag_ordering(ax: plt.Axes, ordering: pd.DataFrame) -> None:
    """Join each participant's paired relative lags and overlay group medians."""
    values = ordering[list(LAG_COLUMNS)].to_numpy(float)
    x = np.arange(3)
    for row in values:
        ax.plot(x, row, color="#8F969C", alpha=0.055, linewidth=0.7, zorder=1)

    jitter = np.random.default_rng(20260610).normal(0, 0.035, size=values.shape)
    for index, color in enumerate((AMYGDALA, STRIATE, PEDUNCULAR)):
        ax.scatter(
            x[index] + jitter[:, index],
            values[:, index],
            s=8,
            color=color,
            alpha=0.16 if index else 0.10,
            linewidths=0,
            zorder=2,
        )
    medians = np.median(values, axis=0)
    ax.plot(x, medians, color="#111111", linewidth=2.3, marker="o", zorder=4)
    for index, median in enumerate(medians):
        ax.annotate(
            f"{median:.0f}",
            (x[index], median),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(x, ["Amygdala", "Striate BVR", "Peduncular BVR"])
    ax.set_ylabel("Lag relative to amygdala (ms)")
    ax.set_title("Participant-level ordering", fontsize=10)
    ax.set_xlim(-0.35, 2.35)
    panel_label(ax, "B")
    style_axis(ax, grid_axis="")


def _plot_slope_distribution(ax: plt.Axes, slopes: pd.DataFrame) -> None:
    """Show every finite participant slope, with the group mean in black."""
    values = slopes["slope_ms_per_mm"].to_numpy(float)
    ax.hist(
        values, bins=32, color=PEDUNCULAR, alpha=0.82, edgecolor="white", linewidth=0.45
    )
    ax.axvline(0, color="#777777", linewidth=1)
    ax.axvline(np.mean(values), color="#111111", linewidth=2)
    ax.set_title("Participant-level lag-gradient slopes", fontsize=10)
    ax.set_xlabel("Slope along striate/peduncular BVR (ms/mm)")
    ax.set_ylabel("Participants")
    panel_label(ax, "C")
    style_axis(ax, grid_axis="")


def _plot_sensitivity(ax: plt.Axes, table: pd.DataFrame) -> None:
    order = ["pAmy >= 0.50", "pAmy >= 0.20", "pAmy >= 0.05"]
    x = np.arange(len(order))
    colors = {"Striate BVR": STRIATE, "Peduncular BVR": PEDUNCULAR}
    for segment, group in table.groupby("segment", sort=False):
        indexed = group.set_index("mask_definition").reindex(order)
        ax.plot(
            x,
            indexed["median_lag_ms"],
            marker="o",
            linewidth=2,
            color=colors.get(segment, "#777777"),
            label=segment,
        )
    ax.set_xticks(x, ["≥0.50", "≥0.20", "≥0.05"])
    ax.set(
        title="Boundary sensitivity (quantitative summary)",
        xlabel="Exclude voxels with CIT168 amygdala probability",
        ylabel="Median lag relative to amygdala (ms)",
    )
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "D")
    style_axis(ax)


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render primary participant lags/slopes and boundary-sensitivity medians.

    Panel A and the anatomical thumbnails in the manuscript's panel D require
    separate image assembly. This renderer covers B/C and the numerical values in D.
    """
    ordering = read_table(
        data_dir,
        "figure6_panel_b_participant_ordering.tsv",
        {"display_id", *LAG_COLUMNS},
    ).sort_values("display_id")
    slopes = read_table(
        data_dir,
        "figure6_panel_c_participant_slopes.tsv",
        {"display_id", "slope_ms_per_mm"},
    )
    summary = read_table(
        data_dir,
        "figure6_panel_bc_group_summary.tsv",
        {"measure", "value"},
    )
    sensitivity = read_table(
        data_dir,
        "figure6_panel_d_amygdala_probability_sensitivity.tsv",
        {"mask_definition", "segment", "median_lag_ms"},
    )
    _validate_participant_tables(ordering, slopes, summary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    _plot_lag_ordering(axes[0], ordering)
    _plot_slope_distribution(axes[1], slopes)
    _plot_sensitivity(axes[2], sensitivity)
    return save_figure(fig, output_dir, "figure6")
