"""Render group-level summaries for main Figure 6B-D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    PEDUNCULAR,
    STRIATE,
    panel_label,
    read_table,
    save_figure,
    style_axis,
)


def _summary_dict(table) -> dict[str, float]:
    return dict(zip(table["measure"], table["value"], strict=True))


def _plot_lag_summary(ax: plt.Axes, values: dict[str, float]) -> None:
    heights = [
        values["median_lag_striatal_minus_amygdala_ms"],
        values["median_lag_peduncular_minus_amygdala_ms"],
    ]
    fractions = [
        values["fraction_striatal_after_amygdala"],
        values["fraction_peduncular_after_amygdala"],
    ]
    x = np.arange(2)
    ax.bar(x, heights, color=[STRIATE, PEDUNCULAR], width=0.65)
    ax.set_xticks(x, ["Striate BVR", "Peduncular BVR"])
    for index, (height, fraction) in enumerate(zip(heights, fractions, strict=True)):
        ax.text(
            index,
            height,
            f"{height:.0f} ms\n{fraction:.1%} later",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Median lag relative to amygdala (ms)")
    panel_label(ax, "B")
    style_axis(ax)


def _plot_slope_summary(ax: plt.Axes, values: dict[str, float]) -> None:
    labels = ["Mean", "Median"]
    slopes = [
        values["combined_slope_mean_ms_per_mm"],
        values["combined_slope_median_ms_per_mm"],
    ]
    ax.bar(labels, slopes, color=["#4C78A8", "#9ECAE1"], width=0.6)
    ax.axhline(0, color="#555555", linewidth=0.8)
    for index, slope in enumerate(slopes):
        ax.text(index, slope, f"{slope:.1f}", ha="center", va="bottom", fontsize=8)
    fraction = values.get("fraction_positive_slopes")
    if fraction is not None:
        ax.text(
            0.98,
            0.96,
            f"Positive in {fraction:.1%} of participants",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    ax.set_ylabel("Combined BVR lag gradient (ms/mm)")
    panel_label(ax, "C")
    style_axis(ax)


def _plot_sensitivity(ax: plt.Axes, table) -> None:
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
    ax.set_xticks(x, ["Exclude ≥0.50", "Exclude ≥0.20", "Exclude ≥0.05"])
    ax.set(xlabel="CIT168 amygdala probability", ylabel="Median lag (ms)")
    ax.legend(frameon=False, fontsize=8)
    panel_label(ax, "D")
    style_axis(ax)


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render the reported group-level lag and sensitivity summaries."""

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
    values = _summary_dict(summary)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    _plot_lag_summary(axes[0], values)
    _plot_slope_summary(axes[1], values)
    _plot_sensitivity(axes[2], sensitivity)
    return save_figure(fig, output_dir, "figure6")
