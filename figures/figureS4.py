"""Reproduce the quantitative panels of Supplementary Figure S4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    ALTERNATIVE,
    AMYGDALA,
    clean_label,
    interval_bounds,
    panel_label,
    read_table,
    save_figure,
    style_axis,
)


def _plot_controls(ax: plt.Axes, table) -> None:
    table = table.sort_values("display_order")
    x = np.arange(len(table))
    width = 0.34
    labels = [
        f"{target}\n(expected: {source})"
        for target, source in zip(
            table["target_label"], table["source_label"], strict=True
        )
    ]
    ax.bar(
        x - width / 2,
        table["unique_amy_beyond_rival"],
        width,
        color=AMYGDALA,
        label="Amygdala beyond expected source",
    )
    ax.bar(
        x + width / 2,
        table["unique_rival_beyond_amy"],
        width,
        color=ALTERNATIVE,
        label="Expected source beyond amygdala",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Unique explained variance ($\Delta R^2$)")
    ax.legend(frameon=False, fontsize=7)
    panel_label(ax, "A")
    style_axis(ax)


def _plot_rankings(ax: plt.Axes, table) -> None:
    selected = table.loc[
        (table["rank"] <= 8) | (table["source_roi"] == "amy_proper")
    ].copy()
    selected = selected.sort_values("group_profile_r")
    labels = []
    for row in selected.itertuples(index=False):
        if row.source_roi == "amy_proper":
            labels.append("Amygdala")
        elif isinstance(row.source_short_label, str) and row.source_short_label:
            labels.append(clean_label(row.source_short_label))
        else:
            labels.append(clean_label(row.source_label))
    y = np.arange(len(selected))
    lower, upper = interval_bounds(
        selected["bootstrap_q025_r"], selected["bootstrap_q975_r"]
    )
    ax.hlines(y, lower, upper, color="#777777", linewidth=1)
    ax.scatter(selected["group_profile_r"], y, color="#777777", s=32, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Correlation with mesencephalic-BVR target")
    panel_label(ax, "C")
    style_axis(ax, grid_axis="x")


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    controls = read_table(
        data_dir,
        "figureS4_pairwise_positive_controls.tsv",
        {
            "display_order",
            "target_label",
            "source_label",
            "unique_amy_beyond_rival",
            "unique_rival_beyond_amy",
        },
    )
    rankings = read_table(
        data_dir,
        "figureS4_mesencephalic_source_rankings.tsv",
        {
            "rank",
            "source_roi",
            "source_label",
            "source_short_label",
            "group_profile_r",
            "bootstrap_q025_r",
            "bootstrap_q975_r",
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _plot_controls(axes[0], controls)
    _plot_rankings(axes[1], rankings)
    return save_figure(fig, output_dir, "figureS4")
