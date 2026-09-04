"""Reproduce the quantitative panel of Supplementary Figure S5."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    ALTERNATIVE,
    AMYGDALA,
    interval_bounds,
    panel_label,
    read_table,
    save_figure,
    style_axis,
)


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render amygdala-profile robustness across target definitions."""

    table = read_table(
        data_dir,
        "figureS5_segment_grid.tsv",
        {
            "segment",
            "exclusion",
            "r_amy",
            "r_amy_lo",
            "r_amy_hi",
            "rank_amy",
            "r_pir",
        },
    )
    exclusions = ["pseg50", "pseg20", "pseg05"]
    exclusion_labels = ["Exclude ≥0.50", "Exclude ≥0.20", "Exclude ≥0.05"]
    segments = ["combined", "striatal", "peduncular"]

    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.8), sharey=True, constrained_layout=True
    )
    for index, (ax, segment) in enumerate(zip(axes, segments, strict=True)):
        group = (
            table.loc[table["segment"] == segment]
            .set_index("exclusion")
            .reindex(exclusions)
        )
        x = np.arange(len(group))
        lower, upper = interval_bounds(group["r_amy_lo"], group["r_amy_hi"])
        ax.vlines(x, lower, upper, color=AMYGDALA, linewidth=1)
        ax.plot(
            x,
            group["r_amy"],
            color=AMYGDALA,
            marker="o",
            linewidth=2,
            label="Amygdala",
        )
        ax.plot(
            x,
            group["r_pir"],
            color=ALTERNATIVE,
            marker="o",
            linewidth=1.5,
            label="Piriform cortex",
        )
        for x_value, row in zip(x, group.itertuples(), strict=True):
            ax.annotate(
                f"rank {int(row.rank_amy)}",
                (x_value, row.r_amy),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        ax.set_xticks(x, exclusion_labels, rotation=30, ha="right")
        ax.set_title(segment.replace("striatal", "striate").title())
        if index == 0:
            ax.set_ylabel("Task-profile correlation")
            ax.legend(frameon=False, fontsize=7)
            panel_label(ax, "B")
        style_axis(ax)
    return save_figure(fig, output_dir, "figureS5")
