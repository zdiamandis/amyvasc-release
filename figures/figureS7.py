"""Reproduce the quantitative panel of Supplementary Figure S7."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import panel_label, read_table, save_figure, style_axis


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render atlas overlap with the pre-exclusion peri-amygdalar trace."""

    table = read_table(
        data_dir,
        "figureS7_atlas_trace_overlap.tsv",
        {
            "atlas_label",
            "pre_exclusion_trace_overlap_voxels",
            "pre_exclusion_trace_overlap_percent",
        },
    )
    x = np.arange(len(table))
    colors = ["#2CA25F", "#4C78A8", "#E3A72F"][: len(table)]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = ax.bar(x, table["pre_exclusion_trace_overlap_percent"], color=colors)
    ax.set_xticks(x, table["atlas_label"], rotation=20, ha="right")
    ax.set_ylabel("Pre-exclusion trace classified as amygdala (%)")
    for bar, row in zip(bars, table.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row.pre_exclusion_trace_overlap_percent:.1f}%\n"
            f"({int(row.pre_exclusion_trace_overlap_voxels)} voxels)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    panel_label(ax, "B")
    style_axis(ax)
    return save_figure(fig, output_dir, "figureS7")
