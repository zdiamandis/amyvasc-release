"""Reproduce the quantitative panel of main Figure 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import panel_label, read_table, save_figure, style_axis


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render Figure 1A from its group summary."""

    table = read_table(
        data_dir,
        "figure1_panel_a_summary.tsv",
        {"plot_index", "x_label", "color", "group_mean", "group_sem"},
    ).sort_values("plot_index")

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    ax.bar(
        x,
        table["group_mean"],
        yerr=table["group_sem"],
        color=table["color"],
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x, table["x_label"], rotation=45, ha="right")
    ax.set_ylabel("Mean canonical beta (% signal change)")
    ax.set_title("Anatomical-amygdala response across HCP-YA task conditions")
    panel_label(ax, "A")
    style_axis(ax)
    return save_figure(fig, output_dir, "figure1")
