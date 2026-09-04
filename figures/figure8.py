"""Reproduce the quantitative panel of main Figure 8."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    TARGET,
    as_bool,
    panel_label,
    read_table,
    save_figure,
    style_axis,
)

X_COLUMN = "anatomical_amygdala_mean_canonical_beta_pct_signal"
Y_COLUMN = "peri_amygdalar_target_mean_canonical_beta_pct_signal"


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render the 23-condition sensitivity/specificity scatter in Figure 8A."""

    table = read_table(
        data_dir,
        "figure8_source_data.tsv",
        {
            "condition_label",
            X_COLUMN,
            Y_COLUMN,
            "included_in_emotion_excluded_correlation",
            "annotated",
        },
    )
    x = table[X_COLUMN].to_numpy()
    y = table[Y_COLUMN].to_numpy()
    annotated = as_bool(table["annotated"])

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    ax.scatter(x[~annotated], y[~annotated], color=TARGET, alpha=0.78, s=35)
    ax.scatter(
        x[annotated],
        y[annotated],
        color="#C44E52",
        edgecolor="black",
        marker="D",
        s=65,
        zorder=4,
    )

    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(x.min(), x.max(), 100)
    ax.plot(line_x, intercept + slope * line_x, color="#333333", linewidth=1.3)

    for row in table.loc[annotated].itertuples(index=False):
        ax.annotate(
            row.condition_label,
            (getattr(row, X_COLUMN), getattr(row, Y_COLUMN)),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
        )

    all_r = np.corrcoef(x, y)[0, 1]
    subset = as_bool(table["included_in_emotion_excluded_correlation"])
    subset_r = np.corrcoef(x[subset], y[subset])[0, 1]
    ax.text(
        0.03,
        0.97,
        f"All conditions: r = {all_r:.2f}\nEmotion excluded: r = {subset_r:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.axhline(0, color="#AAAAAA", linewidth=0.7)
    ax.axvline(0, color="#AAAAAA", linewidth=0.7)
    ax.set(
        xlabel="Anatomical-amygdala mean canonical beta (% signal change)",
        ylabel="Peri-amygdalar-target mean canonical beta (% signal change)",
    )
    panel_label(ax, "A")
    style_axis(ax)
    return save_figure(fig, output_dir, "figure8")
