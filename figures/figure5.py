"""Reproduce the quantitative panel of main Figure 5."""

from pathlib import Path

import matplotlib.pyplot as plt

from ._common import TARGET, panel_label, read_table, save_figure, style_axis


def render(data_dir: Path, output_dir: Path) -> list[Path]:
    """Render the HCP-YA smoothing summary in Figure 5B."""

    table = read_table(
        data_dir,
        "figure5_panel_b_hcp_emotion_extent.tsv",
        {
            "smoothing_fwhm_mm",
            "mean_percent_z_ge_3",
            "ci95_low_percent_z_ge_3",
            "ci95_high_percent_z_ge_3",
            "mean_effect",
        },
    ).sort_values("smoothing_fwhm_mm")

    lower = table["mean_percent_z_ge_3"] - table["ci95_low_percent_z_ge_3"]
    upper = table["ci95_high_percent_z_ge_3"] - table["mean_percent_z_ge_3"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.errorbar(
        table["smoothing_fwhm_mm"],
        table["mean_percent_z_ge_3"],
        yerr=[lower, upper],
        marker="o",
        capsize=3,
        color=TARGET,
        linewidth=2,
    )
    beta_min = table["mean_effect"].min()
    beta_max = table["mean_effect"].max()
    ax.text(
        0.03,
        0.96,
        f"Mean beta remained {beta_min:.3f}–{beta_max:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.set(
        xlabel="Spatial smoothing (mm FWHM)",
        ylabel="Amygdala voxels with z ≥ 3 (%)",
        title="Smoothing expands apparent anatomical-amygdala extent",
    )
    panel_label(ax, "B")
    style_axis(ax)
    return save_figure(fig, output_dir, "figure5")
