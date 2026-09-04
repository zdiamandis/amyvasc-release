"""Helpers shared by the figure renderers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AMYGDALA = "#2CA25F"
STRIATE = "#D95F5F"
PEDUNCULAR = "#756BB1"
TARGET = "#3B75AF"
ALTERNATIVE = "#E28E2C"
NEUTRAL = "#777777"


def read_table(data_dir: Path, name: str, required: set[str]) -> pd.DataFrame:
    """Read a tab-separated source-data table and validate its schema."""

    data_dir = Path(data_dir)
    candidates = (data_dir / "tables" / name, data_dir / name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find {name}; searched {searched}")

    table = pd.read_csv(path, sep="\t")
    missing = required.difference(table.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {columns}")
    return table


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    """Save a renderer's PNG and vector PDF outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply shared axis styling."""

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E6E6E6", linewidth=0.7)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def clean_label(value: object) -> str:
    """Shorten labels for compact, readable axes."""
    return str(value).replace("†", "").strip()


def as_bool(series: pd.Series) -> pd.Series:
    """Normalize booleans read from TSV files."""

    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def interval_bounds(lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """Return validated lower and upper confidence-interval endpoints."""

    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if lower_values.shape != upper_values.shape:
        raise ValueError("Confidence-interval endpoints have different shapes")
    if not np.isfinite(lower_values).all() or not np.isfinite(upper_values).all():
        raise ValueError("Confidence-interval endpoints must be finite")
    if np.any(lower_values > upper_values):
        raise ValueError("Confidence-interval lower endpoint exceeds upper endpoint")
    return lower_values, upper_values
