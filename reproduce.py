#!/usr/bin/env python3
"""Render quantitative manuscript panels from locally generated figure tables."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "outputs" / "source_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

FIGURE_TABLES = {
    "1": ("figure1_panel_a_summary.tsv",),
    "4": (
        "figure4_panel_b_task_mean_pstc.tsv",
        "figure4_panel_c_directional_trimming.tsv",
        "figure4_panel_d_voxel_observations.tsv",
        "figure4_panel_d_voxel_gradient_summary.tsv",
        "figure4_panel_e_subnucleus_summary.tsv",
    ),
    "5": ("figure5_panel_b_hcp_emotion_extent.tsv",),
    "6": (
        "figure6_panel_bc_group_summary.tsv",
        "figure6_panel_d_amygdala_probability_sensitivity.tsv",
    ),
    "7": (
        "figure7_candidate_correlations.tsv",
        "figure7_group_condition_profiles.tsv",
        "figure7_pairwise_commonality.tsv",
        "figure7_modeled_blur.tsv",
    ),
    "8": ("figure8_source_data.tsv",),
    "S4": (
        "figureS4_pairwise_positive_controls.tsv",
        "figureS4_mesencephalic_source_rankings.tsv",
    ),
    "S5": ("figureS5_segment_grid.tsv",),
    "S7": ("figureS7_atlas_trace_overlap.tsv",),
}
FIGURES = tuple(FIGURE_TABLES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "figures",
        nargs="+",
        help=f"Figure IDs ({', '.join(FIGURES)}) or 'all'",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = list(FIGURES) if args.figures == ["all"] else args.figures
    unknown = sorted(set(requested).difference(FIGURES))
    if unknown:
        raise SystemExit(f"Unknown figure ID(s): {', '.join(unknown)}")

    table_dir = args.data_dir / "tables"
    required = {name for figure_id in requested for name in FIGURE_TABLES[figure_id]}
    missing = sorted(name for name in required if not (table_dir / name).is_file())
    if missing:
        preview = "\n".join(f"  - {name}" for name in missing)
        raise SystemExit(
            "Required figure tables have not been generated:\n"
            f"{preview}\n"
            "Run analysis/10_export_figure_tables.py after completing the "
            "analysis stages, or pass --data-dir to an existing export."
        )

    outputs: list[Path] = []
    for figure_id in requested:
        module = importlib.import_module(f"figures.figure{figure_id}")
        outputs.extend(module.render(args.data_dir, args.output_dir))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
