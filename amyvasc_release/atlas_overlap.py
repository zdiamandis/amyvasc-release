"""Atlas-overlap summary used by Supplementary Figure S7."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import pandas as pd

from .masks import load_mask, require_binary_mask, save_mask


def prepare_harvard_oxford_amygdala_mask(
    atlas_path: Path, labels_path: Path, output_path: Path
) -> Path:
    """Extract both amygdala labels from an FSL 3D maxprob atlas.

    FSL XML indices are zero-based; the corresponding maxprob labels are index + 1.
    Use the distributed thresholded atlas so competition with other structures
    is retained, rather than thresholding amygdala probability alone.
    """
    labels = {
        (label.text or "").strip(): int(label.attrib["index"]) + 1
        for label in ET.parse(labels_path).findall(".//label")
    }
    names = ("Left Amygdala", "Right Amygdala")
    if any(name not in labels for name in names):
        raise ValueError(
            "Harvard-Oxford XML must label Left Amygdala and Right Amygdala"
        )
    image = nib.load(atlas_path)
    data = image.get_fdata()
    if image.ndim != 3 or not np.all(np.isfinite(data)):
        raise ValueError(f"Expected a finite 3D maxprob label atlas: {atlas_path}")
    if np.any(data < 0) or not np.all(data == np.round(data)):
        raise ValueError(f"Expected integer maxprob labels: {atlas_path}")
    mask = np.isin(data, [labels[name] for name in names])
    if not mask.any():
        raise ValueError(f"No Harvard-Oxford amygdala labels found: {atlas_path}")
    return save_mask(mask, image, output_path)


def summarize_atlas_trace_overlap(
    *,
    pre_exclusion_trace_path: Path,
    primary_trace_path: Path,
    cit168_p50_path: Path,
    harvard_oxford_maxprob50_path: Path,
    harvard_oxford_maxprob25_path: Path,
) -> pd.DataFrame:
    """Count atlas overlap with the pre-exclusion and primary trace masks."""
    for path in (
        pre_exclusion_trace_path,
        primary_trace_path,
        cit168_p50_path,
        harvard_oxford_maxprob50_path,
        harvard_oxford_maxprob25_path,
    ):
        require_binary_mask(path)
    reference, pre_exclusion = load_mask(pre_exclusion_trace_path)
    _, primary = load_mask(primary_trace_path, reference=reference)
    atlas_paths: Mapping[str, tuple[str, Path]] = {
        "cit168_p50": ("CIT168 pAmy >= 0.50", cit168_p50_path),
        "ho_maxprob50": (
            "Harvard–Oxford maxprob50",
            harvard_oxford_maxprob50_path,
        ),
        "ho_maxprob25": (
            "Harvard–Oxford maxprob25",
            harvard_oxford_maxprob25_path,
        ),
    }
    atlases = {
        key: load_mask(path, reference=reference)[1]
        for key, (_label, path) in atlas_paths.items()
    }

    if np.any(primary & ~pre_exclusion):
        raise ValueError(
            "The primary trace must be a subset of the pre-exclusion trace"
        )
    expected_primary = pre_exclusion & ~atlases["cit168_p50"]
    if not np.array_equal(primary, expected_primary):
        raise ValueError(
            "The primary trace must equal the pre-exclusion trace after CIT168 "
            "pAmy >= 0.50 exclusion"
        )
    if np.any(atlases["ho_maxprob50"] & ~atlases["ho_maxprob25"]):
        raise ValueError("Harvard–Oxford maxprob50 must be nested within maxprob25")

    pre_count = int(pre_exclusion.sum())
    primary_count = int(primary.sum())
    rows = []
    for key, (label, _path) in atlas_paths.items():
        atlas = atlases[key]
        pre_overlap = int((pre_exclusion & atlas).sum())
        primary_overlap = int((primary & atlas).sum())
        rows.append(
            {
                "atlas_key": key,
                "atlas_label": label,
                "atlas_voxels": int(atlas.sum()),
                "pre_exclusion_trace_voxels": pre_count,
                "pre_exclusion_trace_overlap_voxels": pre_overlap,
                "pre_exclusion_trace_overlap_percent": 100.0 * pre_overlap / pre_count,
                "primary_trace_voxels": primary_count,
                "primary_trace_overlap_voxels": primary_overlap,
                "primary_trace_overlap_percent": 100.0
                * primary_overlap
                / primary_count,
            }
        )
    return pd.DataFrame(rows)
