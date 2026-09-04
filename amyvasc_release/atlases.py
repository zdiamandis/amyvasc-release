"""Prepare the cortical and subcortical candidates used for source attribution."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot

from .masks import image_like, resample_to_reference, same_grid


def _annot_names(path: Path, hemisphere: str) -> list[str]:
    """Read the 180 parcel names from one HCP-MMP1 annotation."""
    _labels, _colors, raw_names = read_annot(str(path))
    names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_names
    ]
    if len(names) != 181 or names[0] != "???":
        raise ValueError(f"Expected unknown + 180 parcel labels in {path}")
    prefix = f"{hemisphere}_"
    cleaned = []
    for name in names[1:]:
        if not name.startswith(prefix) or not name.endswith("_ROI"):
            raise ValueError(f"Unexpected HCP-MMP1 label {name!r} in {path}")
        cleaned.append(name.removeprefix(prefix).removesuffix("_ROI"))
    if len(cleaned) != 180 or len(set(cleaned)) != 180:
        raise ValueError(f"HCP-MMP1 labels are incomplete or duplicated in {path}")
    return cleaned


def hcp_mmp1_label_names(lh_annot: Path, rh_annot: Path) -> dict[int, str]:
    """Map the 360 one-based probability volumes to bilateral area names."""
    mapping: dict[int, str] = {}
    for path, hemisphere, offset in (
        (lh_annot, "L", 0),
        (rh_annot, "R", 180),
    ):
        for index, name in enumerate(_annot_names(path, hemisphere), start=1):
            mapping[offset + index] = name
    return mapping


def _resample_4d_trilinear(
    source_path: Path,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path,
    *,
    flirt_bin: str,
) -> nib.Nifti1Image:
    """Apply the manuscript's FLIRT trilinear resampling to a 4D atlas."""
    source = nib.load(str(source_path))
    if source.ndim != 4 or source.shape[3] != 360:
        raise ValueError(f"Expected a 360-volume HCP-MMP1 atlas: {source_path}")
    if same_grid(source, reference):
        return source
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="amyvasc_hcp_mmp1_") as temporary:
        reference_path = Path(temporary) / "reference.nii.gz"
        reference_3d = image_like(
            np.zeros(reference.shape[:3], dtype=np.float32),
            reference,
            dtype=np.float32,
        )
        nib.save(reference_3d, reference_path)
        subprocess.run(
            [
                flirt_bin,
                "-in",
                str(source_path),
                "-ref",
                str(reference_path),
                "-out",
                str(output_path),
                "-applyxfm",
                "-usesqform",
                "-interp",
                "trilinear",
                "-datatype",
                "float",
            ],
            check=True,
        )
    return nib.load(str(output_path))


def hcp_mmp1_argmax(
    probability_path: Path,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path,
    *,
    minimum_probability: float = 0.20,
    flirt_bin: str = "flirt",
) -> nib.Nifti1Image:
    """Create the one-based p20 maximum-probability HCP-MMP1 image."""
    if not 0 <= minimum_probability <= 1:
        raise ValueError("minimum_probability must be between zero and one")
    temporary_output = output_path.with_name(
        f".{output_path.name}.probabilities.nii.gz"
    )
    probability = _resample_4d_trilinear(
        probability_path,
        reference,
        temporary_output,
        flirt_bin=flirt_bin,
    )
    maximum = np.zeros(reference.shape[:3], dtype=np.float32)
    labels = np.zeros(reference.shape[:3], dtype=np.int16)
    for index in range(360):
        values = np.clip(
            np.asarray(probability.dataobj[..., index], dtype=np.float32), 0, 1
        )
        update = values > maximum
        maximum[update] = values[update]
        labels[update] = index + 1
    labels[maximum < minimum_probability] = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = image_like(labels, reference, dtype=np.int16)
    nib.save(result, output_path)
    if temporary_output.exists():
        temporary_output.unlink()
    return result


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _crosswalk(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    table = pd.read_csv(path, sep="\t", comment="#", keep_default_na=False)
    if not {"label", "long_label"}.issubset(table.columns):
        raise ValueError(f"HCP-MMP1 crosswalk lacks label/long_label: {path}")
    division_column = (
        "cortical_division" if "cortical_division" in table.columns else None
    )
    return {
        str(row.label): (
            str(row.long_label),
            "" if division_column is None else str(getattr(row, division_column)),
        )
        for row in table.itertuples(index=False)
        if str(row.label)
    }


def hcp_mmp1_candidates(
    label_path: Path,
    names: dict[int, str],
    *,
    crosswalk_path: Path | None = None,
) -> list[dict[str, object]]:
    """Return the 180 bilateral HCP-MMP1 candidate definitions."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for value, name in names.items():
        grouped[name].append(value)
    crosswalk = _crosswalk(crosswalk_path)
    rows = []
    for name in sorted(grouped):
        values = sorted(grouped[name])
        if len(values) != 2:
            raise ValueError(f"Expected two hemispheres for HCP-MMP1 area {name}")
        long_name, division = crosswalk.get(name, ("", ""))
        detail = f"{name} - {long_name}" if long_name else name
        label = f"Glasser: {detail} ({division})" if division else detail
        rows.append(
            {
                "key": f"glasser__{_slug(name)}",
                "label": label,
                "path": label_path.name,
                "values": ";".join(str(value) for value in values),
                "family": "HCP-MMP1",
                "members": "",
                "short_label": name,
                "long_label": long_name,
                "cortical_division": division,
            }
        )
    if len(rows) != 180:
        raise ValueError(
            f"Expected 180 bilateral HCP-MMP1 candidates, found {len(rows)}"
        )
    return rows


def tian_candidates(
    atlas_path: Path,
    labels_path: Path,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path,
) -> list[dict[str, object]]:
    """Resample Tian S2 and return its 14 bilateral non-amygdala candidates."""
    labels = [
        line.strip() for line in labels_path.read_text().splitlines() if line.strip()
    ]
    if len(labels) != 32:
        raise ValueError(f"Expected 32 Tian S2 labels in {labels_path}")
    source = nib.load(str(atlas_path))
    resampled = resample_to_reference(source, reference, interpolation="nearest")
    values = np.rint(resampled.get_fdata(dtype=np.float32)).astype(np.int16)
    if values.min() < 0 or values.max() > 32:
        raise ValueError(f"Unexpected Tian S2 values after resampling: {atlas_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image_like(values, reference, dtype=np.int16), output_path)

    grouped: dict[str, list[int]] = defaultdict(list)
    for value, raw_label in enumerate(labels, start=1):
        name = re.sub(r"-(?:lh|rh)$", "", raw_label, flags=re.IGNORECASE)
        if "amy" not in name.lower():
            grouped[name].append(value)
    rows = []
    for name in sorted(grouped):
        bilateral_values = sorted(grouped[name])
        if len(bilateral_values) != 2:
            raise ValueError(f"Expected two Tian hemispheres for {name}")
        rows.append(
            {
                "key": f"tian_s2__{_slug(name)}",
                "label": f"Tian S2: {name}",
                "path": output_path.name,
                "values": ";".join(str(value) for value in bilateral_values),
                "family": "Tian S2",
                "members": "",
                "short_label": "",
                "long_label": "",
                "cortical_division": "",
            }
        )
    if len(rows) != 14:
        raise ValueError(
            f"Expected 14 bilateral non-amygdala Tian candidates, found {len(rows)}"
        )
    return rows


def prepare_source_candidates(
    *,
    reference_path: Path,
    hcp_mmp1_probability: Path,
    lh_annot: Path,
    rh_annot: Path,
    tian_atlas: Path,
    tian_labels: Path,
    output_dir: Path,
    crosswalk_path: Path | None = None,
    minimum_probability: float = 0.20,
    flirt_bin: str = "flirt",
) -> Path:
    """Write atlas images, 194 ranked candidates, and displayed composites."""
    reference = nib.load(str(reference_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    hcp_labels_path = output_dir / "hcp_mmp1_p20_labels.nii.gz"
    hcp_mmp1_argmax(
        hcp_mmp1_probability,
        reference,
        hcp_labels_path,
        minimum_probability=minimum_probability,
        flirt_bin=flirt_bin,
    )
    rows = hcp_mmp1_candidates(
        hcp_labels_path,
        hcp_mmp1_label_names(lh_annot, rh_annot),
        crosswalk_path=crosswalk_path,
    )
    tian_path = output_dir / "tian_s2_labels.nii.gz"
    rows.extend(tian_candidates(tian_atlas, tian_labels, reference, tian_path))
    rows.extend(
        [
            {
                "key": "anterior_rhinal",
                "label": "Rhinal cortex",
                "path": "",
                "values": "",
                "family": "HCP-MMP1 composite",
                "members": "glasser__ec;glasser__peec",
                "ranked": False,
                "short_label": "",
                "long_label": "",
                "cortical_division": "",
            },
            {
                "key": "posterior_insula",
                "label": "Posterior insula",
                "path": "",
                "values": "",
                "family": "HCP-MMP1 composite",
                "members": "glasser__ig;glasser__pi;glasser__poi1;glasser__poi2",
                "ranked": False,
                "short_label": "",
                "long_label": "",
                "cortical_division": "",
            },
        ]
    )
    table_path = output_dir / "source_candidates.tsv"
    pd.DataFrame(rows).to_csv(table_path, sep="\t", index=False)
    return table_path
