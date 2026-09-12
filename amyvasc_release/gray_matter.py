"""Build the gray-matter support used by the source-profile analyses."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from .masks import image_like, resample_to_reference, same_grid

GM_PROBABILITY_THRESHOLD = 0.50
FAST_OPTIONS = (
    "-t",
    "1",
    "-n",
    "3",
    "-I",
    "4",
    "-l",
    "20",
    "-f",
    "0.02",
    "-W",
    "15",
    "-R",
    "0.3",
    "-O",
    "4",
    "-H",
    "0.1",
)


def _fast_gray_matter_probability(
    template_path: Path, fast_bin: str
) -> nib.Nifti1Image:
    executable = shutil.which(fast_bin)
    if executable is None:
        raise FileNotFoundError(f"FSL FAST executable not found: {fast_bin}")
    with tempfile.TemporaryDirectory(prefix="amyvasc_fast_") as temporary:
        prefix = Path(temporary) / "mni152nlin6asym_2mm"
        subprocess.run(
            [executable, *FAST_OPTIONS, "-o", str(prefix), str(template_path)],
            check=True,
            env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"},
        )
        path = prefix.with_name(f"{prefix.name}_pve_1.nii.gz")
        if not path.is_file():
            raise FileNotFoundError(f"FAST did not create its gray-matter PVE: {path}")
        image = nib.load(str(path))
        return nib.Nifti1Image(
            image.get_fdata(dtype=np.float32),
            image.affine,
            image.header.copy(),
        )


def build_gray_matter_support(
    *,
    reference_path: Path,
    template_path: Path,
    hcp_subcortical_path: Path,
    output_path: Path,
    fast_bin: str = "fast",
) -> tuple[Path, Path]:
    """Write FAST pGM>=0.50 union HCP subcortical support and provenance.

    The brain-extracted MNI152NLin6Asym template must already share the
    fixed-effect grid. The HCP segmentation is resampled with nearest-neighbor
    interpolation because its distributed orientation differs from that grid.
    """
    reference = nib.load(str(reference_path))
    template = nib.load(str(template_path))
    if reference.ndim < 3 or template.ndim != 3:
        raise ValueError("Reference and template must define three-dimensional grids")
    if not same_grid(template, reference):
        raise ValueError(
            "The brain-extracted MNI template must share the fixed-effect grid"
        )

    probability = _fast_gray_matter_probability(template_path, fast_bin)
    if not same_grid(probability, reference):
        raise ValueError("FAST gray-matter output does not share the fixed-effect grid")
    fast_support = probability.get_fdata(dtype=np.float32) >= GM_PROBABILITY_THRESHOLD

    hcp_source = nib.load(str(hcp_subcortical_path))
    if hcp_source.ndim != 3:
        raise ValueError(f"Expected a 3D HCP segmentation: {hcp_subcortical_path}")
    hcp_on_reference = resample_to_reference(
        hcp_source,
        reference,
        interpolation="nearest",
    )
    hcp_support = np.asarray(hcp_on_reference.dataobj) > 0
    combined = fast_support | hcp_support
    if not np.any(combined):
        raise ValueError("Gray-matter support is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image_like(combined, reference, dtype=np.uint8), output_path)
    stem = output_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    provenance_path = output_path.with_name(f"{stem}_provenance.json")
    overlap = fast_support & hcp_support
    provenance = {
        "definition": (
            "FSL FAST gray-matter partial-volume estimate >= 0.50 on the "
            "brain-extracted 2-mm MNI152NLin6Asym template, union nonzero "
            "labels from the standard HCP subcortical segmentation"
        ),
        "reference": str(reference_path),
        "template": str(template_path),
        "hcp_subcortical": str(hcp_subcortical_path),
        "fast_command": [
            fast_bin,
            *FAST_OPTIONS,
            "-o",
            "<temporary-prefix>",
            str(template_path),
        ],
        "fast_probability_threshold": GM_PROBABILITY_THRESHOLD,
        "fast_output_type": "NIFTI_GZ",
        "hcp_resampling": "nearest neighbor to the fixed-effect grid",
        "voxel_counts": {
            "fast_pgm50": int(fast_support.sum()),
            "hcp_subcortical": int(hcp_support.sum()),
            "overlap": int(overlap.sum()),
            "fast_only": int((fast_support & ~hcp_support).sum()),
            "hcp_only": int((hcp_support & ~fast_support).sum()),
            "union": int(combined.sum()),
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, provenance_path
