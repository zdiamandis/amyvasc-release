"""Mask construction and grid-handling utilities.

The manuscript analyses use CIT168 probabilities on the fMRI grid and a
functionally defined peri-amygdalar BVR label image. This module contains the
operations that define those masks without assuming a local data layout.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

AMYGDALA_LABELS = (
    "AMY_BLN_La",
    "AMY_BLN_BL_BLD+BLI",
    "AMY_BLN_BM",
    "AMY_CEN",
    "AMY_CMN",
    "AMY_BL_BLV",
    "AMY_ATA",
    "AMY_ATA_ASTA",
    "AMY_AAA",
    "AMY",
)

# Odd labels are left hemisphere and even labels are right hemisphere.
BVR_LABELS: Mapping[str, Mapping[str, tuple[int, ...]]] = {
    "striate": {"left": (1,), "right": (2,), "bilateral": (1, 2)},
    "peduncular": {"left": (3,), "right": (4,), "bilateral": (3, 4)},
    "mesencephalic": {"left": (5,), "right": (6,), "bilateral": (5, 6)},
    "striate_peduncular": {
        "left": (1, 3),
        "right": (2, 4),
        "bilateral": (1, 2, 3, 4),
    },
}

# Voxel-index boxes on the HCP-YA 2-mm grid. Within each box, the manuscript
# target contains voxels whose group fear-minus-shape beta is greater than 0.5%.
BVR_GRID_SHAPE = (91, 109, 91)
BVR_GRID_AFFINE = (
    (-2.0, 0.0, 0.0, 90.0),
    (0.0, 2.0, 0.0, -126.0),
    (0.0, 0.0, 2.0, -72.0),
    (0.0, 0.0, 0.0, 1.0),
)
BVR_BOXES: Mapping[int, tuple[int, int, int, int, int, int]] = {
    1: (49, 8, 59, 6, 27, 5),
    2: (32, 8, 59, 6, 27, 5),
    3: (52, 6, 51, 8, 27, 5),
    4: (31, 6, 51, 8, 27, 5),
    5: (47, 7, 43, 7, 30, 9),
    6: (35, 7, 43, 7, 30, 9),
}

S4_MESENCEPHALIC_LABELS = (5, 6)
S4_THADP_PREFIX = "THA-DP-"
S4_CONTROL_COMPONENT_KEYS = (
    "glasser__v1",
    "glasser__v2",
    "glasser__a1",
    "glasser__lbelt",
    "glasser__mbelt",
)


@dataclass(frozen=True)
class S4MesencephalicCounts:
    """Voxel counts defining the Supplementary Figure S4 target."""

    raw_mesencephalic: int
    thadp_overlap: int
    final_target: int


def load_image(path: Path | str, *, ndim: int | None = None) -> nib.Nifti1Image:
    """Load a NIfTI image and optionally require its dimensionality."""
    image = nib.load(str(path))
    if ndim is not None and image.ndim != ndim:
        raise ValueError(f"Expected a {ndim}D image, found {image.shape}: {path}")
    return image


def require_binary_mask(path: Path | str) -> nib.Nifti1Image:
    """Load a nonempty 3D mask whose finite values are exactly zero or one."""

    image = load_image(path, ndim=3)
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Binary mask contains nonfinite values: {path}")
    binary = np.isclose(data, 0.0) | np.isclose(data, 1.0)
    if not np.all(binary):
        raise ValueError(f"Expected a binary 0/1 mask: {path}")
    if not np.any(data > 0.5):
        raise ValueError(f"Binary mask is empty: {path}")
    return image


def same_grid(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    *,
    atol: float = 1e-5,
) -> bool:
    """Return whether two images share a three-dimensional voxel grid."""
    return first.shape[:3] == second.shape[:3] and np.allclose(
        first.affine,
        second.affine,
        atol=atol,
        rtol=0,
    )


def resample_to_reference(
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    *,
    interpolation: str = "continuous",
) -> nib.Nifti1Image:
    """Resample a 3D image to a reference grid.

    ``nearest`` is appropriate for labels and binary masks. ``linear`` and
    ``continuous`` use first- and third-order interpolation, respectively.
    CIT168 probability maps used in the manuscript are instead resampled with
    :func:`resample_probability_sinc`.
    """
    if source.ndim != 3 or reference.ndim < 3:
        raise ValueError("Resampling requires a 3D source and a 3D/4D reference.")
    orders = {"nearest": 0, "linear": 1, "continuous": 3}
    try:
        order = orders[interpolation]
    except KeyError as exc:
        choices = ", ".join(orders)
        raise ValueError(f"interpolation must be one of: {choices}") from exc
    if same_grid(source, reference):
        data = source.get_fdata(dtype=np.float32)
        return image_like(data, reference, dtype=np.float32)
    return resample_from_to(
        source,
        (reference.shape[:3], reference.affine),
        order=order,
    )


def resample_probability_sinc(
    source: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    *,
    flirt_bin: str = "flirt",
) -> nib.Nifti1Image:
    """Resample a probability image with the FLIRT sinc operation used here."""
    if source.ndim != 3:
        raise ValueError(f"Expected a 3D probability image, found {source.shape}")
    if same_grid(source, reference):
        data = np.clip(source.get_fdata(dtype=np.float32), 0.0, 1.0)
        return image_like(data, reference, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="amyvasc_masks_") as temporary:
        directory = Path(temporary)
        source_path = directory / "probability.nii.gz"
        reference_path = directory / "reference.nii.gz"
        output_path = directory / "resampled.nii.gz"
        nib.save(source, source_path)
        # FLIRT only needs the first three dimensions of a 4D reference.
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
                "sinc",
                "-datatype",
                "float",
            ],
            check=True,
        )
        data = np.clip(
            nib.load(str(output_path)).get_fdata(dtype=np.float32),
            0.0,
            1.0,
        )
    return image_like(data, reference, dtype=np.float32)


def image_like(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    *,
    dtype: np.dtype | type = np.float32,
) -> nib.Nifti1Image:
    """Create an image on a reference grid with explicit datatype and forms."""
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)
    image.set_qform(reference.affine, code=1)
    image.set_sform(reference.affine, code=1)
    return image


def save_mask(
    mask: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    output_path: Path | str,
) -> Path:
    """Save a boolean mask as uint8 on a reference grid."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image_like(np.asarray(mask, dtype=bool), reference, dtype=np.uint8), path)
    return path


def load_mask(
    path: Path | str,
    reference: nib.spatialimages.SpatialImage | None = None,
    threshold: float = 0.5,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a finite, nonempty mask, optionally checking a reference grid."""
    image = load_image(path, ndim=3)
    if reference is not None and not same_grid(image, reference):
        raise ValueError(f"Mask grid does not match reference: {path}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Mask contains non-finite values: {path}")
    mask = data >= threshold
    if not np.any(mask):
        raise ValueError(f"Mask is empty at threshold {threshold:g}: {path}")
    return image, mask


def load_labels(labels_path: Path | str) -> list[str]:
    """Load a one-label-per-volume atlas label file."""
    labels = [line.strip() for line in Path(labels_path).read_text().splitlines()]
    labels = [label for label in labels if label]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"Atlas labels are empty or non-unique: {labels_path}")
    return labels


def build_cit168_probability(
    pseg: nib.spatialimages.SpatialImage,
    labels: Sequence[str],
    *,
    selected_labels: Sequence[str] = AMYGDALA_LABELS,
) -> nib.Nifti1Image:
    """Combine the ten CIT168 amygdala probability volumes."""
    if pseg.ndim != 4 or pseg.shape[3] != len(labels):
        raise ValueError(
            "CIT168 probability volume count does not match the label file: "
            f"{pseg.shape} and {len(labels)} labels."
        )
    missing = [label for label in selected_labels if label not in labels]
    if missing:
        raise ValueError(f"Missing CIT168 labels: {', '.join(missing)}")
    probability = np.zeros(pseg.shape[:3], dtype=np.float32)
    for label in selected_labels:
        index = labels.index(label)
        volume = np.asarray(pseg.dataobj[..., index], dtype=np.float32)
        probability += np.clip(volume, 0.0, 1.0)
    return image_like(np.clip(probability, 0.0, 1.0), pseg, dtype=np.float32)


def validate_dseg_labels(
    dseg: nib.spatialimages.SpatialImage,
    labels: Sequence[str],
    *,
    selected_labels: Sequence[str] = AMYGDALA_LABELS,
) -> None:
    """Verify one-based dseg values for the selected pseg volumes."""
    if dseg.ndim != 3:
        raise ValueError(f"Expected a 3D CIT168 dseg, found {dseg.shape}")
    present = {
        int(value)
        for value in np.unique(dseg.get_fdata(dtype=np.float32))
        if np.isfinite(value) and value > 0
    }
    expected = {labels.index(label) + 1 for label in selected_labels}
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Selected CIT168 dseg values are absent: {missing}")


def threshold_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a probability array using the manuscript's inclusive rule."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Probability threshold must be between zero and one.")
    values = np.asarray(probability, dtype=np.float32)
    return np.isfinite(values) & (values >= threshold)


def exclude_probability(
    mask: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Remove voxels with probability greater than or equal to a threshold."""
    mask = np.asarray(mask, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    if mask.shape != probability.shape:
        raise ValueError("Mask and probability arrays must have the same shape.")
    return mask & ~threshold_probability(probability, threshold)


def build_label_mask(label_data: np.ndarray, values: Sequence[int]) -> np.ndarray:
    """Return a mask selecting integer values from a label image."""
    data = np.asarray(label_data)
    if not np.isfinite(data).all():
        raise ValueError("Label image contains non-finite values.")
    rounded = np.rint(data)
    if not np.allclose(data, rounded):
        raise ValueError("Label image contains non-integer values.")
    return np.isin(rounded.astype(np.int16), tuple(values))


def candidate_label_values(
    candidates_path: Path | str,
    keys: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    """Read integer atlas values for named rows in a source-candidate table."""
    path = Path(candidates_path)
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    table = pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False)
    required = {"key", "values"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Candidate table is missing {sorted(missing)}: {path}")
    if table["key"].duplicated().any():
        duplicated = sorted(table.loc[table["key"].duplicated(), "key"].unique())
        raise ValueError(f"Candidate keys are duplicated: {duplicated}")

    by_key = table.set_index("key")["values"]
    output: dict[str, tuple[int, ...]] = {}
    for key in keys:
        if key not in by_key:
            raise ValueError(f"Candidate table has no row for {key!r}: {path}")
        values = tuple(
            int(item.strip())
            for item in str(by_key.loc[key]).replace(",", ";").split(";")
            if item.strip()
        )
        if not values or len(values) != len(set(values)) or min(values) < 1:
            raise ValueError(f"Invalid atlas values for {key!r}: {values}")
        output[key] = values
    return output


def build_s4_mesencephalic_target(
    bvr_label_data: np.ndarray,
    tian_label_data: np.ndarray,
    tian_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, S4MesencephalicCounts]:
    """Return the full mesencephalic mask and its portion outside THA-DP."""
    bvr = np.asarray(bvr_label_data)
    tian = np.asarray(tian_label_data)
    if bvr.shape != tian.shape:
        raise ValueError("BVR and Tian label arrays must have the same shape")
    thadp_values = tuple(
        index
        for index, label in enumerate(tian_labels, start=1)
        if label.upper().startswith(S4_THADP_PREFIX)
    )
    if len(thadp_values) != 2:
        raise ValueError(
            f"Expected bilateral Tian S2 THA-DP labels, found {len(thadp_values)}"
        )

    raw = build_label_mask(bvr, S4_MESENCEPHALIC_LABELS)
    thadp = build_label_mask(tian, thadp_values)
    target = raw & ~thadp
    if not np.any(raw):
        raise ValueError("Empty mesencephalic BVR mask")
    if not np.any(target):
        raise ValueError("Empty mesencephalic BVR target after THA-DP exclusion")
    counts = S4MesencephalicCounts(
        raw_mesencephalic=int(raw.sum()),
        thadp_overlap=int((raw & thadp).sum()),
        final_target=int(target.sum()),
    )
    return raw, target, counts


def build_s4_control_masks(
    hcp_mmp1_label_data: np.ndarray,
    candidate_values: Mapping[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    """Build the parcel masks used by the two Figure S4 positive controls.

    The auditory-belt analysis standardizes LBelt and MBelt separately before
    averaging their profiles. The union mask returned here records their
    anatomical support; it is not a substitute for that profile operation.
    """
    missing = set(S4_CONTROL_COMPONENT_KEYS).difference(candidate_values)
    if missing:
        raise ValueError(f"Missing Figure S4 control parcels: {sorted(missing)}")
    components = {
        key: build_label_mask(hcp_mmp1_label_data, candidate_values[key])
        for key in S4_CONTROL_COMPONENT_KEYS
    }
    empty = [key for key, mask in components.items() if not np.any(mask)]
    if empty:
        raise ValueError(f"Empty Figure S4 control parcels: {empty}")
    if np.any(components["glasser__v2"] & components["glasser__v1"]):
        raise ValueError("V2 target overlaps the V1 positive-control source")
    auditory_belt = components["glasser__lbelt"] | components["glasser__mbelt"]
    if np.any(auditory_belt & components["glasser__a1"]):
        raise ValueError("Auditory-belt target overlaps the A1 source")
    return {**components, "auditory_belt": auditory_belt}


def build_bvr_label_image(
    group_effect: nib.spatialimages.SpatialImage,
    *,
    threshold: float = 0.50,
) -> nib.Nifti1Image:
    """Create the six BVR labels on the original HCP 2-mm voxel grid.

    The fixed boxes index the native LAS storage order, so a reoriented image
    must be restored to that grid even if it represents the same physical map.
    """
    if group_effect.ndim != 3:
        raise ValueError(
            f"Expected a 3D group effect image, found {group_effect.shape}"
        )
    if group_effect.shape[:3] != BVR_GRID_SHAPE:
        raise ValueError(
            "The BVR bounding boxes are defined on the 91 x 109 x 91 HCP grid"
        )
    if not np.allclose(group_effect.affine, BVR_GRID_AFFINE, atol=1e-5, rtol=0):
        raise ValueError(
            "BVR voxel-index boxes require the native HCP MNI152NLin6Asym "
            "2-mm LAS affine "
            "[[-2,0,0,90],[0,2,0,-126],[0,0,2,-72],[0,0,0,1]]. "
            "Supply the original HCP-grid effect map; if reoriented, restore "
            "its native voxel order and affine together before defining labels."
        )
    if not np.isfinite(threshold):
        raise ValueError("BVR effect threshold must be finite")
    effect = group_effect.get_fdata(dtype=np.float32)
    labels = np.zeros(effect.shape, dtype=np.int16)
    for label, (i, di, j, dj, k, dk) in BVR_BOXES.items():
        box = np.zeros(effect.shape, dtype=bool)
        box[i : i + di, j : j + dj, k : k + dk] = True
        selected = box & np.isfinite(effect) & (effect > threshold)
        overlap = selected & (labels > 0)
        if np.any(overlap):
            raise ValueError(f"BVR boxes overlap at label {label}")
        labels[selected] = label
    missing = set(BVR_BOXES).difference(np.unique(labels))
    if missing:
        raise ValueError(f"No suprathreshold voxels for BVR labels {sorted(missing)}")
    return image_like(labels, group_effect, dtype=np.int16)
