"""Shared first-level GLM and within-participant fixed-effects operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import compute_fixed_effects
from nilearn.glm.first_level import FirstLevelModel
from nilearn.masking import compute_multi_epi_mask

from .provenance import software_versions


@dataclass(frozen=True)
class Contrast:
    """A t contrast expressed as weights on exact design-matrix columns."""

    stem: str
    weights: dict[str, float]


@dataclass(frozen=True)
class GLMSettings:
    """First-level settings shared by the manuscript task analyses."""

    tr: float
    hrf_model: str = "spm + derivative"
    drift_model: str = "cosine"
    high_pass: float = 1.0 / 128.0
    noise_model: str = "ar1"
    signal_scaling: int | bool = 0
    slice_time_ref: float = 0.0
    smoothing_fwhm: float | None = None


@dataclass(frozen=True)
class RunResult:
    """Paths and degrees of freedom produced by one fitted run."""

    label: str
    output_dir: Path
    degrees_of_freedom: int

    def contrast_path(self, stem: str, kind: str) -> Path:
        """Return a saved effect, variance, stat, or z map."""
        if kind not in {"effect", "variance", "stat", "z"}:
            raise ValueError(f"Unknown contrast-map kind: {kind}")
        return self.output_dir / "contrasts" / f"{stem}_{kind}.nii.gz"


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one compact, stable JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write records as a tab-separated table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def same_grid(
    image: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
) -> bool:
    """Return whether two images share a three-dimensional voxel grid."""
    return image.shape[:3] == reference.shape[:3] and np.allclose(
        image.affine, reference.affine
    )


def image_like(
    reference: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    *,
    dtype: np.dtype | type = np.float32,
) -> nib.Nifti1Image:
    """Wrap an array in a NIfTI image on an existing grid."""
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    return nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header)


def save_image(image: nib.spatialimages.SpatialImage, path: Path) -> None:
    """Save a NIfTI image after creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


def masked_float_image(
    image: nib.spatialimages.SpatialImage,
    mask_img: nib.spatialimages.SpatialImage,
) -> nib.Nifti1Image:
    """Store an image as float32 with NaN outside the analysis mask."""
    if not same_grid(image, mask_img):
        raise ValueError("Image and mask grids do not match.")
    data = image.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata(dtype=np.float32) > 0
    output = np.full(data.shape, np.nan, dtype=np.float32)
    output[mask] = data[mask]
    return image_like(image, output)


def common_epi_mask(bold_paths: list[Path]) -> nib.Nifti1Image:
    """Compute the common positive-signal mask used for task GLMs."""
    if not bold_paths:
        raise ValueError("At least one BOLD image is required.")
    mask_img = compute_multi_epi_mask([str(path) for path in bold_paths])
    mask = mask_img.get_fdata(dtype=np.float32) > 0
    for path in bold_paths:
        bold_img = nib.load(path)
        if bold_img.ndim != 4:
            raise ValueError(f"Expected a 4D BOLD image: {path}")
        if not same_grid(bold_img, mask_img):
            raise ValueError(f"BOLD images do not share one grid: {path}")
        mean = np.asarray(bold_img.dataobj, dtype=np.float32).mean(axis=3)
        mask &= np.isfinite(mean) & (mean > 0)
    if not np.any(mask):
        raise ValueError("The common BOLD mask is empty.")
    return image_like(mask_img, mask, dtype=np.uint8)


def intersect_masks(mask_paths: list[Path]) -> nib.Nifti1Image:
    """Intersect run-specific masks, treating nonzero finite values as support."""
    if not mask_paths:
        raise ValueError("At least one mask image is required.")
    reference = nib.load(mask_paths[0])
    common = np.ones(reference.shape[:3], dtype=bool)
    for path in mask_paths:
        image = nib.load(path)
        if image.ndim != 3 or not same_grid(image, reference):
            raise ValueError(f"Mask grid mismatch: {path}")
        data = image.get_fdata(dtype=np.float32)
        common &= np.isfinite(data) & (data != 0)
    if not np.any(common):
        raise ValueError("The common run mask is empty.")
    return image_like(reference, common, dtype=np.uint8)


def contrast_vector(design: pd.DataFrame, contrast: Contrast) -> np.ndarray:
    """Translate named contrast weights into the fitted design column order."""
    missing = sorted(set(contrast.weights).difference(design.columns))
    if missing:
        raise ValueError(
            f"Contrast {contrast.stem} requires missing columns {missing}; "
            f"available columns are {list(design.columns)}."
        )
    vector = np.zeros(design.shape[1], dtype=np.float64)
    for column, weight in contrast.weights.items():
        vector[int(design.columns.get_loc(column))] = float(weight)
    return vector


def residual_dof(design: pd.DataFrame) -> int:
    """Return scan count minus design rank."""
    rank = int(np.linalg.matrix_rank(design.to_numpy(dtype=np.float64)))
    dof = int(len(design) - rank)
    if dof < 1:
        raise ValueError(f"Design has no residual degrees of freedom (rank={rank}).")
    return dof


def fit_run(
    *,
    bold_path: Path,
    mask_img: nib.spatialimages.SpatialImage | Path,
    output_dir: Path,
    contrasts: tuple[Contrast, ...] | list[Contrast],
    settings: GLMSettings,
    events: pd.DataFrame | None = None,
    design_matrix: pd.DataFrame | None = None,
    run_label: str = "run",
) -> RunResult:
    """Fit one BOLD run and save effect, variance, t, and z maps.

    Supplying ``settings.smoothing_fwhm`` passes the requested FWHM to
    :class:`nilearn.glm.first_level.FirstLevelModel`, so smoothing is applied to
    the four-dimensional BOLD series before the model is fitted.
    """
    if (events is None) == (design_matrix is None):
        raise ValueError("Provide exactly one of events or design_matrix.")
    mask = nib.load(mask_img) if isinstance(mask_img, Path) else mask_img
    output_dir.mkdir(parents=True, exist_ok=True)
    model = FirstLevelModel(
        t_r=settings.tr,
        slice_time_ref=settings.slice_time_ref,
        hrf_model=settings.hrf_model,
        drift_model=settings.drift_model,
        high_pass=settings.high_pass,
        mask_img=mask,
        noise_model=settings.noise_model,
        signal_scaling=settings.signal_scaling,
        smoothing_fwhm=settings.smoothing_fwhm,
        minimize_memory=True,
        verbose=0,
    )
    if design_matrix is None:
        model.fit(str(bold_path), events=events)
    else:
        model.fit(str(bold_path), design_matrices=design_matrix)
    fitted_design = model.design_matrices_[0]
    fitted_design.to_csv(output_dir / "design_matrix.tsv", sep="\t", index=False)
    dof = residual_dof(fitted_design)

    for spec in contrasts:
        outputs = model.compute_contrast(
            contrast_vector(fitted_design, spec), stat_type="t", output_type="all"
        )
        maps = {
            "effect": outputs["effect_size"],
            "variance": outputs["effect_variance"],
            "stat": outputs["stat"],
            "z": outputs["z_score"],
        }
        for kind, image in maps.items():
            save_image(
                masked_float_image(image, mask),
                output_dir / "contrasts" / f"{spec.stem}_{kind}.nii.gz",
            )

    write_json(
        output_dir / "run_metadata.json",
        {
            "run_label": run_label,
            "bold_path": str(bold_path),
            "n_scans": int(len(fitted_design)),
            "design_rank": int(len(fitted_design) - dof),
            "degrees_of_freedom": dof,
            "contrasts": [spec.stem for spec in contrasts],
            "smoothing_fwhm": settings.smoothing_fwhm,
            "software_versions": software_versions(),
        },
    )
    return RunResult(run_label, output_dir, dof)


def combine_runs(
    *,
    run_results: list[RunResult],
    mask_img: nib.spatialimages.SpatialImage | Path,
    output_dir: Path,
    contrast_stems: list[str] | tuple[str, ...],
) -> None:
    """Combine run estimates within participant by precision-weighted fixed effects."""
    if not run_results:
        raise ValueError("At least one fitted run is required for fixed effects.")
    mask = nib.load(mask_img) if isinstance(mask_img, Path) else mask_img
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in contrast_stems:
        effect_paths = [result.contrast_path(stem, "effect") for result in run_results]
        variance_paths = [
            result.contrast_path(stem, "variance") for result in run_results
        ]
        missing = [
            path for path in [*effect_paths, *variance_paths] if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing fixed-effects input: {missing[0]}")
        effect, variance, stat, z_score = compute_fixed_effects(
            contrast_imgs=effect_paths,
            variance_imgs=variance_paths,
            mask=mask,
            precision_weighted=True,
            dofs=np.asarray(
                [result.degrees_of_freedom for result in run_results],
                dtype=np.float64,
            ),
        )
        for kind, image in {
            "effect": effect,
            "variance": variance,
            "stat": stat,
            "z": z_score,
        }.items():
            save_image(
                masked_float_image(image, mask),
                output_dir / "contrasts" / f"{stem}_{kind}.nii.gz",
            )

    support_path = output_dir / "support_mask.nii.gz"
    save_image(image_like(mask, mask.get_fdata() > 0, dtype=np.uint8), support_path)
    write_json(
        output_dir / "fixed_effects_metadata.json",
        {
            "run_labels": [result.label for result in run_results],
            "degrees_of_freedom": [result.degrees_of_freedom for result in run_results],
            "fixed_effects_dof": int(
                sum(result.degrees_of_freedom for result in run_results)
            ),
            "contrasts": list(contrast_stems),
            "precision_weighted": True,
            "software_versions": software_versions(),
        },
    )


def group_mean_maps(
    *,
    fixed_effect_dirs: list[Path],
    output_dir: Path,
    contrast_stems: list[str] | tuple[str, ...],
    minimum_coverage: float = 0.50,
) -> None:
    """Write coverage-aware descriptive means of participant fixed effects."""
    if not fixed_effect_dirs:
        raise ValueError(
            "At least one participant fixed-effects directory is required."
        )
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be between zero and one.")
    support_paths = [path / "support_mask.nii.gz" for path in fixed_effect_dirs]
    reference = nib.load(support_paths[0])
    supports: list[np.ndarray] = []
    for path in support_paths:
        image = nib.load(path)
        if not same_grid(image, reference):
            raise ValueError(f"Participant grid mismatch: {path}")
        supports.append(image.get_fdata(dtype=np.float32) > 0)
    support_stack = np.stack(supports)
    coverage_count = support_stack.sum(axis=0).astype(np.uint16)
    coverage_fraction = coverage_count.astype(np.float32) / len(fixed_effect_dirs)
    required = max(1, int(np.ceil(minimum_coverage * len(fixed_effect_dirs))))
    display_mask = coverage_count >= required

    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        image_like(reference, display_mask, dtype=np.uint8),
        output_dir / "group_display_mask.nii.gz",
    )
    save_image(
        image_like(reference, coverage_count, dtype=np.uint16),
        output_dir / "group_coverage_count.nii.gz",
    )
    save_image(
        image_like(reference, coverage_fraction),
        output_dir / "group_coverage_fraction.nii.gz",
    )

    for stem in contrast_stems:
        total = np.zeros(reference.shape[:3], dtype=np.float64)
        count = np.zeros(reference.shape[:3], dtype=np.uint16)
        for subject_dir, support in zip(fixed_effect_dirs, supports, strict=True):
            path = subject_dir / "contrasts" / f"{stem}_effect.nii.gz"
            image = nib.load(path)
            if not same_grid(image, reference):
                raise ValueError(f"Participant effect grid mismatch: {path}")
            data = image.get_fdata(dtype=np.float32)
            valid = display_mask & support & np.isfinite(data)
            total[valid] += data[valid]
            count[valid] += 1
        mean = np.full(reference.shape[:3], np.nan, dtype=np.float32)
        valid = display_mask & (count >= required)
        mean[valid] = (total[valid] / count[valid]).astype(np.float32)
        save_image(
            image_like(reference, mean),
            output_dir / f"group_effect_{stem}.nii.gz",
        )

    write_json(
        output_dir / "summary.json",
        {
            "n_participants": len(fixed_effect_dirs),
            "minimum_coverage": minimum_coverage,
            "minimum_participants": required,
            "contrasts": list(contrast_stems),
            "inference": "descriptive participant mean",
        },
    )
