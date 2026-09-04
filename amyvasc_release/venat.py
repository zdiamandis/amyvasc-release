"""Register the VENAT template to the HCP/CIT168 template before display."""

from __future__ import annotations

import json
import math
from pathlib import Path


def prepare_venat(
    *,
    venat: Path,
    fixed_template: Path,
    moving_template: Path,
    fixed_mask: Path,
    moving_mask: Path,
    output_dir: Path,
) -> Path:
    """Apply the finalized ANTsPy SyN recipe, retaining transforms and overlap QC.

    The fixed image is the 1-mm MNI152NLin6Asym T1w template; moving is the
    1-mm MNI152NLin2009cAsym T1w template. The masks are their brain masks.
    ANTsPy 0.6.3 runs in the optional environment documented with the renderer.
    """
    inputs = {
        "venat": venat,
        "fixed_template": fixed_template,
        "moving_template": moving_template,
        "fixed_mask": fixed_mask,
        "moving_mask": moving_mask,
    }
    for name, path in inputs.items():
        if not path.is_file() or not str(path).endswith((".nii", ".nii.gz")):
            raise ValueError(f"{name} must be an existing NIfTI image: {path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output directory is a file: {output_dir}")

    import ants

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(str(fixed_template))
    moving = ants.image_read(str(moving_template))
    fixed_brain = ants.image_read(str(fixed_mask))
    moving_brain = ants.image_read(str(moving_mask))
    atlas = ants.image_read(str(venat))
    if atlas.dimension != 3:
        raise ValueError("VENAT must be a 3D partial-volume image")
    for template, brain in ((fixed, fixed_brain), (moving, moving_brain)):
        if template.dimension != 3 or any(
            not math.isfinite(value) or abs(value - 1) > 1e-5
            for value in template.spacing
        ):
            raise ValueError("Registration requires 3D T1 templates at 1 mm resolution")
        if template.shape != brain.shape or not ants.image_physical_space_consistency(
            template, brain
        ):
            raise ValueError("Each brain mask must share its T1 template's grid")
        values = brain.numpy()
        if not math.isfinite(float(values.sum())) or not (values > 0.5).any():
            raise ValueError("Brain masks must be finite and nonempty")
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        verbose=False,
        outprefix=str(output_dir / "registration_"),
    )
    target = fixed_brain.numpy() > 0.5

    def dice(image):
        candidate = image.numpy() > 0.5
        return float(2 * (candidate & target).sum() / (candidate.sum() + target.sum()))

    baseline = dice(
        ants.resample_image_to_target(moving_brain, fixed_brain, interp_type="linear")
    )
    registered = dice(
        ants.apply_transforms(
            fixed=fixed_brain,
            moving=moving_brain,
            transformlist=registration["fwdtransforms"],
            interpolator="linear",
        )
    )
    if not math.isfinite(registered) or registered <= baseline:
        raise ValueError(
            f"Template registration did not improve brain-mask Dice: {registered:.4f} <= {baseline:.4f}"
        )
    reference = ants.resample_image(
        fixed, (0.5, 0.5, 0.5), use_voxels=False, interp_type=4
    )
    warped = ants.apply_transforms(
        fixed=reference,
        moving=atlas,
        transformlist=registration["fwdtransforms"],
        interpolator="linear",
    )
    destination = (
        output_dir / "VENAT_PartialVolume_space-MNI152NLin6Asym_res-0p5.nii.gz"
    )
    ants.image_write(warped, str(destination))
    provenance = {
        "source_space": "MNI152NLin2009cAsym",
        "target_space": "MNI152NLin6Asym",
        "method": "ANTsPy SyN, 1-mm T1w templates; linear atlas interpolation",
        "source_atlas": str(venat.resolve()),
        "fixed_template": str(fixed_template.resolve()),
        "moving_template": str(moving_template.resolve()),
        "fixed_mask": str(fixed_mask.resolve()),
        "moving_mask": str(moving_mask.resolve()),
        "brain_mask_dice_baseline_world_coords": baseline,
        "brain_mask_dice_after_registration": registered,
        "output_resolution_mm": 0.5,
        "antspyx": ants.__version__,
        "forward_transforms": registration["fwdtransforms"],
        "inverse_transforms": registration["invtransforms"],
        "output": str(destination),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination
