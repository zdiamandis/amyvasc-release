"""Display the manuscript's HCP image comparisons from locally generated maps.

Inputs must be registered to MNI152NLin6Asym. Coordinates and display scales
match the paper; these standalone panels do not reproduce its page typography.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from amyvasc_release.source_profiles import CONDITIONS
from amyvasc_release.masks import require_binary_mask
from ._common import read_table, save_figure

AMY_COLOR = "#00E676"
SEGMENTS = (
    ("Striate", (1, 2), "#D95F5F"),
    ("Peduncular", (3, 4), "#7A5EA8"),
    ("Mesencephalic", (5, 6), "white"),
)
ACTIVATION_CMAP = ListedColormap(plt.get_cmap("hot")(np.linspace(0, 0.85, 256)))


def slice_or_mip(data, reference, axis, coordinate, slab=None):
    """Take an axial/sagittal slice or an inclusive slab maximum in world mm."""
    if axis not in (0, 2):
        raise ValueError("Use axis 0 (sagittal) or 2 (axial)")
    affine = reference.affine
    if not np.allclose(affine[:3, :3], np.diag(np.diag(affine)[:3])):
        raise ValueError("HCP display requires an axis-aligned MNI grid")
    if np.any(np.diag(affine)[:3] <= 0):
        raise ValueError("Canonicalize the image before displaying it")
    coordinates = np.arange(data.shape[axis]) * affine[axis, axis] + affine[axis, 3]
    if slab is None:
        if not coordinates[0] <= coordinate <= coordinates[-1]:
            raise ValueError("Slice coordinate lies outside the image")
        plane = np.take(data, np.argmin(abs(coordinates - coordinate)), axis=axis)
    else:
        indices = np.flatnonzero((coordinates >= slab[0]) & (coordinates <= slab[1]))
        if not len(indices):
            raise ValueError("Projection slab does not intersect the image")
        values = np.take(data, indices, axis=axis)
        plane = np.max(np.where(np.isfinite(values), values, -np.inf), axis=axis)
        plane = np.where(np.isfinite(plane), plane, np.nan)
    remaining = [index for index in range(3) if index != axis]
    extent = []
    for index in remaining:
        start, step = affine[index, 3], affine[index, index]
        extent.extend((start - step / 2, start + (data.shape[index] - 0.5) * step))
    return plane.T, extent


def on_grid(path, reference, *, labels=False):
    image = nib.load(path)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D input: {path}")
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine
    ):
        image = resample_from_to(image, reference, order=0 if labels else 1)
    return image.get_fdata(dtype=np.float32)


def draw(
    ax,
    reference,
    background,
    amygdala,
    *,
    effect=None,
    axis=2,
    coordinate=-12,
    slab=None,
    wide=False,
    contours=(),
):
    back, extent = slice_or_mip(background, reference, axis, coordinate)
    finite = back[np.isfinite(back)]
    ax.imshow(
        back,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=np.percentile(finite, 1),
        vmax=np.percentile(finite, 99),
    )
    if effect is not None:
        plane, _ = slice_or_mip(effect, reference, axis, coordinate, slab)
        positive = np.isfinite(plane) & (plane > 0)
        alpha = np.where(positive, 0.94 * np.clip(plane, 0, 1) ** 0.45, 0)
        ax.imshow(
            np.where(positive, plane, 0),
            origin="lower",
            extent=extent,
            cmap=ACTIVATION_CMAP,
            vmin=0,
            vmax=1,
            alpha=alpha,
            interpolation="nearest",
        )
    # Projecting the amygdala would collapse the separation axis in the MIPs.
    mask_contours = ((amygdala, AMY_COLOR, "solid"),) if slab is None else ()
    for mask, color, style in (*mask_contours, *contours):
        plane, _ = slice_or_mip(mask.astype(float), reference, axis, coordinate, slab)
        if np.nanmin(plane) < 0.5 < np.nanmax(plane):
            ax.contour(
                plane,
                levels=[0.5],
                colors=[color],
                linestyles=[style],
                linewidths=1,
                origin="lower",
                extent=extent,
            )
    if axis == 2:
        ax.set_xlim((-90, 90) if wide else (-38, 38))
        ax.set_ylim((-126, 30) if wide else (-36, 14))
        ax.text(0.02, 0.96, "L", color="white", transform=ax.transAxes, va="top")
        ax.text(
            0.98, 0.96, "R", color="white", transform=ax.transAxes, ha="right", va="top"
        )
    else:
        ax.set_xlim(-126, 90)
        ax.set_ylim(-46, 36)
        ax.text(0.02, 0.96, "P", color="white", transform=ax.transAxes, va="top")
        ax.text(
            0.98, 0.96, "A", color="white", transform=ax.transAxes, ha="right", va="top"
        )
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def add_scale(fig, axes):
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=ACTIVATION_CMAP),
        ax=list(np.asarray(axes).ravel()),
        orientation="horizontal",
        shrink=0.45,
        pad=0.03,
        label="Effect (% signal change)",
    )


def add_mask_legend(fig, entries, *, ncols=3):
    handles = [
        Line2D([], [], color=color, linestyle=style, label=label)
        for label, color, style in entries
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=ncols,
        facecolor="#222222",
        labelcolor="white",
        framealpha=1,
        fontsize=9,
    )


def render(
    *,
    group_root: Path,
    background_path: Path,
    amygdala_path: Path,
    probability_path: Path,
    bvr_labels_path: Path,
    data_dir: Path,
    output_dir: Path,
    ho50_path: Path | None = None,
    ho25_path: Path | None = None,
) -> list[Path]:
    if (ho50_path is None) != (ho25_path is None):
        raise ValueError("Supply both Harvard-Oxford amygdala masks for S7")
    require_binary_mask(amygdala_path)
    if ho50_path is not None:
        require_binary_mask(ho50_path)
        require_binary_mask(ho25_path)
    reference = nib.as_closest_canonical(nib.load(amygdala_path))
    amygdala = reference.get_fdata() > 0
    background = on_grid(background_path, reference)
    probability = on_grid(probability_path, reference)
    labels = on_grid(bvr_labels_path, reference, labels=True).round().astype(int)
    outputs = []

    def effect(task, stem):
        return on_grid(
            group_root / task / "group_display" / f"group_effect_{stem}.nii.gz",
            reference,
        )

    # Figure 1B: four task contrasts on the same axial slice and color scale.
    contrasts = (
        ("emotion", "fear_minus_shape", "Fear - shape"),
        ("wm", "0bk_faces_minus_0bk_places", "Faces - places"),
        ("social", "mental_minus_random", "Mental - random"),
        ("language", "story_minus_math", "Story - math"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), layout="constrained")
    for ax, (task, stem, title) in zip(axes, contrasts, strict=True):
        draw(ax, reference, background, amygdala, effect=effect(task, stem))
        ax.set_title(title)
    add_scale(fig, axes)
    outputs += save_figure(fig, output_dir, "figure1_image_panels")

    # S1: condition order and title values come from the exported ROI results.
    profile = read_table(
        data_dir,
        "figureS1_condition_amygdala_trace_profile.tsv",
        {"condition_key", "condition_label", "amy_proper"},
    )
    profile = profile.sort_values("amy_proper", ascending=False)
    conditions = {item.key: item for item in CONDITIONS}
    if set(profile.condition_key) != set(conditions) or len(profile) != len(conditions):
        raise ValueError("S1 requires one row for each of the 23 task conditions")
    fig, axes = plt.subplots(6, 4, figsize=(12, 16), layout="constrained")
    for ax, row in zip(axes.ravel(), profile.itertuples(index=False)):
        condition = conditions[row.condition_key]
        draw(
            ax,
            reference,
            background,
            amygdala,
            effect=effect(condition.task, condition.effect),
        )
        ax.set_title(f"{row.condition_label}\nAmy: {row.amy_proper:.3f}%", fontsize=10)
    axes.ravel()[-1].set_visible(False)
    add_scale(fig, axes.ravel()[:-1])
    outputs += save_figure(fig, output_dir, "figureS1")

    fear = effect("emotion", "fear_minus_shape")
    # S6: expanded field, followed by the exact two limited MIP slabs.
    views = (
        (2, -12, None, "Axial z = -12 mm"),
        (0, 14, None, "Sagittal x = +14 mm"),
        (0, 18, None, "Sagittal x = +18 mm"),
        (0, 22, None, "Sagittal x = +22 mm"),
        (0, 18, (6, 30), "Sagittal MIP: x = +6 to +30 mm"),
        (2, -12, (-28, 2), "Axial MIP: z = -28 to +2 mm"),
    )
    segment_contours = [
        (np.isin(labels, values), color, "solid") for _, values, color in SEGMENTS
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), layout="constrained")
    for ax, (axis, coordinate, slab, title) in zip(axes.ravel(), views, strict=True):
        draw(
            ax,
            reference,
            background,
            amygdala,
            effect=fear,
            axis=axis,
            coordinate=coordinate,
            slab=slab,
            wide=True,
            contours=segment_contours,
        )
        ax.set_title(title)
    add_scale(fig, axes)
    add_mask_legend(
        fig,
        [
            ("CIT168 amygdala (slices)", AMY_COLOR, "solid"),
            *[(name, color, "solid") for name, _, color in SEGMENTS],
        ],
        ncols=4,
    )
    outputs += save_figure(fig, output_dir, "figureS6")

    # Mask comparisons: use the computed segment labels and probability cutoffs.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), layout="constrained")
    for ax, threshold in zip(axes, (0.50, 0.20, 0.05), strict=True):
        contours = [
            (np.isin(labels, values) & (probability < threshold), color, "solid")
            for _, values, color in SEGMENTS[:2]
        ]
        draw(ax, reference, background, amygdala, contours=contours)
        ax.set_title(f"Exclude p(Amy) >= {threshold:.2f}")
    mask_legend = [
        ("CIT168 amygdala", AMY_COLOR, "solid"),
        *[(name, color, "solid") for name, _, color in SEGMENTS[:2]],
    ]
    add_mask_legend(fig, mask_legend)
    outputs += save_figure(fig, output_dir, "figureS5_mask_panels")
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    draw(
        ax,
        reference,
        background,
        amygdala,
        contours=[
            (np.isin(labels, values) & ~amygdala, color, "solid")
            for _, values, color in SEGMENTS[:2]
        ],
    )
    ax.set_title("Amygdala, striate, and peduncular masks")
    add_mask_legend(fig, mask_legend, ncols=2)
    outputs += save_figure(fig, output_dir, "figure4_mask_panel")
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    draw(ax, reference, background, np.isin(labels, (5, 6)), effect=fear, coordinate=-2)
    ax.set_title("Mesencephalic mask; z = -2 mm")
    add_scale(fig, [ax])
    outputs += save_figure(fig, output_dir, "figureS4_mask_panel")
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    draw(
        ax, reference, background, amygdala, effect=effect("motor", "tongue_canonical")
    )
    ax.set_title("Motor tongue; z = -12 mm")
    add_scale(fig, [ax])
    outputs += save_figure(fig, output_dir, "figure8_image_panel")

    if ho50_path is not None:
        contours = [
            (on_grid(ho50_path, reference, labels=True) > 0, "#4285B4", "dashed"),
            (on_grid(ho25_path, reference, labels=True) > 0, "#E5A52D", "dotted"),
            *[
                (np.isin(labels, values) & ~amygdala, color, "solid")
                for _, values, color in SEGMENTS[:2]
            ],
        ]
        fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
        draw(ax, reference, background, amygdala, contours=contours)
        ax.set_title("CIT168 and Harvard-Oxford boundaries")
        add_mask_legend(
            fig,
            [
                *mask_legend,
                ("HO maxprob50", "#4285B4", "dashed"),
                ("HO maxprob25", "#E5A52D", "dotted"),
            ],
            ncols=2,
        )
        outputs += save_figure(fig, output_dir, "figureS7_atlas_panel")
    return outputs
