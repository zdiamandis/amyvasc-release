"""Figure 3 and Supplements S2/S3 from prepared, registered NIfTI images.

The renderer retains the manuscript coordinates, projections, display ranges,
and contour sampling. It does not register images or reconstruct angiography.
See ``analysis/vascular_inputs.md`` for the JSON input manifest and preparation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates

from amyvasc_release.masks import (
    AMYGDALA_LABELS,
    load_labels,
    require_binary_mask,
    same_grid,
)

XLIM, YLIM = (-42.0, 42.0), (-36.0, 14.0)
EXPANDED_XLIM, EXPANDED_YLIM = (-58.0, 58.0), (-55.0, 30.0)
RESOLUTION = 0.5
AMY_COLOR = "#00E676"
FRANGI_MAX = 0.25
HOT = colors.ListedColormap(plt.get_cmap("hot")(np.linspace(0, 0.85, 256)))


def load_manifest(path: Path) -> dict:
    """Resolve image paths relative to the JSON file, leaving coordinates explicit."""
    config = json.loads(path.read_text())
    if len(config.get("subjects", [])) != 3:
        raise ValueError("The manuscript panels require three subject rows.")
    path_keys = {
        "effect",
        "venat",
        "anatomy",
        "amygdala",
        "emotion_effect",
        "movie_effect",
        "frangi",
        "t2w",
        "tof",
        "movie_amygdala_pseg",
        "cit168_labels",
    }
    for row in [config.get("hcp", {}), *config["subjects"]]:
        for key in row.keys() & path_keys:
            value = Path(row[key]).expanduser()
            row[key] = str(
                value if value.is_absolute() else (path.parent / value).resolve()
            )
    return config


def _load(path: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(path)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D prepared image: {path}")
    return image, image.get_fdata(dtype=np.float32)


def world_grid(xlim=XLIM, ylim=YLIM):
    x = np.arange(xlim[0], xlim[1] + RESOLUTION * 0.5, RESOLUTION)
    y = np.arange(ylim[0], ylim[1] + RESOLUTION * 0.5, RESOLUTION)
    extent = (
        x[0] - RESOLUTION / 2,
        x[-1] + RESOLUTION / 2,
        y[0] - RESOLUTION / 2,
        y[-1] + RESOLUTION / 2,
    )
    return x, y, extent


def sample_slice(image, data, x, y, z, *, order=0):
    """Sample a plane in registered world coordinates for display only."""
    xx, yy = np.meshgrid(x, y)
    world = np.vstack([xx.ravel(), yy.ravel(), np.full(xx.size, z), np.ones(xx.size)])
    voxel = np.linalg.inv(image.affine) @ world
    return map_coordinates(
        data, voxel[:3], order=order, mode="constant", cval=np.nan, prefilter=order > 1
    ).reshape(xx.shape)


def sample_mip(image, data, x, y, z, *, width, order):
    """Centered axial MIP, sampled every 0.5 mm including both slab endpoints."""
    planes = np.arange(z - width / 2, z + width / 2 + RESOLUTION * 0.5, RESOLUTION)
    stack = np.stack(
        [sample_slice(image, data, x, y, plane, order=order) for plane in planes]
    )
    finite = np.isfinite(stack)
    result = np.max(np.where(finite, stack, -np.inf), axis=0)
    result[~finite.any(axis=0)] = np.nan
    return result


def native_slice(image, data, z):
    """Keep the native voxel grid for HCP activation and binary contours."""
    affine = image.affine
    if not np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))):
        raise ValueError(
            "Native slice display requires axis-aligned registered images."
        )
    k = int(round((np.linalg.inv(affine) @ [0, 0, z, 1])[2]))
    if not 0 <= k < image.shape[2]:
        raise ValueError(f"Slice z={z} lies outside the image.")
    x = affine[0, 0] * np.arange(image.shape[0]) + affine[0, 3]
    y = affine[1, 1] * np.arange(image.shape[1]) + affine[1, 3]
    ix = np.flatnonzero((x >= XLIM[0]) & (x <= XLIM[1]))
    iy = np.flatnonzero((y >= YLIM[0]) & (y <= YLIM[1]))
    if not len(ix) or not len(iy):
        raise ValueError("No image voxels fall inside the manuscript crop.")
    ix, iy = ix[np.argsort(x[ix])], iy[np.argsort(y[iy])]
    return data[np.ix_(ix, iy, [k])][:, :, 0].T, (
        x[ix[0]],
        x[ix[-1]],
        y[iy[0]],
        y[iy[-1]],
    )


def _zlabel(z):
    return f"{round(z / RESOLUTION) * RESOLUTION:.1f}".rstrip("0").rstrip(".")


def _format_axis(ax, title, z, *, expanded=False, movie=False):
    title_size = 11.6 if expanded else 13.2 if movie else 14.85
    coordinate_size = 8.68 if expanded else 9.1 if movie else 10.85
    ax.set_title(title, fontsize=title_size, pad=5)
    ax.set(
        xlim=EXPANDED_XLIM if expanded else XLIM,
        ylim=EXPANDED_YLIM if expanded else YLIM,
        aspect="equal",
        facecolor="black",
        xticks=[],
        yticks=[],
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.02,
        0.035 if expanded else 0.04,
        f"z={_zlabel(z)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=coordinate_size,
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.45, pad=1.6),
    )


def _subject_label(ax, label, *, movie=False):
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.4 if movie else 12.4,
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.45, pad=1.6),
    )


def _layer(ax, data, extent, *, cmap, vmax, alpha=1.0):
    display = np.ma.masked_invalid(
        np.where(np.isfinite(data) & (data > 0), data, np.nan)
    )
    ax.imshow(
        display,
        cmap=cmap,
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=vmax,
        alpha=alpha,
        interpolation="nearest",
    )


def _outline(ax, data, extent):
    if np.isfinite(data).any() and np.nanmax(data) >= 0.5:
        ax.contour(
            data,
            levels=[0.5],
            colors=AMY_COLOR,
            linewidths=1.1,
            origin="lower",
            extent=extent,
        )


def _colorbar(fig, ax, cmap, vmax, label, *, zero_as_int=False, scale=1.55):
    scalar = ScalarMappable(norm=colors.Normalize(0, vmax), cmap=cmap)
    scalar.set_array([])
    bar = fig.colorbar(scalar, cax=ax, orientation="horizontal")
    bar.set_label(label, fontsize=8 * scale)
    bar.ax.tick_params(labelsize=7 * scale)
    if zero_as_int:
        bar.ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda value, _: "0" if abs(value) < 1e-9 else f"{value:g}"
            )
        )


def _hcp_row(axes, row):
    z = float(row.get("z", -12))
    effect_img, effect = _load(row["effect"])
    venat_img, venat = _load(row["venat"])
    anatomy_img, anatomy = _load(row["anatomy"])
    amy_img = require_binary_mask(row["amygdala"])
    if not same_grid(effect_img, amy_img):
        raise ValueError("The HCP amygdala mask must be on the activation grid.")
    x, y, extent = world_grid()
    vessel = sample_mip(venat_img, venat, x, y, z, width=3, order=1)
    activation, activation_extent = native_slice(effect_img, effect, z)
    amy, amy_extent = native_slice(amy_img, amy_img.get_fdata(dtype=np.float32), z)
    background, background_extent = native_slice(anatomy_img, anatomy, z)
    for ax, title in zip(
        axes,
        (
            "Task activation (HCP-YA group)",
            "Vascular map (VENAT MIP, 3 mm)",
            "Activation + vascular map",
        ),
    ):
        _format_axis(ax, title, z)
    low, high = np.percentile(background[np.isfinite(background)], [1, 99])
    axes[0].imshow(
        background,
        cmap="gray",
        origin="lower",
        extent=background_extent,
        vmin=low,
        vmax=high,
        interpolation="bilinear",
    )
    for ax in axes[1:]:
        ax.imshow(
            np.zeros_like(vessel),
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=1,
        )
        _layer(ax, vessel, extent, cmap="gray", vmax=0.4)
    positive = np.isfinite(activation) & (activation > 0)
    for ax, ceiling in ((axes[0], 0.94), (axes[2], 0.68)):
        opacity = np.where(positive, ceiling * np.clip(activation, 0, 1) ** 0.45, 0)
        _layer(ax, activation, activation_extent, cmap=HOT, vmax=1, alpha=opacity)
    for ax in axes:
        _outline(ax, amy, amy_extent)


def _movie_amygdala(row):
    """S2 uses the continuous subject-space CIT168 union, as in the manuscript."""
    image = nib.load(row["movie_amygdala_pseg"])
    labels = load_labels(row["cit168_labels"])
    if image.ndim != 4 or image.shape[3] != len(labels):
        raise ValueError("Subject CIT168 probabilities do not match the label table.")
    probability = np.zeros(image.shape[:3], np.float32)
    for label in AMYGDALA_LABELS:
        probability += np.asarray(image.dataobj[..., labels.index(label)], np.float32)
    return image, np.clip(probability, 0, 1)


def _subject_row(axes, row, *, movie=False, titles=False):
    z = float(row["z"])
    effect_img, effect = _load(row["movie_effect" if movie else "emotion_effect"])
    frangi_img, frangi = _load(row["frangi"])
    x, y, extent = world_grid()
    activation = sample_slice(effect_img, effect, x, y, z, order=0)
    vessel = sample_mip(frangi_img, frangi, x, y, z, width=5, order=0)
    if movie:
        amy_img, probability = _movie_amygdala(row)
        amy = sample_slice(amy_img, probability, x, y, z, order=1)
        amy_extent = extent
    else:
        amy_img = require_binary_mask(row["amygdala"])
        if not same_grid(effect_img, amy_img):
            raise ValueError(f"Emotion mask must be on the effect grid: {row['label']}")
        amy, amy_extent = native_slice(amy_img, amy_img.get_fdata(dtype=np.float32), z)
    headings = (
        ("Movie face presence", "Vascular map", "Movie + vascular")
        if movie
        else (
            "Task activation (internal)",
            "Vascular map (Frangi MIP, 5 mm)",
            "Activation + vascular map",
        )
    )
    for ax, title in zip(axes, headings if titles else ("", "", "")):
        _format_axis(ax, title, z, movie=movie)
    _subject_label(axes[0], row["label"], movie=movie)
    _layer(axes[0], activation, extent, cmap=HOT, vmax=0.8 if movie else 3)
    _layer(axes[1], vessel, extent, cmap="gray", vmax=FRANGI_MAX)
    _layer(axes[2], activation, extent, cmap=HOT, vmax=0.8 if movie else 3)
    _layer(axes[2], vessel, extent, cmap="gray", vmax=FRANGI_MAX, alpha=0.45)
    for ax in axes:
        _outline(ax, amy, amy_extent)


def render_figure3(config):
    fig = plt.figure(figsize=(11.8, 11.2), constrained_layout=False)
    grid = fig.add_gridspec(
        7,
        3,
        height_ratios=(1, 0.08, 0.26, 1, 1, 1, 0.08),
        left=0.04,
        right=0.985,
        top=0.965,
        bottom=0.045,
        wspace=0.06,
        hspace=0.13,
    )
    for grid_row, column in ((1, 2), (2, 0), (2, 1), (2, 2), (6, 2)):
        fig.add_subplot(grid[grid_row, column]).axis("off")
    hcp_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    _hcp_row(hcp_axes, config["hcp"])
    internal_axes = []
    for index, row in enumerate(config["subjects"]):
        axes = [fig.add_subplot(grid[index + 3, column]) for column in range(3)]
        _subject_row(axes, row, titles=index == 0)
        internal_axes.append(axes)
    _colorbar(
        fig, fig.add_subplot(grid[1, 0]), HOT, 1, "Task activation (HCP-YA group beta)"
    )
    _colorbar(
        fig,
        fig.add_subplot(grid[1, 1]),
        "gray",
        0.4,
        "VENAT partial-volume fraction",
        zero_as_int=True,
    )
    effect_bar, vessel_bar = fig.add_subplot(grid[6, 0]), fig.add_subplot(grid[6, 1])
    _colorbar(fig, effect_bar, HOT, 3, "Task activation (fixed-effects effect)")
    _colorbar(fig, vessel_bar, "gray", FRANGI_MAX, "Frangi vesselness intensity")
    # Approved correction: adjacent endpoint labels remain within their bars.
    effect_bar.get_xticklabels()[-1].set_ha("right")
    vessel_bar.get_xticklabels()[0].set_ha("left")
    for ax, label in ((hcp_axes[0], "A"), (internal_axes[0][0], "B")):
        ax.text(
            -0.12,
            1.12,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
        )
    return fig


def render_figure_s2(config):
    fig = plt.figure(figsize=(10.8, 7.9), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(1, 1, 1, 0.08),
        left=0.07,
        right=0.985,
        top=0.91,
        bottom=0.08,
        wspace=0.06,
        hspace=0.10,
    )
    for index, row in enumerate(config["subjects"]):
        _subject_row(
            [fig.add_subplot(grid[index, column]) for column in range(3)],
            row,
            movie=True,
            titles=index == 0,
        )
    _colorbar(
        fig,
        fig.add_subplot(grid[3, 0]),
        HOT,
        0.8,
        "Movie face-presence effect",
        scale=1.30,
    )
    vessel_bar = fig.add_subplot(grid[3, 1:3])
    _colorbar(
        fig, vessel_bar, "gray", FRANGI_MAX, "Frangi vesselness intensity", scale=1.30
    )
    vessel_bar.get_xticklabels()[0].set_visible(False)
    return fig


def _fixed_color(data, low, high, color, alpha, gamma):
    normalized = np.clip((data - low) / max(high - low, np.finfo(float).eps), 0, 1)
    normalized[~np.isfinite(data)] = 0
    rgba = np.zeros((*data.shape, 4), np.float32)
    rgba[..., :3] = colors.to_rgb(color)
    rgba[..., 3] = alpha * normalized**gamma
    return rgba


def render_figure_s3(config):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=False)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.91, bottom=0.17, wspace=0.045)
    x, y, extent = world_grid(EXPANDED_XLIM, EXPANDED_YLIM)
    for ax, row in zip(axes, config["subjects"]):
        z = float(row["z"])
        t2_img, t2 = _load(row["t2w"])
        tof_img, tof = _load(row["tof"])
        frangi_img, frangi = _load(row["frangi"])
        amy_img = require_binary_mask(row["amygdala"])
        anatomy = sample_slice(t2_img, t2, x, y, z, order=1)
        arteries = sample_mip(tof_img, tof, x, y, z, width=8, order=1)
        veins = sample_mip(frangi_img, frangi, x, y, z, width=8, order=0)
        amy = sample_slice(
            amy_img, amy_img.get_fdata(dtype=np.float32), x, y, z, order=1
        )
        _format_axis(ax, row["label"], z, expanded=True)
        positive = anatomy[np.isfinite(anatomy) & (anatomy > 0)]
        low, high = np.percentile(positive, [2, 98])
        background = np.clip(
            (anatomy - low) / max(high - low, np.finfo(float).eps), 0, 1
        )
        ax.imshow(
            0.58 * background,
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.imshow(
            _fixed_color(veins, 0.025, 0.25, "#00BFFF", 0.92, 0.68),
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        positive = arteries[np.isfinite(arteries) & (arteries > 0)]
        if positive.size:
            low, high = np.percentile(positive, [94.5, 99.7])
            ax.imshow(
                _fixed_color(arteries, low, high, "#FF6A1A", 1, 0.38),
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
        _outline(ax, amy, extent)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#FF6A1A", lw=4, label="TOF arteriogram"),
            Line2D([0], [0], color="#00BFFF", lw=4, label="QSM venogram"),
            Line2D([0], [0], color=AMY_COLOR, lw=2, label="CIT168 amygdala"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.2,
        columnspacing=2,
    )
    return fig


RENDERERS = {
    "3": ("figure3", render_figure3),
    "S2": ("figureS2_movie_face_presence", render_figure_s2),
    "S3": ("figureS3_vascular_context", render_figure_s3),
}


def render(config: dict, figures: list[str], output_dir: Path, *, dpi: int = 300):
    """Write the selected panels and the resolved input manifest beside them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for figure_id in figures:
        stem, renderer = RENDERERS[figure_id]
        with plt.rc_context({"font.family": "DejaVu Sans"}):
            fig = renderer(config)
            fig.savefig(
                output_dir / f"{stem}.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            fig.savefig(
                output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white"
            )
            plt.close(fig)
    (output_dir / "vascular_inputs.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
