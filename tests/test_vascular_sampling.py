"""Check slice orientation and projection geometry used by anatomical panels."""

import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from figures.vascular import (
    _layer,
    _outline,
    _subject_row,
    native_slice,
    plt,
    sample_mip,
    sample_slice,
    render_figure_s3,
    world_grid,
)


class VascularSamplingTests(unittest.TestCase):
    def test_subject_figures_keep_the_same_native_mask_at_an_off_grid_plane(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = (-6, -6, -2)
        data = np.ones((7, 7, 3), dtype=np.float32)
        anatomy = nib.Nifti1Image(data, affine)
        mask = np.zeros_like(data)
        mask[3, 3, 1] = 1  # z=0: a single voxel centered at x=y=0.
        mask[2, 2, 2] = 1  # z=2 differs, so interpolation would alter the contour.
        mask_img = nib.Nifti1Image(mask, affine)
        row = dict(
            label="subject",
            z=0.6,
            emotion_effect="emotion",
            movie_effect="movie",
            frangi="frangi",
            amygdala="amygdala",
            t2w="t2w",
            tof="tof",
        )
        expected = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
        with patch("figures.vascular._load", return_value=(anatomy, data)), patch(
            "figures.vascular.require_binary_mask", return_value=mask_img
        ):
            for movie in (False, True):
                fig, axes = plt.subplots(1, 3)
                self.addCleanup(plt.close, fig)
                _subject_row(axes, row, movie=movie)
                for ax in axes:
                    vertices = ax.collections[0].get_paths()[0].vertices
                    np.testing.assert_array_equal(np.unique(vertices, axis=0), expected)
            fig = render_figure_s3({"subjects": [row] * 3})
            self.addCleanup(plt.close, fig)
            for ax in fig.axes:
                vertices = ax.collections[0].get_paths()[0].vertices
                np.testing.assert_array_equal(np.unique(vertices, axis=0), expected)

    def test_world_slice_and_centered_mip_have_the_expected_voxels(self):
        ijk = np.indices((7, 7, 7))
        data = (100 * ijk[0] + 10 * ijk[1] + ijk[2]).astype(np.float32)
        affine = np.eye(4)
        affine[:3, 3] = -3
        image = nib.Nifti1Image(data, affine)
        coords = np.array([-1.0, 0.0, 1.0])
        plane = sample_slice(image, data, coords, coords, 0, order=0)
        np.testing.assert_array_equal(plane, data[2:5, 2:5, 3].T)
        projection = sample_mip(image, data, coords, coords, 0, width=2, order=0)
        np.testing.assert_array_equal(projection, data[2:5, 2:5, 4].T)

    def test_negative_voxel_axis_is_sorted_into_neurological_world_coordinates(self):
        data = np.arange(3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3)
        affine = np.diag([-2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = (2, -2, -2)
        image = nib.Nifti1Image(data, affine)
        plane, extent = native_slice(image, data, 0)
        np.testing.assert_array_equal(plane, data[::-1, :, 1].T)
        self.assertEqual(extent, (-3, 3, -3, 3))

    def test_rendered_hcp_pixel_centers_match_world_sampled_coordinates(self):
        affine = np.array(
            [[-2, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]]
        )
        data = np.arange(91 * 109 * 91, dtype=np.float32).reshape(91, 109, 91)
        image = nib.Nifti1Image(data, affine)
        plane, extent = native_slice(image, data, -12)
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        _layer(ax, plane, extent, cmap="gray", vmax=data.max())
        artist = ax.images[0]
        left, right, bottom, top = artist.get_extent()
        rows, columns = artist.get_array().shape
        x = left + (np.arange(columns) + 0.5) * (right - left) / columns
        y = bottom + (np.arange(rows) + 0.5) * (top - bottom) / rows
        np.testing.assert_allclose(x, np.arange(-42, 43, 2), atol=1e-12)
        np.testing.assert_allclose(y, np.arange(-36, 15, 2), atol=1e-12)
        np.testing.assert_array_equal(
            artist.get_array(), sample_slice(image, data, x, y, -12, order=0)
        )

    def test_native_contour_aligns_with_world_interpolated_boundary(self):
        affine = np.diag([-2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = (6, -6, -6)
        x_native = 6 - 2 * np.arange(7)
        data = np.broadcast_to((x_native >= 2)[:, None, None], (7, 7, 7)).astype(
            np.float32
        )
        image = nib.Nifti1Image(data, affine)
        native, native_extent = native_slice(image, data, 0)
        x, y, world_extent = world_grid((-6, 6), (-6, 6))
        sampled = sample_slice(image, data, x, y, 0, order=1)
        fig, axes = plt.subplots(1, 2)
        self.addCleanup(plt.close, fig)
        for ax, plane, extent in (
            (axes[0], native, native_extent),
            (axes[1], sampled, world_extent),
        ):
            _outline(ax, plane, extent)
            vertices = ax.collections[0].get_paths()[0].vertices
            # The 0.50 boundary is halfway between native centers x=0 and x=2.
            np.testing.assert_allclose(vertices[:, 0], 1.0, atol=1e-12)
            np.testing.assert_allclose(
                [vertices[:, 1].min(), vertices[:, 1].max()], [-6, 6]
            )


if __name__ == "__main__":
    unittest.main()
