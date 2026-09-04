"""Synthetic checks of image orientation and projection coordinates."""

import unittest
from unittest.mock import Mock

import nibabel as nib
import numpy as np

from figures.hcp_maps import draw, slice_or_mip


class HCPMapGeometryTests(unittest.TestCase):
    def setUp(self):
        self.data = np.indices((4, 3, 5))[0].astype(float)
        self.image = nib.Nifti1Image(self.data, np.diag([2.0, 2.0, 2.0, 1.0]))

    def test_axial_slice_keeps_increasing_world_x_left_to_right(self):
        plane, extent = slice_or_mip(self.data, self.image, 2, 2)
        np.testing.assert_array_equal(plane[0], [0, 1, 2, 3])
        self.assertEqual(extent, [-1, 7, -1, 5])

    def test_sagittal_mip_includes_both_slab_endpoints(self):
        plane, extent = slice_or_mip(self.data, self.image, 0, 2, (2, 4))
        np.testing.assert_array_equal(plane, np.full((5, 3), 2))
        self.assertEqual(extent, [-1, 5, -1, 9])

    def test_outside_slab_fails(self):
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            slice_or_mip(self.data, self.image, 0, 2, (20, 30))

    def test_amygdala_contour_is_only_drawn_on_slices(self):
        amygdala = self.data > 1
        slice_ax = Mock()
        draw(slice_ax, self.image, self.data, amygdala, coordinate=2)
        slice_ax.contour.assert_called_once()
        mip_ax = Mock()
        draw(mip_ax, self.image, self.data, amygdala, coordinate=2, slab=(0, 4))
        mip_ax.contour.assert_not_called()

    def test_explicit_segment_contours_are_retained_on_mips(self):
        ax = Mock()
        draw(
            ax,
            self.image,
            self.data,
            self.data > 1,
            coordinate=2,
            slab=(0, 4),
            contours=[(self.data > 0, "red", "solid")],
        )
        ax.contour.assert_called_once()


if __name__ == "__main__":
    unittest.main()
