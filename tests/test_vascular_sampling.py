"""Check slice orientation and projection geometry used by anatomical panels."""

import unittest

import nibabel as nib
import numpy as np

from figures.vascular import native_slice, sample_mip, sample_slice


class VascularSamplingTests(unittest.TestCase):
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
        self.assertEqual(extent, (-2, 2, -2, 2))


if __name__ == "__main__":
    unittest.main()
