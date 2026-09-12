"""Fixed voxel boxes must retain their intended anatomy and hemispheres."""

import unittest

import nibabel as nib
import numpy as np

from amyvasc_release.masks import (
    BVR_BOXES,
    BVR_GRID_AFFINE,
    BVR_GRID_SHAPE,
    build_bvr_label_image,
)


class BVRGridTests(unittest.TestCase):
    def setUp(self):
        self.data = np.zeros(BVR_GRID_SHAPE, dtype=np.float32)
        self.expected = np.zeros(BVR_GRID_SHAPE, dtype=np.int16)
        for label, (i, _, j, _, k, _) in BVR_BOXES.items():
            self.data[i, j, k] = 0.51
            self.data[i + 1, j, k] = 0.50
            self.data[i + 2, j, k] = np.nan
            self.expected[i, j, k] = label
        self.image = nib.Nifti1Image(self.data, np.array(BVR_GRID_AFFINE))

    def test_native_grid_preserves_strict_threshold_and_hemisphere_labels(self):
        labels = build_bvr_label_image(self.image)
        np.testing.assert_array_equal(labels.get_fdata(), self.expected)
        np.testing.assert_array_equal(labels.affine, self.image.affine)
        for label in BVR_BOXES:
            world = nib.affines.apply_affine(
                labels.affine, np.argwhere(labels.get_fdata() == label)
            )
            self.assertTrue(
                np.all(world[:, 0] < 0) if label % 2 else np.all(world[:, 0] > 0)
            )

    def test_same_physical_image_in_canonical_order_is_rejected(self):
        canonical = nib.as_closest_canonical(self.image)
        np.testing.assert_array_equal(canonical.get_fdata()[::-1], self.data)
        with self.assertRaisesRegex(ValueError, "native HCP.*LAS affine"):
            build_bvr_label_image(canonical)

    def test_changed_origin_is_rejected_even_when_shape_and_voxel_sizes_match(self):
        affine = self.image.affine.copy()
        affine[0, 3] += 2
        with self.assertRaisesRegex(
            ValueError, "restore its native voxel order and affine together"
        ):
            build_bvr_label_image(nib.Nifti1Image(self.data, affine))


if __name__ == "__main__":
    unittest.main()
