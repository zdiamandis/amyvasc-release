"""Scientific contracts for Figure 7D, using synthetic arrays only."""

import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from amyvasc_release.source_profiles import modeled_blur, normalize_blur_prediction


class ModeledBlurTests(unittest.TestCase):
    def test_reference_uses_participant_medians_and_separate_hemispheres(self):
        # Left participant medians are .5 and 1; right medians are both .2.
        result = normalize_blur_prediction(
            np.array([0.6, 0.1]),
            np.array(["left", "right"]),
            np.array([0.2, 0.8, 1.0, 0.2]),
            np.array(["left", "left", "left", "right"]),
            np.array([[True, True, False, True], [False, False, True, True]]),
        )
        np.testing.assert_allclose(result, [0.8, 0.5])

    def test_missing_participant_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty anatomical-amygdala"):
            normalize_blur_prediction(
                np.array([0.1]),
                np.array(["left"]),
                np.array([0.5, 0.5]),
                np.array(["left", "right"]),
                np.array([[True, True], [False, True]]),
            )

    def test_normalize_each_kernel_then_maximize_each_voxel(self):
        shape = (9, 1, 1)
        probability = np.zeros(shape)
        probability[[0, 8], 0, 0] = 1
        reference = nib.Nifti1Image(probability, np.diag([2.0, 2.0, 2.0, 1.0]))
        kernels = []
        for values in ([0.8, 0.4, 0.1, 0.4], [0.2, 0.06, 0.08, 0.1]):
            image = np.zeros(shape)
            image[[0, 1, 7, 8], 0, 0] = values
            kernels.append(image)
        profile = np.arange(23, dtype=float)
        observed = np.stack([2 * profile, 3 * profile], axis=-1)[None, ...]
        with patch(
            "amyvasc_release.source_profiles.gaussian_filter", side_effect=kernels
        ):
            # Empty synthetic shells are irrelevant to the populated first shell.
            with np.errstate(invalid="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = modeled_blur(
                        voxel_profiles=observed,
                        voxel_ijk=np.array([[1, 0, 0], [7, 0, 0]]),
                        voxel_hemispheres=np.array(["left", "right"]),
                        amygdala_profiles={"left": profile, "right": profile},
                        amygdala_probability=probability,
                        reference_ijk=np.array([[0, 0, 0], [8, 0, 0]]),
                        reference_hemispheres=np.array(["left", "right"]),
                        reference_support=np.ones((1, 2), dtype=bool),
                        reference=reference,
                        fwhm_mm=(2, 4),
                    )
        shell = result.iloc[0]
        self.assertEqual(shell.n_voxels, 2)
        self.assertAlmostEqual(shell.observed_amygdala_profile_amplitude, 2.5)
        self.assertAlmostEqual(shell.predicted_2mm, 0.375)
        self.assertAlmostEqual(shell.predicted_4mm, 0.55)
        self.assertAlmostEqual(shell.maximum_modeled_blur, 0.65)


if __name__ == "__main__":
    unittest.main()
