"""FAST's file contract must not depend on the caller's FSL output setting."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from amyvasc_release.gray_matter import _fast_gray_matter_probability


class FastOutputContractTests(unittest.TestCase):
    def test_uncompressed_parent_setting_still_returns_loaded_probability(self):
        probability = np.linspace(0, 1, 8, dtype=np.float32).reshape(2, 2, 2)

        def fake_fast(command, *, check, env):
            self.assertTrue(check)
            self.assertEqual(env["FSLOUTPUTTYPE"], "NIFTI_GZ")
            prefix = command[command.index("-o") + 1]
            nib.save(nib.Nifti1Image(probability, np.eye(4)), prefix + "_pve_1.nii.gz")

        with tempfile.TemporaryDirectory() as temporary:
            template = Path(temporary) / "template.nii.gz"
            with (
                patch.dict(os.environ, {"FSLOUTPUTTYPE": "NIFTI"}),
                patch("amyvasc_release.gray_matter.shutil.which", return_value="/fsl/fast"),
                patch("amyvasc_release.gray_matter.subprocess.run", side_effect=fake_fast),
            ):
                image = _fast_gray_matter_probability(template, "fast")
                self.assertEqual(os.environ["FSLOUTPUTTYPE"], "NIFTI")
            # The temporary FAST directory has gone; returned data must survive it.
            np.testing.assert_array_equal(image.get_fdata(), probability)


if __name__ == "__main__":
    unittest.main()
