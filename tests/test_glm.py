"""Analytic fixed-effects regression through the real Nilearn implementation."""

from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from scipy.stats import norm, t

from amyvasc_release.glm import RunResult, combine_runs


class FixedEffectsTests(unittest.TestCase):
    def test_precision_weighting_pooled_dof_and_output_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.zeros((2, 2, 2), dtype=np.uint8)
            mask[0, 0, 0] = mask[1, 1, 1] = 1
            mask_img = nib.Nifti1Image(mask, np.eye(4))
            runs = []
            for index, (effect, variance, dof) in enumerate(
                ((2.0, 4.0, 10), (5.0, 1.0, 30))
            ):
                run = RunResult(f"run-{index + 1}", root / f"run-{index + 1}", dof)
                effect_map = np.zeros(mask.shape, dtype=np.float32)
                effect_map[0, 0, 0] = effect
                effect_map[1, 1, 1] = -effect
                variance_map = np.full(mask.shape, variance, dtype=np.float32)
                for kind, values in (("effect", effect_map), ("variance", variance_map)):
                    path = run.contrast_path("known", kind)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(nib.Nifti1Image(values, np.eye(4)), path)
                runs.append(run)

            output = root / "fixed_effects"
            combine_runs(run_results=runs, mask_img=mask_img, output_dir=output,
                         contrast_stems=("known",))

            # Precision weights are 1/4 and 1. The pooled variance is 1/(1/4+1),
            # effect is (2/4+5)/(1/4+1), and signed z uses the pooled residual DOF.
            expected_stat = 4.4 / np.sqrt(0.8)
            expected_z = norm.isf(t.sf(expected_stat, df=40))
            expected = {
                "effect": [4.4, -4.4],
                "variance": [0.8, 0.8],
                "stat": [expected_stat, -expected_stat],
                "z": [expected_z, -expected_z],
            }
            for kind, values in expected.items():
                image = nib.load(output / "contrasts" / f"known_{kind}.nii.gz")
                data = image.get_fdata()
                np.testing.assert_allclose(data[mask > 0], values, rtol=1e-6)
                self.assertTrue(np.isnan(data[mask == 0]).all())
                np.testing.assert_array_equal(image.affine, mask_img.affine)
            np.testing.assert_array_equal(
                nib.load(output / "support_mask.nii.gz").get_fdata(), mask
            )
            metadata = json.loads((output / "fixed_effects_metadata.json").read_text())
            self.assertEqual(metadata["fixed_effects_dof"], 40)
            self.assertEqual(metadata["degrees_of_freedom"], [10, 30])
            self.assertTrue(metadata["precision_weighted"])
            self.assertEqual(metadata["software_versions"]["nilearn"], version("nilearn"))


if __name__ == "__main__":
    unittest.main()
