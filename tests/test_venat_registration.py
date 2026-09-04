"""Check template direction, output grid, and the overlap QC gate without ANTs."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from amyvasc_release.venat import prepare_venat


def image(value):
    array = np.full((2, 2, 2), value, dtype=float)
    return SimpleNamespace(
        numpy=lambda: array, dimension=3, shape=array.shape, spacing=(1, 1, 1)
    )


class VenatRegistrationTests(unittest.TestCase):
    def inputs(self, directory):
        paths, images = {}, {}
        for name in (
            "venat",
            "fixed_template",
            "moving_template",
            "fixed_mask",
            "moving_mask",
        ):
            paths[name] = directory / f"{name}.nii.gz"
            paths[name].touch()
            images[str(paths[name])] = image(1)
        paths["output_dir"] = directory / "output"
        forward = [str(paths["output_dir"] / "forward.nii.gz")]
        ants = SimpleNamespace(
            __version__="mock",
            image_read=Mock(side_effect=images.__getitem__),
            image_physical_space_consistency=Mock(return_value=True),
            registration=Mock(
                return_value={"fwdtransforms": forward, "invtransforms": []}
            ),
            resample_image_to_target=Mock(return_value=image(0)),
            apply_transforms=Mock(return_value=image(1)),
            resample_image=Mock(return_value=image(1)),
            image_write=Mock(),
        )
        return paths, images, ants, forward

    def test_forward_transform_and_half_mm_fixed_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths, images, ants, forward = self.inputs(Path(temporary))
            with patch.dict(sys.modules, {"ants": ants}):
                destination = prepare_venat(**paths)
            registration = ants.registration.call_args.kwargs
            self.assertIs(registration["fixed"], images[str(paths["fixed_template"])])
            self.assertIs(registration["moving"], images[str(paths["moving_template"])])
            self.assertEqual(registration["type_of_transform"], "SyN")
            ants.resample_image.assert_called_once_with(
                registration["fixed"], (0.5, 0.5, 0.5), use_voxels=False, interp_type=4
            )
            atlas_warp = ants.apply_transforms.call_args.kwargs
            self.assertIs(atlas_warp["fixed"], ants.resample_image.return_value)
            self.assertIs(atlas_warp["moving"], images[str(paths["venat"])])
            self.assertEqual(atlas_warp["transformlist"], forward)
            self.assertEqual(atlas_warp["interpolator"], "linear")
            self.assertEqual(
                destination.name,
                "VENAT_PartialVolume_space-MNI152NLin6Asym_res-0p5.nii.gz",
            )
            self.assertTrue((paths["output_dir"] / "provenance.json").is_file())

    def test_invalid_grid_or_unimproved_overlap_does_not_write_atlas(self):
        for failure in ("grid", "overlap"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as temporary,
            ):
                paths, _, ants, _ = self.inputs(Path(temporary))
                if failure == "grid":
                    ants.image_physical_space_consistency.return_value = False
                else:
                    ants.resample_image_to_target.return_value = image(1)
                with (
                    patch.dict(sys.modules, {"ants": ants}),
                    self.assertRaises(ValueError),
                ):
                    prepare_venat(**paths)
                ants.image_write.assert_not_called()
                ants.resample_image.assert_not_called()
                if failure == "grid":
                    ants.registration.assert_not_called()


if __name__ == "__main__":
    unittest.main()
