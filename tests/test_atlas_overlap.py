"""Protect FSL label indexing and the binary atlas-overlap input contract."""

from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from amyvasc_release.atlas_overlap import (
    prepare_harvard_oxford_amygdala_mask,
    summarize_atlas_trace_overlap,
)


class AtlasOverlapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.xml = self.root / "labels.xml"
        self.xml.write_text(
            '<atlas><data><label index="2">Left Amygdala</label>'
            '<label index="6">Right Amygdala</label>'
            '<label index="9">Other structure</label></data></atlas>'
        )

    def save(self, name, values):
        path = self.root / name
        data = np.asarray(values, dtype=np.float32).reshape(2, 2, 2)
        nib.save(nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0])), path)
        return path

    def test_xml_indices_select_only_bilateral_amygdala(self):
        atlas = self.save("atlas.nii.gz", [0, 3, 7, 10, 10, 0, 0, 0])
        output = prepare_harvard_oxford_amygdala_mask(
            atlas, self.xml, self.root / "amygdala.nii.gz"
        )
        image = nib.load(output)
        np.testing.assert_array_equal(
            image.get_fdata().ravel(), [0, 1, 1, 0, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(image.affine, nib.load(atlas).affine)

    def test_missing_amygdala_label_fails(self):
        self.xml.write_text('<atlas><label index="2">Left Amygdala</label></atlas>')
        atlas = self.save("atlas.nii.gz", [3, 0, 0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "Left Amygdala and Right Amygdala"):
            prepare_harvard_oxford_amygdala_mask(
                atlas, self.xml, self.root / "amygdala.nii.gz"
            )

    def test_overlap_rejects_label_images_in_every_binary_mask_input(self):
        trace = self.save("trace.nii.gz", [1, 1, 1, 0, 0, 0, 0, 0])
        primary = self.save("primary.nii.gz", [0, 1, 1, 0, 0, 0, 0, 0])
        cit = self.save("cit.nii.gz", [1, 0, 0, 0, 0, 0, 0, 0])
        atlas = self.save("atlas.nii.gz", [3, 7, 10, 0, 0, 0, 0, 0])
        ho = self.save("ho.nii.gz", [1, 1, 1, 0, 0, 0, 0, 0])
        inputs = dict(
            pre_exclusion_trace_path=trace,
            primary_trace_path=primary,
            cit168_p50_path=cit,
            harvard_oxford_maxprob50_path=ho,
            harvard_oxford_maxprob25_path=ho,
        )
        for name in inputs:
            with self.subTest(input=name):
                with self.assertRaisesRegex(ValueError, "binary 0/1 mask"):
                    summarize_atlas_trace_overlap(**{**inputs, name: atlas})


if __name__ == "__main__":
    unittest.main()
