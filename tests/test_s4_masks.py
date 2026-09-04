"""Figure S4 target contracts using synthetic label images."""

import unittest

import numpy as np

from amyvasc_release.masks import build_s4_mesencephalic_target


class S4TargetTests(unittest.TestCase):
    def setUp(self):
        self.bvr = np.zeros((2, 2, 2), dtype=int)
        self.bvr[0, 0, :] = 5
        self.bvr[1, 0, :] = 6
        self.bvr[0, 1, 0] = 1
        self.tian = np.zeros_like(self.bvr)
        self.tian[0, 0, 0] = 2
        self.tian[1, 0, 0] = 3
        self.labels = ["other", "THA-DP-lh", "THA-DP-rh"]

    def test_removes_bilateral_thalamus_and_reports_local_counts(self):
        raw, target, counts = build_s4_mesencephalic_target(
            self.bvr, self.tian, self.labels
        )
        self.assertEqual(
            np.argwhere(raw).tolist(), [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]
        )
        self.assertEqual(np.argwhere(target).tolist(), [[0, 0, 1], [1, 0, 1]])
        self.assertEqual(counts.raw_mesencephalic, 4)
        self.assertEqual(counts.thadp_overlap, 2)
        self.assertEqual(counts.final_target, 2)

    def test_zero_overlap_is_valid(self):
        raw, target, counts = build_s4_mesencephalic_target(
            self.bvr, np.zeros_like(self.tian), self.labels
        )
        np.testing.assert_array_equal(raw, target)
        self.assertEqual(counts.thadp_overlap, 0)

    def test_empty_input_or_excluded_target_is_rejected(self):
        for bvr, tian in (
            (np.zeros_like(self.bvr), self.tian),
            (self.bvr, np.where(self.bvr >= 5, 2, 0)),
        ):
            with self.subTest(bvr=bvr.tolist(), tian=tian.tolist()):
                with self.assertRaisesRegex(ValueError, "Empty mesencephalic BVR"):
                    build_s4_mesencephalic_target(bvr, tian, self.labels)


if __name__ == "__main__":
    unittest.main()
