"""Scientific contracts for ROI extraction and missing participant support."""

from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

from amyvasc_release.source_profiles import (
    CONDITIONS,
    CandidateSpec,
    TargetSpec,
    analyze_profile,
    extract_profiles,
    unrestricted_profile_column,
)


class ProfileExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (5, 2, 2)
        self.affine = np.eye(4)
        self.affine[0, 3] = -2
        self.target_masks = {}
        first = np.zeros(self.shape, bool)
        first[[0, 3], 0, :] = True
        first[[1, 4], 0, 0] = True
        second = np.zeros(self.shape, bool)
        second[[1, 4], 0, 1] = True
        self.target_masks = {"target": first, "second": second}
        self.targets = [
            TargetSpec(key, key, self.save(key, mask))
            for key, mask in self.target_masks.items()
        ]
        amy = np.zeros(self.shape, bool)
        amy[:, 1, :] = True
        overlap = np.zeros(self.shape, bool)
        overlap[:, 0, :] = True
        self.candidate_masks = {"amy_proper": amy, "overlap": overlap}
        self.candidates = [
            CandidateSpec(key, key, self.save(key, mask))
            for key, mask in self.candidate_masks.items()
        ]
        self.gm = np.ones(self.shape, bool)
        self.gm[0, 1, 0] = False
        self.gm[0, 0, 1] = False  # Targets themselves are not GM-restricted.
        self.gm_path = self.save("gm", self.gm)
        self.subjects = ["one", "two"]
        self.support = {}
        self.data = {}
        for participant, subject in enumerate(self.subjects):
            common = np.ones(self.shape, bool)
            for task in dict.fromkeys(condition.task for condition in CONDITIONS):
                support = np.ones(self.shape, bool)
                if participant == 1 and task == "motor":
                    support[1, :, :] = False
                    support[4, 0, 1] = False
                common &= support
                self.save(
                    f"effects/{task}/fixed_effects/sub-{subject}/support_mask",
                    support,
                )
            self.support[subject] = common
            for index, condition in enumerate(CONDITIONS):
                data = (
                    np.arange(np.prod(self.shape), dtype=np.float32).reshape(
                        self.shape
                    ) ** 2 / 17 + index / 10 + participant
                )
                data[0, 0, 0] = np.nan
                data[3, 1, 0] = np.inf
                self.data[subject, condition.key] = data
                self.save(
                    f"effects/{condition.task}/fixed_effects/sub-{subject}/"
                    f"contrasts/{condition.effect}_effect",
                    data,
                )

    def save(self, name, values):
        path = self.root / f"{name}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(values.astype(np.float32), self.affine), path)
        return path

    def extract(self, **kwargs):
        return extract_profiles(
            subjects=self.subjects,
            fixed_effects_root=self.root / "effects",
            targets=self.targets,
            candidates=self.candidates,
            gray_matter_support_path=self.gm_path,
            unrestricted_profile_keys=("overlap",),
            **kwargs,
        )

    def assert_boolean_mask_oracle(self, tables, statistic):
        # This oracle indexes full-volume boolean masks directly. It exercises
        # target overlap, GM support, common-task support, midline and NaN/Inf
        # handling independently of the extractor's flattened index storage.
        x = np.indices(self.shape)[0] - 2
        reducer = np.median if statistic == "median" else np.mean
        for target in self.targets:
            for row in tables[target.key].to_dict("records"):
                subject = self.subjects[row["participant"] - 1]
                hemi = x < 0 if row["hemisphere"] == "left" else x > 0
                support = self.support[subject] & hemi
                masks = {target.key: self.target_masks[target.key] & support}
                masks.update({
                    key: mask & self.gm & ~self.target_masks[target.key] & support
                    for key, mask in self.candidate_masks.items()
                })
                masks[unrestricted_profile_column("overlap")] = (
                    self.candidate_masks["overlap"] & support
                )
                for key, mask in masks.items():
                    values = self.data[subject, row["condition_key"]][mask]
                    finite = values[np.isfinite(values)]
                    expected = float(reducer(finite)) if finite.size else np.nan
                    np.testing.assert_equal(row[key], expected)
                self.assertEqual(row["voxel_statistic"], statistic)

    def test_default_medians_match_boolean_mask_extraction_exactly(self):
        tables, inventory, _ = self.extract()
        self.assert_boolean_mask_oracle(tables, "median")
        self.assertTrue(inventory["voxel_statistic"].eq("median").all())
        for row in inventory.to_dict("records"):
            hemi = (np.indices(self.shape)[0] - 2) * (
                -1 if row["hemisphere"] == "left" else 1
            ) > 0
            expected = (
                self.candidate_masks[row["candidate_key"]]
                & self.gm & hemi & ~self.target_masks[row["target_key"]]
            ).sum()
            self.assertEqual(row["retained_voxels"], expected)

    def test_means_match_boolean_mask_extraction_and_differ_from_medians(self):
        means, _, _ = self.extract(statistic="mean")
        medians, _, _ = self.extract()
        self.assert_boolean_mask_oracle(means, "mean")
        self.assertFalse(np.array_equal(
            means["target"]["amy_proper"], medians["target"]["amy_proper"]
        ))

    def test_unknown_statistic_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "statistic"):
            self.extract(statistic="mode")


class ProfileAnalysisTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(47)
        self.values = rng.normal(size=(4, len(CONDITIONS) * 2, 3))
        self.values[0, 0, 0] = np.nan  # Missing target and candidate differ.
        self.values[1, 0, 2] = np.nan
        self.values[:2, 1, 2] = np.nan
        self.values[:3, 2, 2] = np.nan
        self.values[0, 3, 1] = np.nan
        self.target = TargetSpec("target", "Target", Path("unused"))
        self.candidates = [
            CandidateSpec("amy_proper", "Amygdala"),
            CandidateSpec("rival", "Rival"),
            CandidateSpec("composite", "Composite", members=("amy_proper", "rival"),
                          ranked=False),
        ]
        rows = []
        for participant in range(4):
            for index, condition in enumerate(CONDITIONS):
                for hemi_index, hemisphere in enumerate(("left", "right")):
                    values = self.values[participant, 2 * index + hemi_index]
                    rows.append({
                        "participant": participant,
                        "task": condition.task,
                        "condition_key": condition.key,
                        "hemisphere": hemisphere,
                        "target": values[0],
                        "amy_proper": values[1],
                        "rival": values[2],
                    })
        self.wide = pd.DataFrame(rows)

    def test_cohort_and_paired_support_counts_include_target_availability(self):
        result = analyze_profile(self.wide, self.target, self.candidates,
                                 n_bootstrap=100, seed=73)
        table = result["correlations"].set_index("candidate_key")
        self.assertNotIn("n_participants", table.columns)
        self.assertTrue(table["n_cohort"].eq(4).all())
        self.assertTrue(result["paired"]["n_cohort"].eq(4).all())
        for key, column in (("amy_proper", 1), ("rival", 2)):
            counts = (np.isfinite(self.values[:, :, 0])
                      & np.isfinite(self.values[:, :, column])).sum(axis=0)
            for suffix, expected in (("min", counts.min()),
                                     ("median", np.median(counts)),
                                     ("max", counts.max())):
                self.assertEqual(table.loc[key, f"n_paired_cell_participants_{suffix}"],
                                 expected)
        self.assertEqual(table.loc["rival", "n_paired_cell_participants_min"], 1)
        self.assertEqual(table.loc["rival", "n_paired_cell_participants_max"], 4)
        self.assertEqual(table.loc["composite", "n_paired_cell_participants_min"], 3)

    def test_bootstrap_still_resamples_full_cohort_with_cellwise_means(self):
        result = analyze_profile(self.wide, self.target, self.candidates,
                                 n_bootstrap=100, seed=73)
        table = result["correlations"].set_index("candidate_key")
        tasks = [condition.task for condition in CONDITIONS for _ in range(2)]
        weights = np.asarray([1 / tasks.count(task) for task in tasks])
        weights /= weights.sum()

        def correlation(x, y):
            keep = np.isfinite(x) & np.isfinite(y)
            w = weights[keep] / weights[keep].sum()
            xc = x[keep] - np.dot(w, x[keep])
            yc = y[keep] - np.dot(w, y[keep])
            return np.dot(w, xc * yc) / np.sqrt(
                np.dot(w, xc * xc) * np.dot(w, yc * yc)
            )

        sampled = np.random.default_rng(73).integers(0, 4, size=(100, 4))
        with np.errstate(invalid="ignore"):
            # Some resamples have no support for one candidate cell.
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Mean of empty slice")
                draws = [np.nanmean(self.values[indices], axis=0)
                         for indices in sampled]
        group = np.nanmean(self.values, axis=0)
        for key, column in (("amy_proper", 1), ("rival", 2)):
            self.assertAlmostEqual(table.loc[key, "group_r"],
                                   correlation(group[:, 0], group[:, column]))
            expected = np.quantile(
                [correlation(draw[:, 0], draw[:, column]) for draw in draws],
                [0.025, 0.5, 0.975],
            )
            np.testing.assert_allclose(
                table.loc[key, ["bootstrap_q025", "bootstrap_median",
                                "bootstrap_q975"]].to_numpy(float),
                expected, atol=1e-14,
            )


if __name__ == "__main__":
    unittest.main()
