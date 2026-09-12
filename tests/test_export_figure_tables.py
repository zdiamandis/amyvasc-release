"""Scientific export boundaries: paired observations, support, and analysis role."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "export_figure_tables", Path(__file__).resolve().parents[1] / "analysis/10_export_figure_tables.py"
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ParticipantLagExportTests(unittest.TestCase):
    def setUp(self):
        # Deliberately unsorted controlled identifiers, mixed legacy segment spelling,
        # and negative observations. Sensitivity rows must never enter primary panels.
        self.lags = pd.DataFrame(
            [
                ("377701", "peduncular", "pseg50", 950),
                ("991337", "striate", "pseg50", -20),
                ("100123", "peduncular", "pseg50", -40),
                ("377701", "striate", "pseg50", 1000),
                ("991337", "peduncular", "pseg50", 200),
                ("100123", "striatal", "pseg50", 500),
                ("000001", "striate", "pseg05", 1e9),
                ("000001", "peduncular", "pseg05", -1e9),
            ],
            columns=["participant", "segment", "exclusion", "target_minus_amy_ms"],
        )
        self.gradients = pd.DataFrame(
            [
                ("377701", "combined", "pseg50", -3),
                ("991337", "combined", "pseg50", 11),
                ("100123", "combined", "pseg50", np.nan),
                ("000001", "combined", "pseg05", 1e9),
                ("991337", "striate", "pseg50", -1e9),
            ],
            columns=["participant", "segment", "exclusion", "slope_ms_per_mm"],
        )

    def test_primary_pairing_is_preserved_with_shared_anonymous_display_ids(self):
        ordering, slopes = EXPORT.figure6_participant_tables(self.lags, self.gradients)
        self.assertEqual(
            list(ordering.columns),
            ["display_id", "amygdala_lag_ms", "striate_lag_ms", "peduncular_lag_ms"],
        )
        np.testing.assert_array_equal(
            ordering.to_numpy(),
            [[1, 0, 500, -40], [2, 0, 1000, 950], [3, 0, -20, 200]],
        )
        self.assertEqual(list(slopes.columns), ["display_id", "slope_ms_per_mm"])
        np.testing.assert_array_equal(slopes.to_numpy(), [[2, -3], [3, 11]])
        self.assertFalse(set(ordering.display_id.astype(str)) & set(self.lags.participant))
        self.assertFalse(set(slopes.display_id.astype(str)) & set(self.gradients.participant))

    def test_row_order_does_not_change_pairing_or_display_mapping(self):
        expected = EXPORT.figure6_participant_tables(self.lags, self.gradients)
        observed = EXPORT.figure6_participant_tables(
            self.lags.iloc[::-1], self.gradients.iloc[::-1]
        )
        for actual, reference in zip(observed, expected, strict=True):
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), reference.reset_index(drop=True))

    def test_missing_and_nonfinite_primary_pairs_are_rejected(self):
        missing = self.lags.drop(index=2)
        nonfinite = self.lags.copy()
        nonfinite.loc[2, "target_minus_amy_ms"] = np.inf
        for frame in (missing, nonfinite):
            with self.subTest(missing_rows=len(frame) < len(self.lags)):
                with self.assertRaisesRegex(ValueError, "finite paired"):
                    EXPORT.figure6_participant_tables(frame, self.gradients)

    def test_duplicate_primary_segment_is_rejected_even_with_legacy_spelling(self):
        duplicate = self.lags.iloc[[1]].copy()
        duplicate["segment"] = "striatal"
        with self.assertRaisesRegex(ValueError, "one lag per participant"):
            EXPORT.figure6_participant_tables(
                pd.concat([self.lags, duplicate], ignore_index=True), self.gradients
            )

    def test_duplicate_or_absent_finite_combined_slopes_are_rejected(self):
        duplicate = pd.concat([self.gradients, self.gradients.iloc[[1]]], ignore_index=True)
        absent = self.gradients.copy()
        absent.loc[absent.exclusion.eq("pseg50"), "slope_ms_per_mm"] = np.nan
        for gradients in (duplicate, absent):
            with self.subTest(rows=len(gradients)):
                with self.assertRaisesRegex(ValueError, "one finite combined slope"):
                    EXPORT.figure6_participant_tables(self.lags, gradients)


class SourceProfileExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "export"

    def metadata(self, statistic):
        (self.root / "run_metadata.json").write_text(
            json.dumps({"voxel_statistic": statistic})
        )

    def test_primary_exports_reject_mean_metadata_before_writing(self):
        self.metadata("mean")
        for function, arguments in (
            (EXPORT.export_figure7, (self.root, self.out, None)),
            (EXPORT.export_s5, (self.root, self.out)),
            (EXPORT.export_s4_controls, (self.root, self.out)),
            (EXPORT.export_s4_rankings, (self.root, self.out, None)),
        ):
            with self.subTest(export=function.__name__):
                with self.assertRaisesRegex(ValueError, "median-based"):
                    function(*arguments)
                self.assertFalse(self.out.exists())

    def test_mean_export_rejects_median_metadata_and_missing_provenance(self):
        with self.assertRaises(FileNotFoundError):
            EXPORT.require_profile_statistic(self.root, "mean")
        self.metadata("median")
        EXPORT.require_profile_statistic(self.root, "median")
        with self.assertRaisesRegex(ValueError, "mean-based"):
            EXPORT.export_mean_sensitivity(self.root, self.out)
        self.assertFalse(self.out.exists())

    def test_mean_sensitivity_preserves_support_without_subset_rank_claims(self):
        self.metadata("mean")
        rows = []
        for target, threshold in (("primary", "pseg50"), ("strict", "pseg05")):
            for candidate, support, correlation, rank in (
                ("amy_proper", (4, 7, 10), 0.8, 1),
                ("glasser__pir", (2, 5, 9), 0.7, 2),
            ):
                rows.append(
                    {
                        "target_key": target,
                        "target_label": target,
                        "segment": "combined",
                        "exclusion": threshold,
                        "candidate_key": candidate,
                        "candidate_label": candidate,
                        "group_r": correlation,
                        "bootstrap_median": correlation - 0.01,
                        "bootstrap_q025": correlation - 0.1,
                        "bootstrap_q975": correlation + 0.1,
                        "n_cohort": 10,
                        "n_paired_cell_participants_min": support[0],
                        "n_paired_cell_participants_median": support[1],
                        "n_paired_cell_participants_max": support[2],
                        "n_conditions": 23,
                        "n_tasks": 7,
                        "ranked": True,
                        "rank": rank,
                        "bootstrap_probability_rank_1": float(rank == 1),
                        "voxel_statistic": "mean",
                        "voxel_statistic_role": "sensitivity",
                    }
                )
        pd.DataFrame(rows).to_csv(
            self.root / "source_profile_correlations.tsv", sep="\t", index=False
        )
        paths = EXPORT.export_mean_sensitivity(self.root, self.out)
        self.assertEqual([path.name for path in paths], ["figure7_voxelwise_mean_sensitivity.tsv"])
        result = pd.read_csv(paths[0], sep="\t")
        self.assertEqual(set(result.target_key), {"primary"})
        self.assertEqual(set(result.candidate_key), {"amy_proper", "glasser__pir"})
        self.assertTrue(result.voxel_statistic.eq("mean").all())
        self.assertTrue(result.voxel_statistic_role.eq("sensitivity").all())
        self.assertFalse(any("rank" in column for column in result.columns))
        self.assertFalse(any("winner" in column for column in result.columns))
        self.assertNotIn("n_subjects", result.columns)
        self.assertNotIn("n_participants", result.columns)
        result = result.set_index("candidate_key")
        np.testing.assert_array_equal(result.n_cohort, [10, 10])
        np.testing.assert_array_equal(result.n_paired_cell_participants_min, [4, 2])
        np.testing.assert_array_equal(result.n_paired_cell_participants_median, [7, 5])
        np.testing.assert_array_equal(result.n_paired_cell_participants_max, [10, 9])
        np.testing.assert_allclose(result.group_r, [0.8, 0.7])
        np.testing.assert_allclose(result.bootstrap_q025, [0.7, 0.6])
        np.testing.assert_allclose(result.bootstrap_q975, [0.9, 0.8])


if __name__ == "__main__":
    unittest.main()
