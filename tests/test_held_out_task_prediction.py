"""Synthetic checks of task-level prediction, weighting, and score definition."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from amyvasc_release.source_profiles import (
    CandidateSpec,
    TargetSpec,
    held_out_task_prediction,
)


class HeldOutTaskPredictionTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetSpec("target", "Target", Path("unused.nii.gz"))
        self.candidates = [
            CandidateSpec("source", "Source"),
            CandidateSpec("constant", "Constant"),
            CandidateSpec("composite", "Composite", members=("source",)),
            CandidateSpec("unranked", "Unranked", ranked=False),
        ]
        rows = []
        for participant in ("synthetic_a", "synthetic_b"):
            for task, values in (
                ("emotion", (1.0, 2.0, 3.0, 4.0)),
                ("language", (6.0, 7.0)),
                ("motor", (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
            ):
                for cell, value in enumerate(values):
                    rows.append(
                        {
                            "participant": participant,
                            "task": task,
                            "condition_key": f"{task}__{cell // 2}",
                            "hemisphere": "left" if cell % 2 == 0 else "right",
                            "source": value,
                            "constant": 0.3,
                            "target": 2 + 3 * value,
                        }
                    )
        self.wide = pd.DataFrame(rows)

    def analyze(self, frame=None, **kwargs):
        return held_out_task_prediction(
            self.wide if frame is None else frame,
            self.target,
            self.candidates,
            **kwargs,
        ).set_index("candidate_key")

    def test_intercept_recovers_affine_prediction_and_ranks_only_regions(self):
        result = self.analyze()
        self.assertEqual(set(result.index), {"source", "constant"})
        self.assertAlmostEqual(result.loc["source", "q2_global_mean"], 1.0)
        self.assertEqual(result.loc["source", "rank"], 1)
        self.assertEqual(result.loc["source", "n_cohort"], 2)

    def test_duplicating_conditions_within_one_task_preserves_task_weight(self):
        duplicated = self.wide.loc[self.wide.task.eq("emotion")].copy()
        duplicated["condition_key"] += "_replicate"
        result = self.analyze(pd.concat([self.wide, duplicated], ignore_index=True))
        np.testing.assert_allclose(
            result["q2_global_mean"], self.analyze()["q2_global_mean"], atol=1e-14
        )

    def test_emotion_is_absent_from_training_and_evaluation(self):
        expected = self.analyze(exclude_tasks={"emotion"})
        changed = self.wide.copy()
        changed.loc[changed.task.eq("emotion"), ["source", "constant", "target"]] = 1e8
        observed = self.analyze(changed, exclude_tasks={"emotion"})
        pd.testing.assert_frame_equal(observed, expected)
        self.assertTrue(observed["n_tasks"].eq(2).all())

    def test_constant_predictor_and_overall_mean_denominator(self):
        task_values = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([6.0, 7.0]),
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ]
        target_values = [2 + 3 * values for values in task_values]
        means = np.array([values.mean() for values in target_values])
        overall_mean = means.mean()
        expected_sst = np.mean(
            [np.mean((values - overall_mean) ** 2) for values in target_values]
        )
        expected_sse = np.mean(
            [
                np.mean((values - np.delete(means, index).mean()) ** 2)
                for index, values in enumerate(target_values)
            ]
        )
        result = self.analyze().loc["constant"]
        self.assertAlmostEqual(result.weighted_sse, expected_sse)
        self.assertAlmostEqual(result.weighted_sst, expected_sst)
        self.assertAlmostEqual(result.q2_global_mean, 1 - expected_sse / expected_sst)

    def test_missing_participant_cell_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same condition/hemisphere cells"):
            self.analyze(self.wide.iloc[1:])

    def test_undefined_scores_are_rejected(self):
        constant = self.wide.copy()
        constant["target"] = 0.3
        with self.assertRaisesRegex(ValueError, "constant target"):
            self.analyze(constant)
        missing = self.wide.copy()
        missing.loc[missing.condition_key.eq("language__0"), "source"] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.analyze(missing)


if __name__ == "__main__":
    unittest.main()
