"""Synthetic checks for the manuscript's HCP cohort-selection criteria."""

from pathlib import Path
import runpy
import unittest

import pandas as pd


SELECTOR = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "analysis" / "01_select_hcp_cohort.py")
)


class CohortSelectionTests(unittest.TestCase):
    def example_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Subject": ["eligible", "lower_accuracy", "incomplete", "qc_issue"],
                "Emotion_Task_Acc": [80.0, 100.0, 100.0, 100.0],
                "Language_Task_Acc": [90.0, 79.9, 100.0, 100.0],
                "Relational_Task_Acc": [95.0, 100.0, 100.0, 100.0],
                "WM_Task_Acc": [100.0, 100.0, 100.0, 100.0],
                "3T_Full_Task_fMRI": [True, True, False, True],
                "QC_Issue": [None, None, None, "reported imaging issue"],
            }
        )

    def test_cli_default_uses_percentage_units(self) -> None:
        args = SELECTOR["parse_args"](
            ["--behavior", "synthetic.csv", "--output", "selected.tsv"]
        )
        self.assertEqual(args.minimum_accuracy, 80.0)

    def test_requires_80_percent_complete_tasks_and_no_qc_issue(self) -> None:
        selected = SELECTOR["eligible_subjects"](self.example_table())
        self.assertEqual(selected.tolist(), ["eligible"])

    def test_missing_accuracy_and_fractional_scores_do_not_pass(self) -> None:
        for score in (None, "", "not scored", 0.95):
            with self.subTest(score=score):
                frame = self.example_table().astype(object)
                frame.loc[0, "Emotion_Task_Acc"] = score
                self.assertTrue(SELECTOR["eligible_subjects"](frame).empty)

    def test_missing_screening_columns_raise(self) -> None:
        for column in SELECTOR["REQUIRED_SCREENING_COLUMNS"]:
            with self.subTest(column=column):
                frame = self.example_table().drop(columns=column)
                with self.assertRaisesRegex(ValueError, column):
                    SELECTOR["eligible_subjects"](frame)

    def test_missing_completeness_flag_does_not_pass(self) -> None:
        frame = self.example_table().astype(object)
        frame.loc[0, "3T_Full_Task_fMRI"] = pd.NA
        self.assertTrue(SELECTOR["eligible_subjects"](frame).empty)

    def test_csv_boolean_strings_are_supported(self) -> None:
        frame = self.example_table()
        frame["3T_Full_Task_fMRI"] = frame["3T_Full_Task_fMRI"].astype(str)
        self.assertEqual(SELECTOR["eligible_subjects"](frame).tolist(), ["eligible"])


if __name__ == "__main__":
    unittest.main()
