"""Check participant pairing, population displays, and Figure 6 input consistency."""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.figure6 import (
    _plot_lag_ordering,
    _plot_slope_distribution,
    _validate_participant_tables,
)


class Figure6Tests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(plt.close, "all")
        self.ordering = pd.DataFrame(
            {
                "display_id": [1, 2, 3],
                "amygdala_lag_ms": [0, 0, 0],
                "striate_lag_ms": [-1, 4, 9],
                "peduncular_lag_ms": [3, 10, 5],
            }
        )
        self.slopes = pd.DataFrame(
            {"display_id": [1, 2, 3], "slope_ms_per_mm": [-8, 2, 12]}
        )
        self.summary = pd.DataFrame(
            [
                ("median_lag_striatal_minus_amygdala_ms", 4),
                ("median_lag_peduncular_minus_amygdala_ms", 5),
                ("fraction_striatal_after_amygdala", 2 / 3),
                ("fraction_peduncular_after_amygdala", 1),
                ("fraction_peduncular_after_striatal", 2 / 3),
                ("combined_slope_mean_ms_per_mm", 2),
                ("combined_slope_median_ms_per_mm", 2),
                ("fraction_positive_slopes", 2 / 3),
                ("n_subjects_ordering", 3),
                ("n_subjects_slopes", 3),
            ],
            columns=["measure", "value"],
        )

    def test_lag_display_retains_pairs_and_negative_observations(self):
        _validate_participant_tables(self.ordering, self.slopes, self.summary)
        _, ax = plt.subplots()
        _plot_lag_ordering(ax, self.ordering)
        # Participant lines retain the original pairing, not independently sorted lags.
        for line, expected in zip(
            ax.lines[:3], ([0, -1, 3], [0, 4, 10], [0, 9, 5]), strict=True
        ):
            np.testing.assert_array_equal(line.get_ydata(), expected)
        np.testing.assert_array_equal(ax.lines[3].get_ydata(), [0, 4, 5])
        self.assertEqual(sum(len(points.get_offsets()) for points in ax.collections), 9)
        self.assertLess(ax.get_ylim()[0], -1)
        self.assertGreater(ax.get_ylim()[1], 10)

    def test_histogram_counts_all_participants_and_marks_the_mean(self):
        _, ax = plt.subplots()
        _plot_slope_distribution(ax, self.slopes)
        self.assertEqual(sum(bar.get_height() for bar in ax.patches), 3)
        occupied = [bar for bar in ax.patches if bar.get_height() > 0]
        self.assertEqual(len(occupied), 3)
        self.assertLess(occupied[0].get_x(), 0)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 0])
        np.testing.assert_array_equal(ax.lines[1].get_xdata(), [2, 2])

    def test_pairing_corruption_cannot_hide_behind_unchanged_segment_medians(self):
        corrupted = self.ordering.copy()
        corrupted["peduncular_lag_ms"] = [10, 3, 5]
        # Marginal values/medians are unchanged, but the within-person ordering is not.
        with self.assertRaisesRegex(ValueError, "fraction_peduncular_after_striatal"):
            _validate_participant_tables(corrupted, self.slopes, self.summary)

    def test_incomplete_duplicate_and_nonreference_rows_are_rejected(self):
        cases = []
        missing = self.ordering.copy()
        missing.loc[0, "peduncular_lag_ms"] = np.nan
        cases.append((missing, "must be finite"))
        duplicate = self.ordering.copy()
        duplicate.loc[1, "display_id"] = 1
        cases.append((duplicate, "duplicate participant"))
        nonreference = self.ordering.copy()
        nonreference.loc[0, "amygdala_lag_ms"] = 1
        cases.append((nonreference, "zero reference"))
        for ordering, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_participant_tables(ordering, self.slopes, self.summary)

    def test_stale_population_summary_is_rejected(self):
        stale = self.summary.copy()
        stale.loc[stale.measure.eq("n_subjects_slopes"), "value"] = 4
        with self.assertRaisesRegex(ValueError, "n_subjects_slopes"):
            _validate_participant_tables(self.ordering, self.slopes, stale)


if __name__ == "__main__":
    unittest.main()
