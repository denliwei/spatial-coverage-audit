import math
import unittest

import numpy as np

from spatial_coverage_audit.metrics import (
    coverage_indicators,
    interval_scores,
    root_mean_square_error,
    summarize_intervals,
    wilson_interval,
)


class MetricTests(unittest.TestCase):
    def test_rmse_is_explicit_and_optional_in_summary(self):
        self.assertAlmostEqual(
            root_mean_square_error([0.0, 2.0], [0.0, 0.0]), 2**0.5
        )
        summary = summarize_intervals(
            [0.0, 2.0],
            [-1.0, -1.0],
            [1.0, 3.0],
            0.10,
            [0.0, 0.0],
            include_rmse=True,
        )
        self.assertAlmostEqual(summary["rmse"], 2**0.5)

    def test_inclusive_coverage_and_interval_score(self):
        y = np.array([0.0, 5.0])
        lower = np.array([0.0, 1.0])
        upper = np.array([3.0, 3.0])
        np.testing.assert_array_equal(coverage_indicators(y, lower, upper), [True, False])
        np.testing.assert_allclose(interval_scores(y, lower, upper, 0.10), [3.0, 42.0])

    def test_wilson_matches_frozen_value(self):
        low, high = wilson_interval(90, 100)
        self.assertAlmostEqual(low, 0.8256343384950865, places=15)
        self.assertAlmostEqual(high, 0.9447708629393249, places=15)

    def test_unbounded_intervals_remain_in_overall_metrics(self):
        summary = summarize_intervals(
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 2.0, math.inf],
            alpha=0.10,
            point_prediction=[0.0, 1.5, 2.5],
        )
        self.assertEqual(summary["covered_n"], 2)
        self.assertAlmostEqual(summary["picp"], 2.0 / 3.0)
        self.assertTrue(math.isinf(summary["mpiw"]))
        self.assertTrue(math.isinf(summary["interval_score"]))
        self.assertEqual(summary["finite_interval_n"], 2)
        self.assertAlmostEqual(summary["mpiw_finite"], 1.0)
        self.assertAlmostEqual(summary["unbounded_interval_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["mae"], 1.0 / 3.0)

    def test_invalid_interval_rejected(self):
        with self.assertRaises(ValueError):
            summarize_intervals([1.0], [2.0], [1.0], alpha=0.10)

    def test_indeterminate_infinite_interval_rejected(self):
        with self.assertRaises(ValueError):
            summarize_intervals([1.0], [math.inf], [math.inf], alpha=0.10)


if __name__ == "__main__":
    unittest.main()
