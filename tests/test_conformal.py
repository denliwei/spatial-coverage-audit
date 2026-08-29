import math
import unittest

import numpy as np

from spatial_coverage_audit.conformal import (
    conformal_rank,
    effective_sample_size,
    exact_split_conformal_quantile,
    target_point_weighted_quantiles,
)


class ConformalTests(unittest.TestCase):
    def test_exact_rank_and_infinity_for_too_few_scores(self):
        self.assertEqual(conformal_rank(9, 0.10), 9)
        self.assertEqual(exact_split_conformal_quantile(np.arange(1, 10), 0.10), 9.0)
        self.assertTrue(math.isinf(exact_split_conformal_quantile(np.arange(1, 9), 0.10)))

    def test_exact_quantile_rejects_nonfinite_scores(self):
        scores = [1.0, 3.0, math.nan, math.inf, 2.0]
        with self.assertRaises(ValueError):
            exact_split_conformal_quantile(scores, 0.50)

    def test_negative_cqr_scores_are_not_clipped(self):
        # CQR scores may be negative when the response lies inside the base interval.
        self.assertEqual(
            exact_split_conformal_quantile([-3.0, -2.0, -1.0], 0.50), -2.0
        )

    def test_invalid_input_is_distinct_from_valid_unbounded_rank(self):
        self.assertTrue(math.isinf(exact_split_conformal_quantile([1.0], 0.10)))
        with self.assertRaises(ValueError):
            exact_split_conformal_quantile([math.nan, math.inf], 0.10)
        with self.assertRaises(ValueError):
            exact_split_conformal_quantile([1.0, 2.0], 1.0)

    def test_target_point_weighted_infinity_mass(self):
        result = target_point_weighted_quantiles(
            scores=[1.0, 2.0, 3.0],
            calibration_weights=[1.0, 1.0, 1.0],
            target_weights=[0.0, 1.0, 2.0],
            alpha=0.25,
        )
        np.testing.assert_allclose(result.quantiles[:2], [3.0, 3.0])
        self.assertTrue(math.isinf(result.quantiles[2]))
        np.testing.assert_allclose(result.target_infinity_mass, [0.0, 0.25, 0.4])
        np.testing.assert_array_equal(result.unbounded_mask, [False, False, True])
        self.assertAlmostEqual(result.unbounded_rate, 1.0 / 3.0)

    def test_invalid_target_weight_rejected(self):
        with self.assertRaises(ValueError):
            target_point_weighted_quantiles([1.0], [1.0], [-1.0], 0.10)

    def test_invalid_calibration_weights_are_rejected_and_zero_is_counted(self):
        with self.assertRaises(ValueError):
            target_point_weighted_quantiles([1.0], [-1.0], [1.0], 0.10)
        with self.assertRaises(ValueError):
            target_point_weighted_quantiles([1.0], [math.nan], [1.0], 0.10)
        result = target_point_weighted_quantiles(
            [1.0, 2.0], [0.0, 1.0], [0.0], 0.50
        )
        self.assertEqual(result.calibration_rows, 2)
        self.assertEqual(result.positive_weight_rows, 1)
        self.assertEqual(result.zero_weight_rows, 1)

    def test_effective_sample_size(self):
        self.assertEqual(effective_sample_size([1.0, 1.0, 1.0]), 3.0)
        self.assertAlmostEqual(effective_sample_size([1.0, 2.0]), 9.0 / 5.0)


if __name__ == "__main__":
    unittest.main()
