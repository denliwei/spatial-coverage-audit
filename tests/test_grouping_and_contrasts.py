import math
import unittest

import pandas as pd

from spatial_coverage_audit.contrasts import (
    random_vs_group_contrasts,
    summarize_design_replicates,
)
from spatial_coverage_audit.grouping import aggregate_group_metrics, group_interval_summaries


class GroupingTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame(
            {
                "domain": ["A", "A", "B"],
                "y_true": [0.0, 2.0, 5.0],
                "y_pred": [0.0, 1.0, 5.0],
                "lower": [-1.0, 0.0, 0.0],
                "upper": [1.0, 1.0, math.inf],
            }
        )

    def test_macro_micro_worst_and_infinity(self):
        groups = group_interval_summaries(self.predictions, "domain", alpha=0.10)
        aggregate = aggregate_group_metrics(
            self.predictions, groups, "domain", alpha=0.10
        ).iloc[0]
        self.assertEqual(aggregate["groups"], 2)
        self.assertAlmostEqual(aggregate["macro_picp"], 0.75)
        self.assertAlmostEqual(aggregate["micro_picp"], 2.0 / 3.0)
        self.assertEqual(aggregate["worst_group"], "A")
        self.assertAlmostEqual(aggregate["worst_group_picp"], 0.5)
        self.assertTrue(math.isinf(aggregate["macro_mpiw"]))
        self.assertTrue(math.isinf(aggregate["micro_mpiw"]))
        self.assertAlmostEqual(aggregate["macro_mpiw_finite"], 1.5)
        self.assertAlmostEqual(aggregate["micro_mpiw_finite"], 1.5)
        self.assertAlmostEqual(aggregate["micro_unbounded_interval_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(aggregate["macro_mae"], 0.25)
        self.assertAlmostEqual(aggregate["micro_mae"], 1.0 / 3.0)

    def test_missing_stratum_is_matched(self):
        predictions = self.predictions.assign(scenario=None)
        groups = group_interval_summaries(
            predictions, "domain", strata_cols=["scenario"], alpha=0.10
        )
        aggregate = aggregate_group_metrics(
            predictions,
            groups,
            "domain",
            strata_cols=["scenario"],
            alpha=0.10,
        )
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate.loc[0, "groups"], 2)

    def test_optional_group_rmse_outputs(self):
        groups = group_interval_summaries(
            self.predictions, "domain", alpha=0.10, include_rmse=True
        )
        aggregate = aggregate_group_metrics(
            self.predictions,
            groups,
            "domain",
            alpha=0.10,
            include_rmse=True,
        ).iloc[0]
        self.assertIn("rmse", groups)
        self.assertIn("macro_rmse", aggregate.index)
        self.assertAlmostEqual(aggregate["micro_rmse"], (1.0 / 3.0) ** 0.5)

    def test_missing_declared_group_is_rejected(self):
        invalid = self.predictions.copy()
        invalid.loc[0, "domain"] = None
        with self.assertRaisesRegex(ValueError, "declared group"):
            group_interval_summaries(invalid, "domain", alpha=0.10)


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.metrics = pd.DataFrame(
            {
                "repeat": [0, 0, 1, 1],
                "design": [
                    "random_marginal",
                    "spatial_target_holdout",
                    "random_marginal",
                    "spatial_target_holdout",
                ],
                "picp": [0.90, 0.70, 0.95, 0.75],
                "mpiw": [10.0, 12.0, 11.0, 15.0],
                "interval_score": [12.0, 20.0, 13.0, 25.0],
                "mae": [2.0, 3.0, 2.5, 4.0],
                "coverage_failure_detected": [False, True, False, True],
            }
        )

    def test_paired_contrasts(self):
        result = random_vs_group_contrasts(self.metrics, pair_cols=["repeat"])
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result.loc[0, "random_minus_group_picp"], 0.20)
        self.assertAlmostEqual(result.loc[1, "group_minus_random_mpiw"], 4.0)
        self.assertAlmostEqual(result.loc[1, "group_minus_random_interval_score"], 12.0)

    def test_single_pair_without_pair_columns(self):
        result = random_vs_group_contrasts(
            self.metrics.iloc[:2], pair_cols=[]
        )
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "random_minus_group_picp"], 0.20)

    def test_design_summary(self):
        result = summarize_design_replicates(self.metrics, scenario_cols=[])
        spatial = result[result.design.eq("spatial_target_holdout")].iloc[0]
        self.assertEqual(spatial["failure_detection_rate"], 1.0)
        self.assertAlmostEqual(spatial["mean_picp"], 0.725)


if __name__ == "__main__":
    unittest.main()
