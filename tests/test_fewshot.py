import math
import unittest

import numpy as np
import pandas as pd

from spatial_coverage_audit.fewshot import build_fewshot_recalibration_ledger


class FewShotTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for group in ("A", "B"):
            for index in range(12):
                rows.append(
                    {
                        "domain": group,
                        "plotID": f"{group}{index:02d}",
                        "y_true": float(index),
                        "y_pred": float(index) - (index + 1) / 10.0,
                    }
                )
        self.data = pd.DataFrame(rows)

    def test_deterministic_membership_and_unbounded_small_k(self):
        first = build_fewshot_recalibration_ledger(
            self.data,
            group_col="domain",
            row_id_col="plotID",
            calibration_sizes=[5, 9],
            repeats=2,
            alpha=0.10,
            seed=42,
        )
        second = build_fewshot_recalibration_ledger(
            self.data.sample(frac=1.0, random_state=7),
            group_col="domain",
            row_id_col="plotID",
            calibration_sizes=[9, 5],
            repeats=2,
            alpha=0.10,
            seed=42,
        )
        pd.testing.assert_frame_equal(first.calibration_ledger, second.calibration_ledger)
        pd.testing.assert_frame_equal(first.runs, second.runs)
        self.assertEqual(len(first.calibration_ledger), 56)
        self.assertEqual(len(first.runs), 8)
        self.assertTrue(first.runs.loc[first.runs.target_calibration_n.eq(5), "unbounded"].all())
        self.assertFalse(first.runs.loc[first.runs.target_calibration_n.eq(9), "unbounded"].any())
        self.assertTrue(
            np.isinf(
                first.runs.loc[first.runs.target_calibration_n.eq(5), "calibration_q"]
            ).all()
        )

    def test_duplicate_group_identifier_rejected(self):
        duplicate = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_fewshot_recalibration_ledger(
                duplicate,
                group_col="domain",
                row_id_col="plotID",
                calibration_sizes=[5],
                repeats=1,
            )


if __name__ == "__main__":
    unittest.main()

