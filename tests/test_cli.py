import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from spatial_coverage_audit.cli import main


class CliTests(unittest.TestCase):
    def test_duplicate_prediction_membership_is_scientifically_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "row_id": ["p1", "p1", "p2"],
                    "method": ["m", "m", "m"],
                    "repeat": [0, 0, 0],
                    "design": [
                        "spatial_target_holdout",
                        "spatial_target_holdout",
                        "random_marginal",
                    ],
                    "y_true": [0.0, 0.0, 0.0],
                    "y_pred": [0.0, 0.0, 0.0],
                    "lower": [-1.0, -1.0, -1.0],
                    "upper": [1.0, 1.0, 1.0],
                }
            )
            prediction_path = root / "predictions.csv"
            output_dir = root / "audit"
            predictions.to_csv(prediction_path, index=False)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(
                    [
                    "--predictions",
                    str(prediction_path),
                    "--output-dir",
                    str(output_dir),
                    "--method-col",
                    "method",
                    "--repeat-col",
                    "repeat",
                    "--design-col",
                    "design",
                    ]
                )
            result = json.loads((output_dir / "audit_result.json").read_text("utf-8"))
            validation = result["prediction_identifier_validation"]
            self.assertFalse(validation["valid"])
            self.assertEqual(validation["duplicate_rows"], 2)
            self.assertEqual(result["run_status"], "completed")
            self.assertEqual(result["scientific_fitness"]["state"], "not_assessed")
            self.assertEqual(
                result["scientific_fitness"]["contract_verification"][
                    "prediction_identifiers"
                ],
                "failed",
            )
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["scientific_state"], "not_assessed")
            self.assertEqual(status["scientific_fitness"], "not_assessed")
            self.assertFalse(status["full_contract_verified"])
            self.assertNotIn("no_failure_detected", stdout.getvalue())

    def test_membership_failure_keeps_run_success_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "row_id": ["p1", "p2", "p3", "p4"],
                    "method": ["m"] * 4,
                    "repeat": [0] * 4,
                    "design": ["spatial_target_holdout"] * 4,
                    "domain": ["B"] * 4,
                    "y_true": [0.0, 1.0, 2.0, 3.0],
                    "y_pred": [0.0, 1.0, 2.0, 3.0],
                    "lower": [-1.0, 0.0, 1.0, 2.0],
                    "upper": [1.0, 2.0, 3.0, 4.0],
                }
            )
            membership = pd.DataFrame(
                {
                    "fold": [0, 0, 0, 0, 0],
                    "row_id": ["a1", "a2", "b1", "b1", "b2"],
                    "membership": ["fit", "calibration", "test", "test", "fit"],
                    "group": ["A", "A", "B", "B", "B"],
                }
            )
            prediction_path = root / "predictions.csv"
            membership_path = root / "membership.csv"
            output_dir = root / "audit"
            predictions.to_csv(prediction_path, index=False)
            membership.to_csv(membership_path, index=False)
            main(
                [
                    "--predictions",
                    str(prediction_path),
                    "--output-dir",
                    str(output_dir),
                    "--membership-ledger",
                    str(membership_path),
                    "--membership-run-cols",
                    "fold",
                    "--complete-group-holdout",
                ]
            )
            result = json.loads((output_dir / "audit_result.json").read_text("utf-8"))
            self.assertEqual(result["run_status"], "completed")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["scientific_fitness"]["state"], "not_assessed")
            self.assertEqual(
                result["scientific_fitness"]["contract_verification"][
                    "membership_ledger"
                ],
                "failed",
            )
            self.assertFalse(result["membership_validation"]["valid"])
            self.assertTrue((output_dir / "membership_validation_issues.csv").is_file())
            self.assertTrue((output_dir / "fitness_state.json").is_file())

    def test_negative_cqr_score_is_reported_without_clipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "row_id": ["p1"],
                    "y_true": [0.0],
                    "y_pred": [0.0],
                    "lower": [-1.0],
                    "upper": [1.0],
                }
            )
            calibration = pd.DataFrame({"cqr_score": [-3.0, -2.0, -1.0]})
            prediction_path = root / "predictions.csv"
            calibration_path = root / "calibration.csv"
            output_dir = root / "audit"
            predictions.to_csv(prediction_path, index=False)
            calibration.to_csv(calibration_path, index=False)
            main(
                [
                    "--predictions",
                    str(prediction_path),
                    "--calibration",
                    str(calibration_path),
                    "--score-col",
                    "cqr_score",
                    "--alpha",
                    "0.5",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            result = json.loads((output_dir / "audit_result.json").read_text("utf-8"))
            self.assertEqual(result["calibration"]["exact_q"], -2.0)
            self.assertEqual(
                result["scientific_fitness"]["assessment_status"],
                "not_assessed",
            )
            self.assertFalse(
                result["scientific_fitness"]["contract_verification"][
                    "full_contract_verified"
                ]
            )
    def test_scientific_fitness_requires_criteria_and_verified_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "fold": [0, 0],
                    "row_id": ["b1", "b2"],
                    "y_true": [0.0, 1.0],
                    "y_pred": [0.0, 1.0],
                    "lower": [-1.0, 0.0],
                    "upper": [1.0, 2.0],
                }
            )
            membership = pd.DataFrame(
                {
                    "fold": [0, 0, 0, 0],
                    "row_id": ["a1", "a2", "b1", "b2"],
                    "membership": ["fit", "calibration", "test", "test"],
                    "group": ["A", "A", "B", "B"],
                }
            )
            prediction_path = root / "predictions.csv"
            membership_path = root / "membership.csv"
            output_dir = root / "audit"
            predictions.to_csv(prediction_path, index=False)
            membership.to_csv(membership_path, index=False)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(
                    [
                        "--predictions",
                        str(prediction_path),
                        "--output-dir",
                        str(output_dir),
                        "--membership-ledger",
                        str(membership_path),
                        "--membership-run-cols",
                        "fold",
                        "--complete-group-holdout",
                        "--fitness-min-picp",
                        "0.8",
                    ]
                )
            result = json.loads((output_dir / "audit_result.json").read_text("utf-8"))
            fitness = result["scientific_fitness"]
            self.assertEqual(fitness["assessment_status"], "assessed")
            self.assertEqual(fitness["state"], "no_failure_detected")
            self.assertTrue(fitness["contract_verification"]["full_contract_verified"])
            self.assertEqual(fitness["unbounded_policy"]["source"], "default_zero")
            status = json.loads(stdout.getvalue())
            self.assertTrue(status["full_contract_verified"])

            predictions.loc[0, "upper"] = math.inf
            predictions.to_csv(prediction_path, index=False)
            unbounded_output = root / "audit_unbounded"
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--predictions",
                        str(prediction_path),
                        "--output-dir",
                        str(unbounded_output),
                        "--membership-ledger",
                        str(membership_path),
                        "--membership-run-cols",
                        "fold",
                        "--complete-group-holdout",
                        "--fitness-min-picp",
                        "0.8",
                    ]
                )
            unbounded = json.loads(
                (unbounded_output / "audit_result.json").read_text("utf-8")
            )
            self.assertEqual(
                unbounded["scientific_fitness"]["state"],
                "operationally_unusable",
            )

            override_output = root / "audit_override"
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "--predictions",
                        str(prediction_path),
                        "--output-dir",
                        str(override_output),
                        "--membership-ledger",
                        str(membership_path),
                        "--membership-run-cols",
                        "fold",
                        "--complete-group-holdout",
                        "--fitness-min-picp",
                        "0.8",
                        "--fitness-unusable-unbounded-rate",
                        "0.5",
                    ]
                )
            overridden = json.loads(
                (override_output / "audit_result.json").read_text("utf-8")
            )
            self.assertEqual(
                overridden["scientific_fitness"]["state"], "no_failure_detected"
            )
            self.assertEqual(
                overridden["scientific_fitness"]["unbounded_policy"]["source"],
                "user_override",
            )

    def test_generic_prediction_and_calibration_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "row_id": [f"r{i}" for i in range(8)],
                    "repeat": [0, 0, 0, 0, 1, 1, 1, 1],
                    "design": [
                        "random_marginal",
                        "random_marginal",
                        "spatial_target_holdout",
                        "spatial_target_holdout",
                    ]
                    * 2,
                    "domain": ["A", "B", "A", "B"] * 2,
                    "y_true": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
                    "y_pred": [0.1, 1.1, 1.9, 2.8, 0.1, 1.1, 1.9, 2.8],
                    "lower": [-1.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0, 2.0],
                    "upper": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                    "target_weight": [0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1],
                }
            )
            calibration = pd.DataFrame(
                {"score": list(range(1, 10)), "weight": [1.0] * 9}
            )
            prediction_path = root / "predictions.csv"
            calibration_path = root / "calibration.csv"
            output_dir = root / "audit"
            predictions.to_csv(prediction_path, index=False)
            calibration.to_csv(calibration_path, index=False)
            main(
                [
                    "--predictions",
                    str(prediction_path),
                    "--calibration",
                    str(calibration_path),
                    "--output-dir",
                    str(output_dir),
                    "--group-col",
                    "domain",
                    "--repeat-col",
                    "repeat",
                    "--design-col",
                    "design",
                    "--pair-cols",
                    "repeat",
                    "--score-col",
                    "score",
                    "--calibration-weight-col",
                    "weight",
                    "--target-weight-col",
                    "target_weight",
                    "--id-col",
                    "row_id",
                    "--include-rmse",
                ]
            )
            result = json.loads((output_dir / "audit_result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["calibration"]["exact_q"], 9.0)
            self.assertEqual(result["calibration"]["weighted"]["unbounded_n"], 1)
            self.assertTrue((output_dir / "group_summary.csv").is_file())
            self.assertTrue((output_dir / "group_aggregate_summary.csv").is_file())
            self.assertTrue((output_dir / "random_vs_group_contrasts.csv").is_file())
            self.assertTrue((output_dir / "target_weighted_quantiles.csv").is_file())
            self.assertIn("rmse", pd.read_csv(output_dir / "overall_summary.csv"))
            self.assertIn(
                "macro_rmse", pd.read_csv(output_dir / "group_aggregate_summary.csv")
            )

    def test_default_design_pairs_preserve_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = pd.DataFrame(
                {
                    "method": ["m1", "m1", "m2", "m2"],
                    "design": [
                        "random_marginal",
                        "spatial_target_holdout",
                        "random_marginal",
                        "spatial_target_holdout",
                    ],
                    "y_true": [0.0, 0.0, 1.0, 1.0],
                    "y_pred": [0.0, 0.0, 1.0, 1.0],
                    "lower": [-1.0, -0.5, 0.0, 0.5],
                    "upper": [1.0, 0.5, 2.0, 1.5],
                }
            )
            prediction_path = root / "predictions.csv"
            output_dir = root / "audit"
            predictions.to_csv(prediction_path, index=False)
            main(
                [
                    "--predictions",
                    str(prediction_path),
                    "--output-dir",
                    str(output_dir),
                    "--method-col",
                    "method",
                    "--design-col",
                    "design",
                ]
            )
            contrasts = pd.read_csv(output_dir / "random_vs_group_contrasts.csv")
            self.assertEqual(contrasts["method"].tolist(), ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()
