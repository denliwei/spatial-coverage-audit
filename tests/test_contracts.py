import unittest

import pandas as pd

from spatial_coverage_audit.contracts import (
    FitnessCriteria,
    FitnessObservation,
    FitnessState,
    assess_fitness,
    validate_membership_ledger,
    validate_prediction_evaluation_alignment,
)


class MembershipContractTests(unittest.TestCase):
    def setUp(self):
        self.valid = pd.DataFrame(
            {
                "fold": [0, 0, 0, 0, 0],
                "row_id": ["a1", "a2", "a3", "b1", "b2"],
                "membership": ["fit", "fit", "calibration", "test", "test"],
                "domain": ["A", "A", "A", "B", "B"],
            }
        )

    def test_valid_complete_group_holdout(self):
        result = validate_membership_ledger(
            self.valid,
            run_cols=["fold"],
            group_col="domain",
            complete_group_holdout=True,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.run_count, 1)
        self.assertEqual(result.issue_counts, {})

    def test_duplicate_evaluation_and_target_leakage_detected(self):
        invalid = pd.concat(
            [
                self.valid,
                pd.DataFrame(
                    {
                        "fold": [0, 0],
                        "row_id": ["b1", "b3"],
                        "membership": ["test", "calibration"],
                        "domain": ["B", "B"],
                    }
                ),
            ],
            ignore_index=True,
        )
        result = validate_membership_ledger(
            invalid,
            run_cols=["fold"],
            group_col="domain",
            complete_group_holdout=True,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.issue_counts["duplicate_evaluation_membership"], 1)
        self.assertEqual(result.issue_counts["target_group_leakage"], 1)

    def test_cross_membership_overlap_detected(self):
        invalid = pd.concat(
            [
                self.valid,
                pd.DataFrame(
                    {
                        "fold": [0],
                        "row_id": ["b2"],
                        "membership": ["fit"],
                        "domain": ["B"],
                    }
                ),
            ],
            ignore_index=True,
        )
        result = validate_membership_ledger(
            invalid,
            run_cols=["fold"],
            group_col="domain",
            complete_group_holdout=True,
        )
        self.assertEqual(result.issue_counts["cross_membership_overlap"], 1)
        self.assertEqual(result.issue_counts["target_group_leakage"], 1)

    def test_missing_columns_are_invalid_input_not_scientific_state(self):
        with self.assertRaisesRegex(ValueError, "columns missing"):
            validate_membership_ledger(pd.DataFrame({"row_id": ["x"]}))

    def test_missing_run_key_is_invalid_input(self):
        invalid = self.valid.copy()
        invalid.loc[0, "fold"] = None
        with self.assertRaisesRegex(ValueError, "run keys contain missing"):
            validate_membership_ledger(invalid, run_cols=["fold"])

    def test_prediction_keys_must_match_test_ledger_keys(self):
        predictions = pd.DataFrame({"fold": [0, 0], "row_id": ["b1", "b2"]})
        aligned = validate_prediction_evaluation_alignment(
            predictions, self.valid, run_cols=["fold"]
        )
        self.assertEqual(aligned["status"], "passed")
        misaligned = validate_prediction_evaluation_alignment(
            predictions.iloc[:1], self.valid, run_cols=["fold"]
        )
        self.assertEqual(misaligned["status"], "failed")
        self.assertEqual(misaligned["ledger_only_keys"], 1)


class FitnessStateTests(unittest.TestCase):
    def test_all_four_states(self):
        criteria = FitnessCriteria(
            min_picp_failure=0.80,
            min_picp_warning=0.90,
            max_unbounded_rate_failure=0.20,
            max_unbounded_rate_warning=0.10,
            unusable_unbounded_rate=0.50,
        )
        states = [
            assess_fitness(
                FitnessObservation(min_picp=0.92, max_unbounded_rate=0.0), criteria
            ).state,
            assess_fitness(
                FitnessObservation(min_picp=0.85, max_unbounded_rate=0.0), criteria
            ).state,
            assess_fitness(
                FitnessObservation(min_picp=0.75, max_unbounded_rate=0.0), criteria
            ).state,
            assess_fitness(
                FitnessObservation(min_picp=0.92, max_unbounded_rate=0.51), criteria
            ).state,
        ]
        self.assertEqual(
            states,
            [
                FitnessState.NO_FAILURE_DETECTED,
                FitnessState.WARNING,
                FitnessState.FAILURE_DETECTED,
                FitnessState.OPERATIONALLY_UNUSABLE,
            ],
        )

    def test_contract_failure_is_a_scientific_failure(self):
        result = assess_fitness(
            FitnessObservation(min_picp=0.95, max_unbounded_rate=0.0),
            FitnessCriteria(min_picp_failure=0.80),
            contract_failures=["membership:target_group_leakage=1"],
        )
        self.assertEqual(result.state, FitnessState.FAILURE_DETECTED)
        self.assertEqual(result.contract_failures, ("membership:target_group_leakage=1",))

    def test_empty_user_criteria_are_explicitly_not_assessed(self):
        result = assess_fitness(
            FitnessObservation(min_picp=0.95, max_unbounded_rate=0.0),
            FitnessCriteria(),
        )
        payload = result.to_dict()
        self.assertIsNone(result.state)
        self.assertEqual(payload["state"], "not_assessed")
        self.assertEqual(payload["assessment_status"], "not_assessed")
        self.assertEqual(payload["mode"], "metric_only")
        self.assertFalse(payload["user_criteria_supplied"])

    def test_any_unbounded_output_is_unusable_once_assessment_is_enabled(self):
        result = assess_fitness(
            FitnessObservation(min_picp=0.95, max_unbounded_rate=0.001),
            FitnessCriteria(min_picp_failure=0.80),
        )
        self.assertEqual(result.state, FitnessState.OPERATIONALLY_UNUSABLE)

    def test_required_unavailable_metric_is_operationally_unusable(self):
        result = assess_fitness(
            FitnessObservation(min_picp=0.95, max_unbounded_rate=0.0),
            FitnessCriteria(min_worst_group_picp_failure=0.80),
        )
        self.assertEqual(result.state, FitnessState.OPERATIONALLY_UNUSABLE)

    def test_nonmonotone_criteria_are_rejected(self):
        with self.assertRaises(ValueError):
            FitnessCriteria(min_picp_failure=0.90, min_picp_warning=0.80)


if __name__ == "__main__":
    unittest.main()
