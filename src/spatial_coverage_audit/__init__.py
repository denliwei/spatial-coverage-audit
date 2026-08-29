"""Reusable spatial and grouped coverage-audit primitives."""

__version__ = "0.1.0"

from .conformal import (
    WeightedConformalResult,
    conformal_rank,
    effective_sample_size,
    exact_split_conformal_quantile,
    target_point_weighted_quantiles,
)
from .contrasts import random_vs_group_contrasts, summarize_design_replicates
from .contracts import (
    FitnessAssessment,
    FitnessCriteria,
    FitnessObservation,
    FitnessState,
    MembershipValidationResult,
    assess_fitness,
    validate_membership_ledger,
    validate_prediction_evaluation_alignment,
)
from .fewshot import FewShotRecalibrationResult, build_fewshot_recalibration_ledger, stable_seed
from .grouping import aggregate_group_metrics, group_interval_summaries, strata_interval_summaries
from .metrics import (
    coverage_indicators,
    interval_scores,
    interval_widths,
    root_mean_square_error,
    summarize_intervals,
    wilson_interval,
)

__all__ = [
    "FewShotRecalibrationResult",
    "FitnessAssessment",
    "FitnessCriteria",
    "FitnessObservation",
    "FitnessState",
    "MembershipValidationResult",
    "WeightedConformalResult",
    "aggregate_group_metrics",
    "assess_fitness",
    "build_fewshot_recalibration_ledger",
    "conformal_rank",
    "coverage_indicators",
    "effective_sample_size",
    "exact_split_conformal_quantile",
    "group_interval_summaries",
    "interval_scores",
    "interval_widths",
    "random_vs_group_contrasts",
    "root_mean_square_error",
    "stable_seed",
    "strata_interval_summaries",
    "summarize_design_replicates",
    "summarize_intervals",
    "target_point_weighted_quantiles",
    "validate_membership_ledger",
    "validate_prediction_evaluation_alignment",
    "wilson_interval",
    "__version__",
]
