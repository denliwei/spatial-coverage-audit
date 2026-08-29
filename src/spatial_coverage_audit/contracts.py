"""Audit-contract validation and bounded scientific-fitness states.

The functions in this module deliberately distinguish a successfully executed
audit from the scientific state reported by that audit.  A completed run may
therefore report leakage or a failed user criterion without being presented as
a software crash.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


_ISSUE_COLUMNS = [
    "issue_type",
    "severity",
    "run",
    "row_id",
    "membership",
    "group",
    "occurrence_count",
    "detail",
]


def _run_value(columns: Sequence[str], values: object) -> str:
    if not columns:
        return "{}"
    items = values if isinstance(values, tuple) else (values,)
    pairs = [f"{column}={value!r}" for column, value in zip(columns, items, strict=True)]
    return "{" + ", ".join(pairs) + "}"


def _iter_runs(frame: pd.DataFrame, run_cols: Sequence[str]):
    if not run_cols:
        yield "{}", frame
        return
    grouper: str | list[str] = run_cols[0] if len(run_cols) == 1 else list(run_cols)
    for values, subset in frame.groupby(grouper, dropna=False, sort=True):
        yield _run_value(run_cols, values), subset


@dataclass(frozen=True)
class MembershipValidationResult:
    """Machine-readable result of validating a fit/calibration/test ledger."""

    valid: bool
    ledger_rows: int
    run_count: int
    recognized_rows: int
    complete_group_holdout: bool
    issue_counts: dict[str, int]
    issues: pd.DataFrame
    limitations: tuple[str, ...]

    def to_dict(self, *, include_issues: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "valid": self.valid,
            "ledger_rows": self.ledger_rows,
            "run_count": self.run_count,
            "recognized_rows": self.recognized_rows,
            "complete_group_holdout": self.complete_group_holdout,
            "issue_counts": dict(self.issue_counts),
            "limitations": list(self.limitations),
        }
        if include_issues:
            result["issues"] = self.issues.to_dict(orient="records")
        return result


def validate_membership_ledger(
    ledger: pd.DataFrame,
    *,
    id_col: str = "row_id",
    split_col: str = "membership",
    run_cols: Sequence[str] = (),
    group_col: str | None = None,
    complete_group_holdout: bool = False,
    target_group_col: str | None = None,
) -> MembershipValidationResult:
    """Validate fit/calibration/test membership without stopping the run.

    Membership labels are case-insensitive and accept ``fit``/``train``,
    ``cal``/``calibration``, and ``test``/``eval``.  Within each run, the
    validator detects repeated evaluation identifiers and identifiers assigned
    to more than one membership.  For a complete-group holdout it additionally
    infers the single target group from evaluation rows (or reads
    ``target_group_col``) and detects any target-group row in fitting or
    calibration.

    The ledger must contain the full fit/calibration/test membership for every
    run.  Without an external population roster, the validator cannot prove
    that every eligible target-group row was included in evaluation.
    """

    if not isinstance(ledger, pd.DataFrame):
        raise TypeError("ledger must be a pandas DataFrame")
    run_cols = tuple(str(column) for column in run_cols)
    required = [id_col, split_col, *run_cols]
    if complete_group_holdout:
        if not group_col:
            raise ValueError("complete_group_holdout requires group_col")
        required.append(group_col)
    if target_group_col:
        if not complete_group_holdout:
            raise ValueError("target_group_col requires complete_group_holdout")
        required.append(target_group_col)
    missing = [column for column in required if column not in ledger]
    if missing:
        raise ValueError(f"membership ledger columns missing: {missing}")
    missing_run_keys = [column for column in run_cols if ledger[column].isna().any()]
    if missing_run_keys:
        raise ValueError(
            f"membership ledger run keys contain missing values: {missing_run_keys}"
        )

    aliases = {
        "fit": "fit",
        "train": "fit",
        "training": "fit",
        "cal": "calibration",
        "calib": "calibration",
        "calibration": "calibration",
        "test": "test",
        "eval": "test",
        "evaluation": "test",
    }
    work = ledger.copy()
    raw_membership = work[split_col].astype("string").str.strip().str.lower()
    work["_membership"] = raw_membership.map(aliases)
    issues: list[dict[str, Any]] = []

    def add_issue(
        issue_type: str,
        *,
        run: str = "{}",
        row_id: object = None,
        membership: object = None,
        group: object = None,
        occurrence_count: int = 1,
        detail: str,
    ) -> None:
        issues.append(
            {
                "issue_type": issue_type,
                "severity": "failure",
                "run": run,
                "row_id": None if pd.isna(row_id) else str(row_id),
                "membership": None if pd.isna(membership) else str(membership),
                "group": None if pd.isna(group) else str(group),
                "occurrence_count": int(occurrence_count),
                "detail": detail,
            }
        )

    if len(work) == 0:
        add_issue("empty_ledger", detail="membership ledger has no rows")
    missing_id = work[id_col].isna()
    for _, row in work.loc[missing_id].iterrows():
        add_issue(
            "missing_identifier",
            membership=row[split_col],
            detail=f"{id_col} is missing",
        )
    unknown = work["_membership"].isna()
    for _, row in work.loc[unknown].iterrows():
        add_issue(
            "unknown_membership",
            row_id=row[id_col],
            membership=row[split_col],
            detail="membership is not fit, calibration, or test",
        )

    valid_rows = work.loc[~missing_id & ~unknown].copy()
    run_count = 0
    for run, subset in _iter_runs(valid_rows, run_cols):
        run_count += 1
        identity = [*run_cols, id_col]
        duplicate_groups = subset.groupby(
            [*identity, "_membership"], dropna=False, sort=True
        ).size()
        for key, count in duplicate_groups[duplicate_groups.gt(1)].items():
            values = key if isinstance(key, tuple) else (key,)
            row_id = values[-2]
            membership = values[-1]
            issue_type = (
                "duplicate_evaluation_membership"
                if membership == "test"
                else "duplicate_membership"
            )
            add_issue(
                issue_type,
                run=run,
                row_id=row_id,
                membership=membership,
                occurrence_count=int(count),
                detail="identifier occurs repeatedly in the same run and membership",
            )

        memberships_per_id = subset.groupby(identity, dropna=False, sort=True)[
            "_membership"
        ].nunique()
        for key, count in memberships_per_id[memberships_per_id.gt(1)].items():
            values = key if isinstance(key, tuple) else (key,)
            add_issue(
                "cross_membership_overlap",
                run=run,
                row_id=values[-1],
                occurrence_count=int(count),
                detail="identifier is assigned to more than one membership in the same run",
            )

        if not complete_group_holdout:
            continue
        assert group_col is not None
        missing_group = subset[group_col].isna()
        for _, row in subset.loc[missing_group].iterrows():
            add_issue(
                "missing_group",
                run=run,
                row_id=row[id_col],
                membership=row["_membership"],
                detail=f"{group_col} is missing",
            )
        grouped = subset.loc[~missing_group]
        target: object | None = None
        if target_group_col:
            targets = grouped[target_group_col].dropna().unique()
            if len(targets) != 1:
                add_issue(
                    "ambiguous_target_group",
                    run=run,
                    occurrence_count=len(targets),
                    detail="complete-group holdout requires exactly one target group per run",
                )
            else:
                target = targets[0]
        else:
            targets = grouped.loc[grouped["_membership"].eq("test"), group_col].unique()
            if len(targets) != 1:
                add_issue(
                    "ambiguous_target_group",
                    run=run,
                    occurrence_count=len(targets),
                    detail="evaluation rows must identify exactly one target group per run",
                )
            else:
                target = targets[0]
        if target is None:
            continue
        target_evaluation = grouped["_membership"].eq("test") & grouped[group_col].eq(target)
        if not bool(target_evaluation.any()):
            add_issue(
                "missing_target_evaluation",
                run=run,
                group=target,
                detail=f"declared target group {target!r} has no evaluation row",
            )
        evaluation_mismatch = grouped["_membership"].eq("test") & grouped[group_col].ne(target)
        for _, row in grouped.loc[evaluation_mismatch].iterrows():
            add_issue(
                "non_target_evaluation_membership",
                run=run,
                row_id=row[id_col],
                membership="test",
                group=row[group_col],
                detail=f"evaluation row is outside declared target group {target!r}",
            )
        leakage = grouped["_membership"].isin(["fit", "calibration"]) & grouped[
            group_col
        ].eq(target)
        for _, row in grouped.loc[leakage].iterrows():
            add_issue(
                "target_group_leakage",
                run=run,
                row_id=row[id_col],
                membership=row["_membership"],
                group=row[group_col],
                detail=f"target group {target!r} appears in fitting or calibration",
            )

    issue_table = pd.DataFrame(issues, columns=_ISSUE_COLUMNS)
    if len(issue_table):
        issue_table = issue_table.sort_values(
            ["issue_type", "run", "row_id", "membership", "group"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    counts = {
        str(key): int(value)
        for key, value in issue_table["issue_type"].value_counts(sort=False).sort_index().items()
    }
    limitations = (
        "The ledger must enumerate all memberships within each declared run.",
        "Without an external population roster, target-group test completeness cannot be proven.",
    )
    return MembershipValidationResult(
        valid=not len(issue_table),
        ledger_rows=int(len(work)),
        run_count=run_count,
        recognized_rows=int(len(valid_rows)),
        complete_group_holdout=bool(complete_group_holdout),
        issue_counts=counts,
        issues=issue_table,
        limitations=limitations,
    )


def validate_prediction_evaluation_alignment(
    predictions: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    prediction_id_col: str = "row_id",
    ledger_id_col: str = "row_id",
    split_col: str = "membership",
    run_cols: Sequence[str] = (),
) -> dict[str, Any]:
    """Compare prediction keys with the ledger's test/evaluation keys.

    Missing prediction identifier/run columns return ``status=unavailable`` so
    a caller can fall back to metric-only reporting. Missing ledger schema is an
    invalid input and raises :class:`ValueError`.
    """

    run_cols = tuple(str(column) for column in run_cols)
    ledger_required = [ledger_id_col, split_col, *run_cols]
    missing_ledger = [column for column in ledger_required if column not in ledger]
    if missing_ledger:
        raise ValueError(f"membership ledger columns missing: {missing_ledger}")
    missing_predictions = [
        column for column in (prediction_id_col, *run_cols) if column not in predictions
    ]
    if missing_predictions:
        return {
            "status": "unavailable",
            "reason": f"prediction key columns missing: {missing_predictions}",
        }
    if ledger[ledger_required].isna().any().any():
        raise ValueError("membership ledger evaluation keys contain missing values")
    if predictions[[prediction_id_col, *run_cols]].isna().any().any():
        return {
            "status": "unavailable",
            "reason": "prediction evaluation keys contain missing values",
        }
    evaluation_labels = {"test", "eval", "evaluation"}
    evaluation = ledger.loc[
        ledger[split_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(evaluation_labels),
        [*run_cols, ledger_id_col],
    ].copy()
    predicted = predictions[[*run_cols, prediction_id_col]].copy()
    evaluation.columns = [*run_cols, "_row_id"]
    predicted.columns = [*run_cols, "_row_id"]
    duplicate_evaluation_rows = int(evaluation.duplicated(keep=False).sum())
    duplicate_prediction_rows = int(predicted.duplicated(keep=False).sum())
    evaluation_keys = {
        tuple(str(value) for value in row)
        for row in evaluation.drop_duplicates().itertuples(index=False, name=None)
    }
    prediction_keys = {
        tuple(str(value) for value in row)
        for row in predicted.drop_duplicates().itertuples(index=False, name=None)
    }
    ledger_only = evaluation_keys - prediction_keys
    prediction_only = prediction_keys - evaluation_keys
    return {
        "status": (
            "passed"
            if not ledger_only
            and not prediction_only
            and not duplicate_evaluation_rows
            and not duplicate_prediction_rows
            else "failed"
        ),
        "run_columns": list(run_cols),
        "evaluation_keys": len(evaluation_keys),
        "prediction_keys": len(prediction_keys),
        "ledger_only_keys": len(ledger_only),
        "prediction_only_keys": len(prediction_only),
        "duplicate_evaluation_rows": duplicate_evaluation_rows,
        "duplicate_prediction_rows": duplicate_prediction_rows,
        "ledger_only_examples": [list(item) for item in sorted(ledger_only)[:20]],
        "prediction_only_examples": [list(item) for item in sorted(prediction_only)[:20]],
    }


class FitnessState(str, Enum):
    """Ordered, finite scientific-fitness states."""

    NO_FAILURE_DETECTED = "no_failure_detected"
    WARNING = "warning"
    FAILURE_DETECTED = "failure_detected"
    OPERATIONALLY_UNUSABLE = "operationally_unusable"


@dataclass(frozen=True)
class FitnessCriteria:
    """User criteria; ``None`` disables an individual threshold."""

    min_picp_failure: float | None = None
    min_picp_warning: float | None = None
    min_worst_group_picp_failure: float | None = None
    min_worst_group_picp_warning: float | None = None
    max_unbounded_rate_failure: float | None = None
    max_unbounded_rate_warning: float | None = None
    unusable_unbounded_rate: float | None = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1 or None")
        if (
            self.min_picp_failure is not None
            and self.min_picp_warning is not None
            and self.min_picp_warning < self.min_picp_failure
        ):
            raise ValueError("min_picp_warning must be at least min_picp_failure")
        if (
            self.min_worst_group_picp_failure is not None
            and self.min_worst_group_picp_warning is not None
            and self.min_worst_group_picp_warning < self.min_worst_group_picp_failure
        ):
            raise ValueError(
                "min_worst_group_picp_warning must be at least "
                "min_worst_group_picp_failure"
            )
        if (
            self.max_unbounded_rate_failure is not None
            and self.max_unbounded_rate_warning is not None
            and self.max_unbounded_rate_warning > self.max_unbounded_rate_failure
        ):
            raise ValueError(
                "max_unbounded_rate_warning must not exceed max_unbounded_rate_failure"
            )

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class FitnessObservation:
    """Worst observed audit metrics used by the state engine."""

    min_picp: float | None = None
    min_worst_group_picp: float | None = None
    max_unbounded_rate: float | None = None
    unbounded_sources: Mapping[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_picp": self.min_picp,
            "min_worst_group_picp": self.min_worst_group_picp,
            "max_unbounded_rate": self.max_unbounded_rate,
            "unbounded_sources": dict(self.unbounded_sources or {}),
        }


@dataclass(frozen=True)
class FitnessAssessment:
    """Scientific state, evidence, and applied criteria."""

    state: FitnessState | None
    observation: FitnessObservation
    criteria: FitnessCriteria
    reasons: tuple[dict[str, Any], ...]
    contract_failures: tuple[str, ...]
    user_criteria_supplied: bool = False

    def to_dict(self) -> dict[str, Any]:
        state_value = "not_assessed" if self.state is None else self.state.value
        rank = None if self.state is None else {
            FitnessState.NO_FAILURE_DETECTED: 0,
            FitnessState.WARNING: 1,
            FitnessState.FAILURE_DETECTED: 2,
            FitnessState.OPERATIONALLY_UNUSABLE: 3,
        }[self.state]
        return {
            "state": state_value,
            "action_state": None if self.state is None else self.state.value,
            "state_rank": rank,
            "observation": self.observation.to_dict(),
            "criteria": self.criteria.to_dict(),
            "reasons": list(self.reasons),
            "contract_failures": list(self.contract_failures),
            "assessment_status": (
                "assessed" if self.state is not None else "not_assessed"
            ),
            "mode": "scientific_fitness" if self.state is not None else "metric_only",
            "user_criteria_supplied": self.user_criteria_supplied,
            "state_interpretation": (
                "Action state evaluated against user criteria."
                if self.state is not None
                else "Metrics are reported, but no scientific-fitness action state was assigned."
            ),
        }


def assess_fitness(
    observation: FitnessObservation,
    criteria: FitnessCriteria,
    *,
    contract_failures: Sequence[str] = (),
    user_criteria_supplied: bool | None = None,
    assessment_enabled: bool | None = None,
) -> FitnessAssessment:
    """Apply user thresholds or return an explicit metric-only assessment.

    The four :class:`FitnessState` values are emitted only when user criteria
    are supplied and assessment is enabled.  Otherwise ``state`` is ``None``
    and the machine-readable state is ``not_assessed``.
    """

    if user_criteria_supplied is None:
        criteria_values = criteria.to_dict()
        user_criteria_supplied = any(
            value is not None
            for name, value in criteria_values.items()
            if name != "unusable_unbounded_rate"
        ) or criteria.unusable_unbounded_rate not in (None, 0.0)
    if assessment_enabled is None:
        assessment_enabled = bool(user_criteria_supplied)
    failures = tuple(str(item) for item in contract_failures)
    if not (assessment_enabled and user_criteria_supplied):
        return FitnessAssessment(
            None,
            observation,
            criteria,
            (),
            failures,
            bool(user_criteria_supplied),
        )

    reasons: list[dict[str, Any]] = []

    def reason(
        severity: FitnessState,
        code: str,
        metric: str,
        observed: float | None,
        threshold: float | None,
        relation: str,
    ) -> None:
        reasons.append(
            {
                "severity": severity.value,
                "code": code,
                "metric": metric,
                "observed": observed,
                "threshold": threshold,
                "violation": relation,
            }
        )

    checks = [
        (
            "min_picp",
            observation.min_picp,
            criteria.min_picp_warning,
            criteria.min_picp_failure,
            "minimum",
        ),
        (
            "min_worst_group_picp",
            observation.min_worst_group_picp,
            criteria.min_worst_group_picp_warning,
            criteria.min_worst_group_picp_failure,
            "minimum",
        ),
        (
            "max_unbounded_rate",
            observation.max_unbounded_rate,
            criteria.max_unbounded_rate_warning,
            criteria.max_unbounded_rate_failure,
            "maximum",
        ),
    ]
    for metric, observed, warning_threshold, failure_threshold, direction in checks:
        if warning_threshold is None and failure_threshold is None:
            continue
        if observed is None or not math.isfinite(float(observed)):
            reason(
                FitnessState.OPERATIONALLY_UNUSABLE,
                "required_metric_unavailable",
                metric,
                observed,
                failure_threshold if failure_threshold is not None else warning_threshold,
                "required metric is unavailable or non-finite",
            )
            continue
        failure = (
            observed < failure_threshold
            if direction == "minimum" and failure_threshold is not None
            else observed > failure_threshold
            if direction == "maximum" and failure_threshold is not None
            else False
        )
        warning = (
            observed < warning_threshold
            if direction == "minimum" and warning_threshold is not None
            else observed > warning_threshold
            if direction == "maximum" and warning_threshold is not None
            else False
        )
        if failure:
            reason(
                FitnessState.FAILURE_DETECTED,
                f"{metric}_failure",
                metric,
                observed,
                failure_threshold,
                f"observed {direction} criterion failed",
            )
        elif warning:
            reason(
                FitnessState.WARNING,
                f"{metric}_warning",
                metric,
                observed,
                warning_threshold,
                f"observed {direction} warning criterion failed",
            )

    unusable = criteria.unusable_unbounded_rate
    observed_unbounded = observation.max_unbounded_rate
    if unusable is not None:
        if observed_unbounded is None or not math.isfinite(float(observed_unbounded)):
            reason(
                FitnessState.OPERATIONALLY_UNUSABLE,
                "unbounded_rate_unavailable",
                "max_unbounded_rate",
                observed_unbounded,
                unusable,
                "unbounded-output usability cannot be assessed",
            )
        elif observed_unbounded > unusable:
            reason(
                FitnessState.OPERATIONALLY_UNUSABLE,
                "unbounded_output_operationally_unusable",
                "max_unbounded_rate",
                observed_unbounded,
                unusable,
                "observed rate exceeds the maximum operationally usable rate",
            )

    for failure in failures:
        reason(
            FitnessState.FAILURE_DETECTED,
            "audit_contract_failure",
            "membership_ledger",
            None,
            None,
            failure,
        )
    order = {
        FitnessState.NO_FAILURE_DETECTED: 0,
        FitnessState.WARNING: 1,
        FitnessState.FAILURE_DETECTED: 2,
        FitnessState.OPERATIONALLY_UNUSABLE: 3,
    }
    state = max(
        (FitnessState(item["severity"]) for item in reasons),
        key=order.__getitem__,
        default=FitnessState.NO_FAILURE_DETECTED,
    )
    return FitnessAssessment(
        state,
        observation,
        criteria,
        tuple(reasons),
        failures,
        bool(user_criteria_supplied),
    )
