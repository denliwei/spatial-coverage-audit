"""Command-line audit for generic prediction and calibration CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conformal import (
    conformal_rank,
    effective_sample_size,
    exact_split_conformal_quantile,
    target_point_weighted_quantiles,
)
from .contrasts import random_vs_group_contrasts
from .contracts import (
    FitnessCriteria,
    FitnessObservation,
    assess_fitness,
    validate_membership_ledger,
    validate_prediction_evaluation_alignment,
)
from .fewshot import build_fewshot_recalibration_ledger
from .grouping import aggregate_group_metrics, group_interval_summaries, strata_interval_summaries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--y-col", default="y_true")
    parser.add_argument("--lower-col", default="lower")
    parser.add_argument("--upper-col", default="upper")
    parser.add_argument("--point-col", default="y_pred")
    parser.add_argument("--group-col")
    parser.add_argument("--method-col")
    parser.add_argument("--repeat-col")
    parser.add_argument("--design-col")
    parser.add_argument("--pair-cols", help="Comma-separated pairing keys for design contrasts")
    parser.add_argument("--random-design-label", default="random_marginal")
    parser.add_argument("--group-design-label", default="spatial_target_holdout")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--score-col")
    parser.add_argument("--calibration-y-col", default="y_true")
    parser.add_argument("--calibration-point-col", default="y_pred")
    parser.add_argument("--calibration-weight-col")
    parser.add_argument("--target-weight-col")
    parser.add_argument("--id-col", default="row_id")
    parser.add_argument("--fewshot-sizes", type=int, nargs="+")
    parser.add_argument("--fewshot-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--include-rmse",
        action="store_true",
        help="Add RMSE to point-prediction, group, and contrast outputs",
    )
    parser.add_argument("--membership-ledger", type=Path)
    parser.add_argument("--membership-id-col", default="row_id")
    parser.add_argument("--membership-split-col", default="membership")
    parser.add_argument(
        "--membership-run-cols",
        help="Comma-separated run keys, for example method,repeat,fold",
    )
    parser.add_argument("--membership-group-col", default="group")
    parser.add_argument("--membership-target-group-col")
    parser.add_argument("--complete-group-holdout", action="store_true")
    parser.add_argument("--fitness-min-picp", type=float)
    parser.add_argument("--fitness-warning-picp", type=float)
    parser.add_argument("--fitness-min-worst-group-picp", type=float)
    parser.add_argument("--fitness-warning-worst-group-picp", type=float)
    parser.add_argument("--fitness-max-unbounded-rate", type=float)
    parser.add_argument("--fitness-warning-unbounded-rate", type=float)
    parser.add_argument(
        "--fitness-unusable-unbounded-rate",
        type=float,
        help=(
            "Explicit maximum usable unbounded-output rate; once assessment is "
            "enabled, the default is 0 (any unbounded output is unusable)"
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _scores(frame: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    if args.score_col:
        if args.score_col not in frame:
            raise ValueError(f"calibration score column not found: {args.score_col}")
        values = pd.to_numeric(frame[args.score_col], errors="coerce").to_numpy(float)
    else:
        missing = [
            column
            for column in (args.calibration_y_col, args.calibration_point_col)
            if column not in frame
        ]
        if missing:
            raise ValueError(f"calibration columns missing: {missing}")
        values = np.abs(
            pd.to_numeric(frame[args.calibration_y_col], errors="coerce").to_numpy(float)
            - pd.to_numeric(frame[args.calibration_point_col], errors="coerce").to_numpy(float)
        )
    if not np.isfinite(values).any():
        raise ValueError("calibration contains no finite scores")
    return values


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(prediction_path)
    include_rmse = bool(getattr(args, "include_rmse", False))
    declared_keys = [
        column
        for column in (args.group_col, args.method_col, args.repeat_col, args.design_col)
        if column
    ]
    missing_declared_keys = [column for column in declared_keys if column not in predictions]
    if missing_declared_keys:
        raise ValueError(f"declared audit key columns missing: {missing_declared_keys}")
    null_declared_keys = [column for column in declared_keys if predictions[column].isna().any()]
    if null_declared_keys:
        raise ValueError(f"declared audit key columns contain missing values: {null_declared_keys}")
    point_col = args.point_col if args.point_col in predictions else None
    strata = [column for column in (args.method_col, args.repeat_col, args.design_col) if column]
    overall = strata_interval_summaries(
        predictions,
        strata,
        y_col=args.y_col,
        lower_col=args.lower_col,
        upper_col=args.upper_col,
        point_col=point_col,
        alpha=args.alpha,
        include_rmse=include_rmse,
    )
    artifacts: dict[str, str] = {}
    overall_path = output_dir / "overall_summary.csv"
    overall.to_csv(overall_path, index=False)
    artifacts["overall_summary"] = str(overall_path)

    group_table: pd.DataFrame | None = None
    aggregate: pd.DataFrame | None = None
    if args.group_col:
        group_table = group_interval_summaries(
            predictions,
            args.group_col,
            strata_cols=strata,
            y_col=args.y_col,
            lower_col=args.lower_col,
            upper_col=args.upper_col,
            point_col=point_col,
            alpha=args.alpha,
            include_rmse=include_rmse,
        )
        aggregate = aggregate_group_metrics(
            predictions,
            group_table,
            args.group_col,
            strata_cols=strata,
            y_col=args.y_col,
            lower_col=args.lower_col,
            upper_col=args.upper_col,
            point_col=point_col,
            alpha=args.alpha,
            include_rmse=include_rmse,
        )
        group_path = output_dir / "group_summary.csv"
        aggregate_path = output_dir / "group_aggregate_summary.csv"
        group_table.to_csv(group_path, index=False)
        aggregate.to_csv(aggregate_path, index=False)
        artifacts.update(group_summary=str(group_path), group_aggregate_summary=str(aggregate_path))

    contrast_table: pd.DataFrame | None = None
    if args.design_col:
        if args.pair_cols is None:
            pair_cols = [
                column for column in (args.method_col, args.repeat_col) if column
            ]
        else:
            pair_cols = [item.strip() for item in args.pair_cols.split(",") if item.strip()]
        design_metrics = strata_interval_summaries(
            predictions,
            [*pair_cols, args.design_col],
            y_col=args.y_col,
            lower_col=args.lower_col,
            upper_col=args.upper_col,
            point_col=point_col,
            alpha=args.alpha,
            include_rmse=include_rmse,
        )
        metric_cols = ["picp", "mpiw", "interval_score"]
        if point_col is not None:
            metric_cols.append("mae")
            if include_rmse:
                metric_cols.append("rmse")
        contrast_table = random_vs_group_contrasts(
            design_metrics,
            pair_cols=pair_cols,
            design_col=args.design_col,
            random_label=args.random_design_label,
            group_label=args.group_design_label,
            metric_cols=metric_cols,
        )
        contrast_path = output_dir / "random_vs_group_contrasts.csv"
        contrast_table.to_csv(contrast_path, index=False)
        artifacts["random_vs_group_contrasts"] = str(contrast_path)

    calibration_summary: dict[str, Any] | None = None
    if args.calibration:
        calibration_path = args.calibration.resolve()
        calibration = pd.read_csv(calibration_path)
        scores = _scores(calibration, args)
        finite_n = int(np.isfinite(scores).sum())
        exact_q = exact_split_conformal_quantile(scores, args.alpha)
        calibration_summary = {
            "input": str(calibration_path),
            "input_sha256": _sha256(calibration_path),
            "finite_scores": finite_n,
            "exact_rank": conformal_rank(finite_n, args.alpha),
            "exact_q": exact_q,
            "exact_q_unbounded": bool(math.isinf(exact_q)),
            "exact_q_status": (
                "valid_unbounded_finite_sample_rank"
                if math.isinf(exact_q)
                else "finite"
            ),
        }
        if bool(args.calibration_weight_col) != bool(args.target_weight_col):
            raise ValueError(
                "calibration-weight-col and target-weight-col must be supplied together"
            )
        if args.calibration_weight_col:
            if args.calibration_weight_col not in calibration:
                raise ValueError(f"missing calibration weight column: {args.calibration_weight_col}")
            if args.target_weight_col not in predictions:
                raise ValueError(f"missing target weight column: {args.target_weight_col}")
            calibration_weights = pd.to_numeric(
                calibration[args.calibration_weight_col], errors="coerce"
            ).to_numpy(float)
            target_weights = pd.to_numeric(
                predictions[args.target_weight_col], errors="coerce"
            ).to_numpy(float)
            weighted = target_point_weighted_quantiles(
                scores, calibration_weights, target_weights, args.alpha
            )
            weighted_table = pd.DataFrame(
                {
                    args.id_col: (
                        predictions[args.id_col].astype(str)
                        if args.id_col in predictions
                        else np.arange(len(predictions)).astype(str)
                    ),
                    "target_weight": target_weights,
                    "target_infinity_mass": weighted.target_infinity_mass,
                    "weighted_q": weighted.quantiles,
                    "unbounded": weighted.unbounded_mask,
                }
            )
            weighted_path = output_dir / "target_weighted_quantiles.csv"
            weighted_table.to_csv(weighted_path, index=False)
            artifacts["target_weighted_quantiles"] = str(weighted_path)
            valid_calibration_weights = calibration_weights[
                np.isfinite(scores) & np.isfinite(calibration_weights) & (calibration_weights > 0)
            ]
            calibration_summary["weighted"] = {
                "calibration_weight_ess": effective_sample_size(valid_calibration_weights),
                "calibration_rows": weighted.calibration_rows,
                "positive_weight_rows": weighted.positive_weight_rows,
                "zero_weight_rows": weighted.zero_weight_rows,
                "target_n": int(len(target_weights)),
                "unbounded_n": int(weighted.unbounded_mask.sum()),
                "unbounded_rate": weighted.unbounded_rate,
            }
    else:
        calibration_path = None

    fewshot_summary: dict[str, Any] | None = None
    if args.fewshot_sizes:
        if not args.group_col:
            raise ValueError("few-shot recalibration requires --group-col")
        if args.id_col not in predictions:
            raise ValueError(f"few-shot row ID column not found: {args.id_col}")
        if point_col is None:
            raise ValueError("few-shot recalibration requires a point-prediction column")
        fewshot = build_fewshot_recalibration_ledger(
            predictions,
            group_col=args.group_col,
            row_id_col=args.id_col,
            calibration_sizes=args.fewshot_sizes,
            repeats=args.fewshot_repeats,
            alpha=args.alpha,
            seed=args.seed,
            y_col=args.y_col,
            point_col=point_col,
        )
        ledger_path = output_dir / "fewshot_calibration_ledger.csv"
        runs_path = output_dir / "fewshot_recalibration_runs.csv"
        fewshot.calibration_ledger.to_csv(ledger_path, index=False)
        fewshot.runs.to_csv(runs_path, index=False)
        artifacts.update(
            fewshot_calibration_ledger=str(ledger_path),
            fewshot_recalibration_runs=str(runs_path),
        )
        fewshot_summary = {
            "ledger_rows": int(len(fewshot.calibration_ledger)),
            "runs": int(len(fewshot.runs)),
            "unbounded_runs": int(fewshot.runs["unbounded"].sum()),
        }

    prediction_identifier_validation: dict[str, Any]
    identifier_keys = [
        column
        for column in (args.method_col, args.design_col, args.repeat_col, args.id_col)
        if column and column in predictions
    ]
    if args.id_col in predictions:
        duplicated = predictions.duplicated(identifier_keys, keep=False)
        duplicate_keys = predictions.loc[duplicated, identifier_keys].drop_duplicates()
        prediction_identifier_validation = {
            "checked": True,
            "key_columns": identifier_keys,
            "valid": not bool(duplicated.any()),
            "duplicate_rows": int(duplicated.sum()),
            "duplicate_keys": int(len(duplicate_keys)),
            "examples": duplicate_keys.head(20).astype(str).to_dict(orient="records"),
        }
    else:
        prediction_identifier_validation = {
            "checked": False,
            "key_columns": [],
            "valid": None,
            "duplicate_rows": None,
            "duplicate_keys": None,
            "examples": [],
            "reason": f"prediction ID column not found: {args.id_col}",
        }

    membership_summary: dict[str, Any] | None = None
    membership_path = getattr(args, "membership_ledger", None)
    membership_validation = None
    membership: pd.DataFrame | None = None
    run_cols: list[str] = []
    if membership_path:
        membership_path = membership_path.resolve()
        membership = pd.read_csv(membership_path)
        run_cols = [
            item.strip()
            for item in (getattr(args, "membership_run_cols", None) or "").split(",")
            if item.strip()
        ]
        membership_validation = validate_membership_ledger(
            membership,
            id_col=getattr(args, "membership_id_col", "row_id"),
            split_col=getattr(args, "membership_split_col", "membership"),
            run_cols=run_cols,
            group_col=(
                getattr(args, "membership_group_col", "group")
                if getattr(args, "complete_group_holdout", False)
                else None
            ),
            complete_group_holdout=bool(
                getattr(args, "complete_group_holdout", False)
            ),
            target_group_col=getattr(args, "membership_target_group_col", None),
        )
        issues_path = output_dir / "membership_validation_issues.csv"
        membership_validation.issues.to_csv(issues_path, index=False)
        artifacts["membership_validation_issues"] = str(issues_path)
        membership_summary = {
            "input": str(membership_path),
            "input_sha256": _sha256(membership_path),
            "run_columns": run_cols,
            **membership_validation.to_dict(include_issues=False),
            "issues_artifact": str(issues_path),
        }

    overall_picp = pd.to_numeric(overall.get("picp"), errors="coerce").to_numpy(float)
    finite_picp = overall_picp[np.isfinite(overall_picp)]
    min_picp = float(finite_picp.min()) if len(finite_picp) else None
    if aggregate is not None:
        worst_values = pd.to_numeric(
            aggregate.get("worst_group_picp"), errors="coerce"
        ).to_numpy(float)
        finite_worst = worst_values[np.isfinite(worst_values)]
        min_worst_group_picp = float(finite_worst.min()) if len(finite_worst) else None
    else:
        min_worst_group_picp = None
    unbounded_sources: dict[str, float] = {}
    empirical_unbounded = pd.to_numeric(
        overall.get("unbounded_interval_rate"), errors="coerce"
    ).to_numpy(float)
    empirical_unbounded = empirical_unbounded[np.isfinite(empirical_unbounded)]
    if len(empirical_unbounded):
        unbounded_sources["prediction_intervals"] = float(empirical_unbounded.max())
    if calibration_summary and calibration_summary.get("weighted"):
        weighted_rate = float(calibration_summary["weighted"]["unbounded_rate"])
        if math.isfinite(weighted_rate):
            unbounded_sources["target_weighted_quantiles"] = weighted_rate
    max_unbounded_rate = max(unbounded_sources.values()) if unbounded_sources else None
    unusable_override = getattr(args, "fitness_unusable_unbounded_rate", None)
    user_fitness_criteria_supplied = any(
        getattr(args, name, None) is not None
        for name in (
            "fitness_min_picp",
            "fitness_warning_picp",
            "fitness_min_worst_group_picp",
            "fitness_warning_worst_group_picp",
            "fitness_max_unbounded_rate",
            "fitness_warning_unbounded_rate",
            "fitness_unusable_unbounded_rate",
        )
    )
    criteria = FitnessCriteria(
        min_picp_failure=getattr(args, "fitness_min_picp", None),
        min_picp_warning=getattr(args, "fitness_warning_picp", None),
        min_worst_group_picp_failure=getattr(
            args, "fitness_min_worst_group_picp", None
        ),
        min_worst_group_picp_warning=getattr(
            args, "fitness_warning_worst_group_picp", None
        ),
        max_unbounded_rate_failure=getattr(
            args, "fitness_max_unbounded_rate", None
        ),
        max_unbounded_rate_warning=getattr(
            args, "fitness_warning_unbounded_rate", None
        ),
        unusable_unbounded_rate=(
            0.0 if unusable_override is None else unusable_override
        ),
    )
    contract_failures: list[str] = []
    if prediction_identifier_validation["valid"] is False:
        contract_failures.append(
            "duplicate prediction row_id membership within method/design/repeat"
        )
    if membership_validation is not None and not membership_validation.valid:
        contract_failures.extend(
            f"membership:{name}={count}"
            for name, count in membership_validation.issue_counts.items()
        )
    contract_verification = {
        "prediction_identifiers": (
            "passed"
            if prediction_identifier_validation["valid"] is True
            else "failed"
            if prediction_identifier_validation["valid"] is False
            else "not_checked"
        ),
        "membership_ledger": (
            "passed"
            if membership_validation is not None and membership_validation.valid
            else "failed"
            if membership_validation is not None
            else "not_supplied"
        ),
        "complete_group_holdout": (
            "passed"
            if membership_validation is not None
            and membership_validation.valid
            and membership_validation.complete_group_holdout
            else "failed"
            if membership_validation is not None
            and membership_validation.complete_group_holdout
            else "not_requested"
        ),
    }
    alignment_summary: dict[str, Any]
    if membership is None:
        alignment_summary = {"status": "unavailable", "reason": "membership_not_supplied"}
    else:
        alignment_summary = validate_prediction_evaluation_alignment(
            predictions,
            membership,
            prediction_id_col=args.id_col,
            ledger_id_col=getattr(args, "membership_id_col", "row_id"),
            split_col=getattr(args, "membership_split_col", "membership"),
            run_cols=run_cols,
        )
    contract_verification["prediction_evaluation_alignment"] = alignment_summary
    contract_verification["leakage_audit"] = (
        "passed"
        if contract_verification["complete_group_holdout"] == "passed"
        else "failed"
        if contract_verification["complete_group_holdout"] == "failed"
        else "unavailable"
    )
    contract_verification["full_contract_verified"] = bool(
        contract_verification["prediction_identifiers"] == "passed"
        and contract_verification["membership_ledger"] == "passed"
        and contract_verification["complete_group_holdout"] == "passed"
        and alignment_summary["status"] == "passed"
    )
    assessment_blockers: list[str] = []
    if not user_fitness_criteria_supplied:
        assessment_blockers.append("no_user_fitness_criteria")
    if contract_verification["prediction_identifiers"] != "passed":
        assessment_blockers.append("prediction_identifier_audit_unavailable_or_failed")
    if contract_verification["membership_ledger"] != "passed":
        assessment_blockers.append("membership_ledger_audit_unavailable_or_failed")
    if contract_verification["complete_group_holdout"] != "passed":
        assessment_blockers.append("complete_group_leakage_audit_unavailable_or_failed")
    if alignment_summary["status"] != "passed":
        assessment_blockers.append("prediction_evaluation_alignment_unavailable_or_failed")
    assessment_enabled = bool(
        user_fitness_criteria_supplied
        and contract_verification["full_contract_verified"]
    )
    fitness = assess_fitness(
        FitnessObservation(
            min_picp=min_picp,
            min_worst_group_picp=min_worst_group_picp,
            max_unbounded_rate=max_unbounded_rate,
            unbounded_sources=unbounded_sources,
        ),
        criteria,
        contract_failures=contract_failures,
        user_criteria_supplied=user_fitness_criteria_supplied,
        assessment_enabled=assessment_enabled,
    )
    fitness_output = {
        **fitness.to_dict(),
        "contract_verification": contract_verification,
        "assessment_blockers": assessment_blockers,
        "unbounded_policy": {
            "max_usable_rate": criteria.unusable_unbounded_rate,
            "source": "user_override" if unusable_override is not None else "default_zero",
        },
    }
    fitness_path = output_dir / "fitness_state.json"
    fitness_payload = {
        "run_status": "completed",
        "scientific_fitness": fitness_output,
        "prediction_identifier_validation": prediction_identifier_validation,
        "membership_validation": membership_summary,
    }
    fitness_path.write_text(
        json.dumps(_json_safe(fitness_payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    artifacts["fitness_state"] = str(fitness_path)

    result = {
        "status": "passed",
        "run_status": "completed",
        "audit_schema": 2,
        "scientific_fitness": fitness_output,
        "prediction_identifier_validation": prediction_identifier_validation,
        "membership_validation": membership_summary,
        "alpha": float(args.alpha),
        "nominal_coverage": 1.0 - float(args.alpha),
        "prediction_input": str(prediction_path),
        "prediction_input_sha256": _sha256(prediction_path),
        "prediction_rows": int(len(predictions)),
        "overall": overall.to_dict(orient="records"),
        "group_aggregate": None if aggregate is None else aggregate.to_dict(orient="records"),
        "calibration": calibration_summary,
        "fewshot": fewshot_summary,
        "artifacts": artifacts,
        "limitations": [
            "Grouped and spatial metrics are empirical audits, not group-conditional guarantees.",
            "Finite-only width and score condition on excluding unbounded intervals.",
            "Target-only few-shot calibration changes the deployment information set.",
        ],
    }
    result_path = output_dir / "audit_result.json"
    result_path.write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    result["result_path"] = str(result_path)
    return result


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = run_audit(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_status": result["run_status"],
                "scientific_state": result["scientific_fitness"]["state"],
                "scientific_fitness": result["scientific_fitness"]["assessment_status"],
                "full_contract_verified": result["scientific_fitness"][
                    "contract_verification"
                ]["full_contract_verified"],
                "result": result["result_path"],
            }
        )
    )


if __name__ == "__main__":
    main()
