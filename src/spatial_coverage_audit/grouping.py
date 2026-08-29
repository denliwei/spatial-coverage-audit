"""Group/domain summaries with explicit macro, micro, and worst-group metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import summarize_intervals


def _columns(frame: pd.DataFrame, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def strata_interval_summaries(
    predictions: pd.DataFrame,
    strata_cols: Sequence[str],
    *,
    y_col: str = "y_true",
    lower_col: str = "lower",
    upper_col: str = "upper",
    point_col: str | None = "y_pred",
    alpha: float = 0.10,
    include_rmse: bool = False,
) -> pd.DataFrame:
    """Interval summaries for arbitrary analysis strata."""

    strata = list(strata_cols)
    required = [*strata, y_col, lower_col, upper_col]
    if point_col is not None:
        required.append(point_col)
    _columns(predictions, required)
    rows: list[dict[str, Any]] = []
    if strata:
        grouper: Any = strata[0] if len(strata) == 1 else strata
        groups = predictions.groupby(grouper, sort=True, dropna=False)
    else:
        groups = [((), predictions)]
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        record = dict(zip(strata, keys, strict=True))
        point = None if point_col is None else group[point_col].to_numpy(float)
        record.update(
            summarize_intervals(
                group[y_col].to_numpy(float),
                group[lower_col].to_numpy(float),
                group[upper_col].to_numpy(float),
                alpha,
                point,
                include_rmse=include_rmse,
            )
        )
        rows.append(record)
    return pd.DataFrame(rows)


def group_interval_summaries(
    predictions: pd.DataFrame,
    group_col: str,
    *,
    strata_cols: Sequence[str] = (),
    y_col: str = "y_true",
    lower_col: str = "lower",
    upper_col: str = "upper",
    point_col: str | None = "y_pred",
    alpha: float = 0.10,
    include_rmse: bool = False,
) -> pd.DataFrame:
    """One interval-summary row per stratum and observed group."""

    _columns(predictions, [group_col])
    if predictions[group_col].isna().any():
        raise ValueError(f"declared group column contains missing values: {group_col}")

    return strata_interval_summaries(
        predictions,
        [*strata_cols, group_col],
        y_col=y_col,
        lower_col=lower_col,
        upper_col=upper_col,
        point_col=point_col,
        alpha=alpha,
        include_rmse=include_rmse,
    )


def _mean_preserve_infinity(values: pd.Series) -> float:
    array = values.to_numpy(float)
    return float(np.mean(array)) if len(array) else math.nan


def aggregate_group_metrics(
    predictions: pd.DataFrame,
    group_table: pd.DataFrame,
    group_col: str,
    *,
    strata_cols: Sequence[str] = (),
    y_col: str = "y_true",
    lower_col: str = "lower",
    upper_col: str = "upper",
    point_col: str | None = "y_pred",
    alpha: float = 0.10,
    include_rmse: bool = False,
) -> pd.DataFrame:
    """Unweighted macro, pooled micro, and worst-group summaries.

    ``group_table`` should come from :func:`group_interval_summaries` with the
    same strata. Overall macro width/score preserve infinity. Finite-only macro
    columns average the explicitly labelled finite-only group columns.
    """

    strata = list(strata_cols)
    _columns(group_table, [*strata, group_col, "picp", "mpiw", "mpiw_finite", "interval_score", "interval_score_finite", "unbounded_interval_rate"])
    rows: list[dict[str, Any]] = []
    if strata:
        grouper: Any = strata[0] if len(strata) == 1 else strata
        prediction_groups = predictions.groupby(grouper, sort=True, dropna=False)
    else:
        prediction_groups = [((), predictions)]
    for key, prediction_subset in prediction_groups:
        keys = key if isinstance(key, tuple) else (key,)
        selector = np.ones(len(group_table), dtype=bool)
        for column, value in zip(strata, keys, strict=True):
            matches = group_table[column].isna() if pd.isna(value) else group_table[column].eq(value)
            selector &= matches.to_numpy()
        group_subset = group_table.loc[selector].copy()
        if group_subset.empty:
            raise ValueError(f"group_table has no rows for stratum {keys}")
        micro = summarize_intervals(
            prediction_subset[y_col].to_numpy(float),
            prediction_subset[lower_col].to_numpy(float),
            prediction_subset[upper_col].to_numpy(float),
            alpha,
            None if point_col is None else prediction_subset[point_col].to_numpy(float),
            include_rmse=include_rmse,
        )
        ranked = group_subset.assign(_group_text=group_subset[group_col].astype(str)).sort_values(
            ["picp", "_group_text"], kind="mergesort"
        )
        worst = ranked.iloc[0]
        record: dict[str, Any] = dict(zip(strata, keys, strict=True))
        record.update(
            {
                "groups": int(len(group_subset)),
                "n": int(len(prediction_subset)),
                "macro_picp": float(group_subset["picp"].mean()),
                "macro_mpiw": _mean_preserve_infinity(group_subset["mpiw"]),
                "macro_mpiw_finite": float(group_subset["mpiw_finite"].mean()),
                "macro_interval_score": _mean_preserve_infinity(group_subset["interval_score"]),
                "macro_interval_score_finite": float(group_subset["interval_score_finite"].mean()),
                "macro_unbounded_interval_rate": float(group_subset["unbounded_interval_rate"].mean()),
                "worst_group": worst[group_col],
                "worst_group_picp": float(worst["picp"]),
                "micro_picp": micro["picp"],
                "micro_picp_wilson_low": micro["picp_wilson_low"],
                "micro_picp_wilson_high": micro["picp_wilson_high"],
                "micro_mpiw": micro["mpiw"],
                "micro_mpiw_finite": micro["mpiw_finite"],
                "micro_interval_score": micro["interval_score"],
                "micro_interval_score_finite": micro["interval_score_finite"],
                "micro_unbounded_interval_rate": micro["unbounded_interval_rate"],
            }
        )
        if point_col is not None:
            record["macro_mae"] = float(group_subset["mae"].mean())
            record["micro_mae"] = micro["mae"]
            if include_rmse:
                if "rmse" not in group_subset:
                    raise ValueError("group_table lacks rmse; build it with include_rmse=True")
                record["macro_rmse"] = float(group_subset["rmse"].mean())
                record["micro_rmse"] = micro["rmse"]
        rows.append(record)
    return pd.DataFrame(rows)
