"""Paired random-versus-group audit contrasts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONTRAST_METRICS = ("picp", "mpiw", "interval_score", "mae")


def random_vs_group_contrasts(
    replicate_metrics: pd.DataFrame,
    *,
    pair_cols: Sequence[str],
    design_col: str = "design",
    random_label: str = "random_marginal",
    group_label: str = "spatial_target_holdout",
    metric_cols: Sequence[str] = DEFAULT_CONTRAST_METRICS,
) -> pd.DataFrame:
    """Build deterministic paired contrasts between two audit designs."""

    pairs = list(pair_cols)
    metrics = list(metric_cols)
    required = [*pairs, design_col, *metrics]
    missing = [column for column in required if column not in replicate_metrics]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    subset = replicate_metrics.loc[
        replicate_metrics[design_col].isin([random_label, group_label]), required
    ].copy()
    if subset.duplicated([*pairs, design_col]).any():
        raise ValueError("each pair must contain at most one row per design")
    sentinel = "__single_pair__"
    while sentinel in subset.columns:
        sentinel += "_"
    pivot_pairs = pairs
    if not pivot_pairs:
        subset[sentinel] = 0
        pivot_pairs = [sentinel]
    wide = subset.pivot(index=pivot_pairs, columns=design_col, values=metrics)
    for metric in metrics:
        for label in (random_label, group_label):
            if (metric, label) not in wide.columns:
                raise ValueError(f"missing design {label!r} for metric {metric!r}")
    wide.columns = [f"{metric}_{design}" for metric, design in wide.columns]
    wide = wide.reset_index()
    if pairs:
        wide = wide.sort_values(pairs, kind="mergesort").reset_index(drop=True)
    else:
        wide = wide.drop(columns=sentinel).reset_index(drop=True)
    if "picp" in metrics:
        wide["random_minus_group_picp"] = (
            wide[f"picp_{random_label}"] - wide[f"picp_{group_label}"]
        )
    for metric in ("mpiw", "interval_score", "mae", "rmse"):
        if metric in metrics:
            wide[f"group_minus_random_{metric}"] = (
                wide[f"{metric}_{group_label}"] - wide[f"{metric}_{random_label}"]
            )
    return wide


def summarize_design_replicates(
    replicate_metrics: pd.DataFrame,
    *,
    scenario_cols: Sequence[str],
    design_col: str = "design",
    metric_cols: Sequence[str] = DEFAULT_CONTRAST_METRICS,
    failure_col: str | None = "coverage_failure_detected",
) -> pd.DataFrame:
    """Mean and 2.5/97.5 percentiles by scenario and design."""

    keys = [*scenario_cols, design_col]
    required = [*keys, *metric_cols]
    if failure_col is not None:
        required.append(failure_col)
    missing = [column for column in required if column not in replicate_metrics]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    rows: list[dict[str, Any]] = []
    grouper: Any = keys[0] if len(keys) == 1 else keys
    for key, group in replicate_metrics.groupby(grouper, sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        record: dict[str, Any] = dict(zip(keys, values, strict=True))
        record["replicates"] = int(len(group))
        if failure_col is not None:
            record["failure_detection_rate"] = float(group[failure_col].astype(bool).mean())
        for metric in metric_cols:
            array = group[metric].to_numpy(float)
            record[f"mean_{metric}"] = float(np.mean(array))
            record[f"p025_{metric}"] = float(np.quantile(array, 0.025))
            record[f"p975_{metric}"] = float(np.quantile(array, 0.975))
        rows.append(record)
    return pd.DataFrame(rows)
