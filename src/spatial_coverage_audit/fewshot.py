"""Deterministic target-only few-shot recalibration ledgers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conformal import exact_split_conformal_quantile


def stable_seed(seed: int, *parts: object) -> int:
    """Stable 32-bit seed independent of Python's randomized hash."""

    token = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "little")


@dataclass(frozen=True)
class FewShotRecalibrationResult:
    calibration_ledger: pd.DataFrame
    runs: pd.DataFrame


def build_fewshot_recalibration_ledger(
    target_predictions: pd.DataFrame,
    *,
    group_col: str,
    row_id_col: str,
    calibration_sizes: Sequence[int],
    repeats: int,
    alpha: float = 0.10,
    seed: int = 20260716,
    score_col: str | None = None,
    y_col: str = "y_true",
    point_col: str = "y_pred",
) -> FewShotRecalibrationResult:
    """Sample target labels and return an auditable calibration ledger.

    Rows are sorted by group and row ID before sampling, so results do not
    depend on input row order. Sizes too small for a finite exact quantile are
    allowed and produce an explicitly unbounded run.
    """

    sizes = sorted({int(value) for value in calibration_sizes})
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("calibration_sizes must contain positive integers")
    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    required = [group_col, row_id_col]
    if score_col is None:
        required.extend([y_col, point_col])
    else:
        required.append(score_col)
    missing = [column for column in required if column not in target_predictions]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    data = target_predictions[required].copy()
    if data[[group_col, row_id_col]].isna().any().any():
        raise ValueError("group and row identifiers may not be missing")
    if data.duplicated([group_col, row_id_col]).any():
        raise ValueError("row IDs must be unique within each group")
    if score_col is None:
        data["_score"] = np.abs(data[y_col].to_numpy(float) - data[point_col].to_numpy(float))
    else:
        data["_score"] = pd.to_numeric(data[score_col], errors="coerce")
    if not np.isfinite(data["_score"].to_numpy(float)).all():
        raise ValueError("few-shot scores must be finite")
    data["_group_sort"] = data[group_col].astype(str)
    data["_id_sort"] = data[row_id_col].astype(str)
    data = data.sort_values(["_group_sort", "_id_sort"], kind="mergesort")
    ledger_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for group_value, group in data.groupby(group_col, sort=True, dropna=False):
        group = group.reset_index(drop=True)
        for size in sizes:
            if size >= len(group):
                raise ValueError(
                    f"calibration size {size} leaves no target test rows in group {group_value!r}"
                )
            for repeat in range(repeats):
                rng = np.random.default_rng(stable_seed(seed, "fewshot", group_value, size, repeat))
                positions = np.sort(rng.choice(len(group), size=size, replace=False))
                selected = group.iloc[positions]
                q = exact_split_conformal_quantile(selected["_score"].to_numpy(float), alpha)
                for _, row in selected.iterrows():
                    ledger_rows.append(
                        {
                            group_col: group_value,
                            "target_calibration_n": size,
                            "repeat": repeat,
                            row_id_col: row[row_id_col],
                            "score": float(row["_score"]),
                            "calibration_q": q,
                            "unbounded": bool(math.isinf(q)),
                        }
                    )
                run_rows.append(
                    {
                        group_col: group_value,
                        "target_calibration_n": size,
                        "target_test_n": int(len(group) - size),
                        "repeat": repeat,
                        "calibration_q": q,
                        "unbounded": bool(math.isinf(q)),
                        "unbounded_interval_rate": 1.0 if math.isinf(q) else 0.0,
                    }
                )
    ledger = pd.DataFrame(ledger_rows).sort_values(
        [group_col, "target_calibration_n", "repeat", row_id_col], kind="mergesort"
    ).reset_index(drop=True)
    runs = pd.DataFrame(run_rows).sort_values(
        [group_col, "target_calibration_n", "repeat"], kind="mergesort"
    ).reset_index(drop=True)
    if ledger.duplicated([group_col, "target_calibration_n", "repeat", row_id_col]).any():
        raise RuntimeError("duplicated few-shot calibration membership")
    return FewShotRecalibrationResult(ledger, runs)

