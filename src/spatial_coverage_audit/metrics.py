"""Coverage, sharpness, and proper interval-score summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


WILSON_95_Z = 1.959963984540054


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    return alpha


def _float_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _validated_intervals(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _float_vector(y_true, "y_true")
    lo = _float_vector(lower, "lower")
    hi = _float_vector(upper, "upper")
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError("y_true, lower, and upper must have the same length")
    if not np.isfinite(y).all():
        raise ValueError("y_true must contain only finite values")
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("interval endpoints may be infinite but not NaN")
    if (hi < lo).any():
        raise ValueError("every interval must satisfy upper >= lower")
    with np.errstate(invalid="ignore"):
        width = hi - lo
    if np.isnan(width).any():
        raise ValueError("interval width is indeterminate for equal infinite endpoints")
    return y, lo, hi


def coverage_indicators(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Inclusive interval-coverage indicators."""

    y, lo, hi = _validated_intervals(y_true, lower, upper)
    return (y >= lo) & (y <= hi)


def interval_widths(
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Interval widths, preserving infinity for unbounded intervals."""

    lo = _float_vector(lower, "lower")
    hi = _float_vector(upper, "upper")
    if len(lo) != len(hi):
        raise ValueError("lower and upper must have the same length")
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("interval endpoints may be infinite but not NaN")
    if (hi < lo).any():
        raise ValueError("every interval must satisfy upper >= lower")
    with np.errstate(invalid="ignore"):
        width = hi - lo
    if np.isnan(width).any():
        raise ValueError("interval width is indeterminate for equal infinite endpoints")
    return width


def interval_scores(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Central prediction-interval score at miscoverage level ``alpha``."""

    alpha = _validate_alpha(alpha)
    y, lo, hi = _validated_intervals(y_true, lower, upper)
    score = hi - lo
    below = y < lo
    above = y > hi
    score = score.copy()
    score[below] += (2.0 / alpha) * (lo[below] - y[below])
    score[above] += (2.0 / alpha) * (y[above] - hi[above])
    return score


def root_mean_square_error(
    y_true: Sequence[float] | np.ndarray,
    point_prediction: Sequence[float] | np.ndarray,
) -> float:
    """Root mean square error for finite, equally sized vectors."""

    y = _float_vector(y_true, "y_true")
    point = _float_vector(point_prediction, "point_prediction")
    if len(y) != len(point) or not np.isfinite(y).all() or not np.isfinite(point).all():
        raise ValueError("y_true and point_prediction must be finite and have equal length")
    return float(np.sqrt(np.mean(np.square(y - point)))) if len(y) else math.nan


def wilson_interval(
    successes: int, n: int, z: float = WILSON_95_Z
) -> tuple[float, float]:
    """Wilson binomial interval; returns ``(nan, nan)`` when ``n == 0``."""

    successes = int(successes)
    n = int(n)
    z = float(z)
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("require 0 <= successes <= n")
    if n == 0:
        return math.nan, math.nan
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    ) / denominator
    return centre - half, centre + half


def _finite_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if finite.any() else math.nan


def summarize_intervals(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    alpha: float,
    point_prediction: Sequence[float] | np.ndarray | None = None,
    *,
    include_rmse: bool = False,
) -> dict[str, Any]:
    """Return coverage, Wilson, width, score, and unbounded-rate summaries.

    Overall mean width and interval score deliberately remain infinite when at
    least one interval is unbounded. Conditional finite-only means are named
    separately and must not be interpreted as overall sharpness.
    """

    alpha = _validate_alpha(alpha)
    y, lo, hi = _validated_intervals(y_true, lower, upper)
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    score = interval_scores(y, lo, hi, alpha)
    finite_interval = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(width)
    successes = int(covered.sum())
    n = int(len(y))
    ci_low, ci_high = wilson_interval(successes, n)
    result: dict[str, Any] = {
        "n": n,
        "covered_n": successes,
        "picp": float(np.mean(covered)) if n else math.nan,
        "picp_wilson_low": float(ci_low),
        "picp_wilson_high": float(ci_high),
        "nominal_coverage": 1.0 - alpha,
        "coverage_failure_detected": bool(ci_high < 1.0 - alpha) if n else False,
        "mpiw": float(np.mean(width)) if n else math.nan,
        "mpiw_finite": _finite_mean(width),
        "interval_score": float(np.mean(score)) if n else math.nan,
        "interval_score_finite": _finite_mean(score),
        "finite_interval_n": int(finite_interval.sum()),
        "unbounded_interval_n": int((~finite_interval).sum()),
        "unbounded_interval_rate": float(np.mean(~finite_interval)) if n else math.nan,
    }
    if point_prediction is not None:
        point = _float_vector(point_prediction, "point_prediction")
        if len(point) != n or not np.isfinite(point).all():
            raise ValueError("point_prediction must be finite and match y_true")
        result["mae"] = float(np.mean(np.abs(y - point))) if n else math.nan
        if include_rmse:
            result["rmse"] = root_mean_square_error(y, point)
    return result
