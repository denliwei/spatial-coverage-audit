"""Exact ordinary and target-point weighted conformal quantiles."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    return alpha


def conformal_rank(n: int, alpha: float) -> int:
    """One-based finite-sample conformal rank ``ceil((n+1)*(1-alpha))``."""

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    alpha = _alpha(alpha)
    return int(math.ceil((n + 1) * (1.0 - alpha)))


def _finite_scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not len(values):
        raise ValueError("calibration scores must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("calibration scores must contain only finite values")
    return values


def exact_split_conformal_quantile(
    scores: Sequence[float] | np.ndarray, alpha: float
) -> float:
    """Exact non-randomized split-conformal quantile.

    If the finite-sample rank exceeds the calibration size, infinity is
    returned. The rank is never clipped to the largest observed score.
    """

    values = _finite_scores(scores)
    rank = conformal_rank(len(values), alpha)
    if rank > len(values):
        return math.inf
    return float(np.partition(values, rank - 1)[rank - 1])


@dataclass(frozen=True)
class WeightedConformalResult:
    """Per-target weighted quantiles and their infinity masses."""

    quantiles: np.ndarray
    target_infinity_mass: np.ndarray
    unbounded_mask: np.ndarray
    calibration_rows: int = 0
    positive_weight_rows: int = 0
    zero_weight_rows: int = 0

    @property
    def unbounded_rate(self) -> float:
        return float(np.mean(self.unbounded_mask)) if len(self.unbounded_mask) else math.nan


def target_point_weighted_quantiles(
    scores: Sequence[float] | np.ndarray,
    calibration_weights: Sequence[float] | np.ndarray,
    target_weights: Sequence[float] | np.ndarray,
    alpha: float,
) -> WeightedConformalResult:
    """Weighted quantile with each target weight placed at infinity.

    For target ``j``, the threshold is
    ``(1-alpha) * (sum(calibration_weights) + target_weight[j])``. If the
    finite calibration mass cannot reach it, that target's quantile is
    infinity.
    """

    alpha = _alpha(alpha)
    score = np.asarray(scores, dtype=float)
    calibration = np.asarray(calibration_weights, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    if score.ndim != 1 or calibration.ndim != 1 or target.ndim != 1:
        raise ValueError("scores and weights must be one-dimensional")
    if len(score) != len(calibration):
        raise ValueError("scores and calibration_weights must have the same length")
    if not np.isfinite(score).all():
        raise ValueError("scores must contain only finite values")
    if not np.isfinite(calibration).all() or (calibration < 0.0).any():
        raise ValueError("calibration_weights must be finite and non-negative")
    positive = calibration > 0.0
    calibration_rows = int(len(calibration))
    zero_weight_rows = int((~positive).sum())
    score = score[positive]
    calibration = calibration[positive]
    if not len(score):
        raise ValueError("no calibration score has a finite positive weight")
    if not np.isfinite(target).all() or (target < 0.0).any():
        raise ValueError("target_weights must be finite and non-negative")
    order = np.argsort(score, kind="mergesort")
    score = score[order]
    calibration = calibration[order]
    cumulative = np.cumsum(calibration)
    total = float(cumulative[-1])
    threshold = (1.0 - alpha) * (total + target)
    unbounded = threshold > total
    quantiles = np.full(len(target), math.inf, dtype=float)
    finite_target = ~unbounded
    if finite_target.any():
        locations = np.searchsorted(cumulative, threshold[finite_target], side="left")
        quantiles[finite_target] = score[np.minimum(locations, len(score) - 1)]
    infinity_mass = target / (total + target)
    return WeightedConformalResult(
        quantiles,
        infinity_mass,
        unbounded,
        calibration_rows=calibration_rows,
        positive_weight_rows=int(len(calibration)),
        zero_weight_rows=zero_weight_rows,
    )


def effective_sample_size(weights: Sequence[float] | np.ndarray) -> float:
    """Kish effective sample size for finite non-negative weights."""

    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("weights must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all() or (values < 0.0).any() or values.sum() <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    return float(values.sum() ** 2 / np.square(values).sum())
