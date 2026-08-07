"""Evaluation metrics, reported by demand tier.

Spec §7.2, as corrected against real data. Percentage-based errors are
meaningless where a zone averages under two pickups an hour — the denominator is
frequently zero and the ratio explodes — yet those zones are 138 of 217. So the
metric set changes by tier rather than one metric being forced everywhere.

Poisson deviance is the primary metric: it matches the training objective,
handles zero counts without special-casing, and does not blow up on small
denominators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Predictions are clipped away from zero before any log is taken. Poisson
#: deviance is undefined at a zero prediction, and a model may legitimately
#: predict arbitrarily close to it in a dead zone.
MIN_PREDICTION = 1e-6


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error.

    Undefined when the actuals sum to zero, which happens in dead zones — hence
    reporting it only for the high and medium tiers.
    """
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Poisson deviance.

    The ``y log(y/yhat)`` term is taken as zero where ``y`` is zero, which is its
    limit and the reason this metric copes with the 46% of grid cells that hold
    no demand at all.
    """
    pred = np.clip(y_pred, MIN_PREDICTION, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(y_true > 0, y_true * np.log(y_true / pred), 0.0)
    return float(np.mean(2 * (log_term - (y_true - pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def volume_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Total predicted demand over total actual demand; 1.0 is unbiased.

    MAE cannot see this. A model minimising MAE on skewed count data estimates
    the conditional *median*, which for right-skewed demand sits below the mean,
    so it can post an excellent MAE while systematically under-forecasting total
    volume. For a system that positions vehicles, that bias is the failure —
    consistent under-supply — and it is invisible in every error metric that
    averages absolute deviations.
    """
    total = np.sum(y_true)
    if total == 0:
        return float("nan")
    return float(np.sum(y_pred) / total)


def zero_hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Share of true zeros the model predicts as near-zero.

    The low tier is mostly zeros, so a model that never predicts one is useless
    there however good its average error looks.
    """
    zeros = y_true == 0
    if not zeros.any():
        return float("nan")
    return float(np.mean(y_pred[zeros] < threshold))


@dataclass(frozen=True)
class TierMetrics:
    """Metrics for one demand tier."""

    tier: str
    n: int
    mae: float
    rmse: float
    poisson_deviance: float
    wape: float
    zero_hit_rate: float
    mean_actual: float
    mean_predicted: float
    volume_bias: float

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "n": self.n,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "poisson_deviance": round(self.poisson_deviance, 4),
            "wape": None if np.isnan(self.wape) else round(self.wape, 4),
            "zero_hit_rate": (
                None if np.isnan(self.zero_hit_rate) else round(self.zero_hit_rate, 4)
            ),
            "mean_actual": round(self.mean_actual, 4),
            "mean_predicted": round(self.mean_predicted, 4),
            "volume_bias": (None if np.isnan(self.volume_bias) else round(self.volume_bias, 4)),
        }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, tier: str = "all") -> TierMetrics:
    return TierMetrics(
        tier=tier,
        n=int(len(y_true)),
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        poisson_deviance=poisson_deviance(y_true, y_pred),
        wape=wape(y_true, y_pred),
        zero_hit_rate=zero_hit_rate(y_true, y_pred),
        mean_actual=float(np.mean(y_true)),
        mean_predicted=float(np.mean(y_pred)),
        volume_bias=volume_bias(y_true, y_pred),
    )


def evaluate_by_tier(
    y_true: np.ndarray, y_pred: np.ndarray, tiers: np.ndarray
) -> list[TierMetrics]:
    """Overall metrics plus a breakdown by demand tier.

    The gated tiers (spec §8) are high and medium; low is reported so its
    behaviour is visible, not so it can be optimised against.
    """
    results = [evaluate(y_true, y_pred, tier="all")]
    for tier in ("high", "medium", "low"):
        mask = tiers == tier
        if mask.any():
            results.append(evaluate(y_true[mask], y_pred[mask], tier=tier))
    return results


def relative_improvement(champion: float, benchmark: float) -> float:
    """Fractional improvement of champion over benchmark; positive is better.

    Lower is better for every metric here, so improvement is a reduction.
    """
    if benchmark == 0:
        return float("nan")
    return (benchmark - champion) / benchmark
