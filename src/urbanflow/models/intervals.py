"""Prediction intervals from the fitted count distribution.

Spec §7.3, revised after measurement. Three approaches were considered:

* **Split conformal** — assumes exchangeability, which a time series violates.
  Rejected before implementation.
* **Quantile gradient boosting** — implemented, measured, abandoned. With 46% of
  the grid at zero demand, pinball loss at q=0.05 starts from the pooled 5th
  percentile (which is 0) and never escapes it: any upward step is charged 0.95
  against every zero. The result was a lower bound pinned at zero even in zones
  averaging 100 pickups an hour. Total coverage looked fine — 94.7% against a 90%
  target — because the interval had silently become one-sided ``[0, p95]``, whose
  true nominal coverage is 95%. It passed the gate for the wrong reason.
* **Distribution-based (this module)** — the champion already produces a
  well-calibrated mean, so intervals follow from the count distribution around
  it. No second model, and no optimiser that has to cooperate.

Counts are overdispersed: variance exceeds the mean, because demand is driven by
factors the features do not capture. Poisson intervals therefore under-cover
badly at high volume. A negative binomial with a single dispersion parameter
fixes that, and the parameter is worth reporting in its own right — it
quantifies how much unexplained variability remains.

Dispersion is fitted on the calibrate block, never on train and never on the
block where coverage is verified. That is what the calibrate block is for
(spec §6.2); fitting and verifying on the same rows produces an optimistic curve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95
NOMINAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE

#: Above this the fitted distribution is effectively Poisson, and the negative
#: binomial parameterisation becomes numerically awkward.
MAX_DISPERSION = 1e6


def estimate_dispersion(y_true: np.ndarray, mu: np.ndarray) -> float:
    """Estimate the negative binomial dispersion ``k`` by moment matching.

    The negative binomial has ``Var = mu + mu^2 / k``, so ``k`` follows from the
    excess of squared residuals over the Poisson variance. Large ``k`` means the
    data is nearly Poisson; small ``k`` means substantial overdispersion.
    """
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-9, None)
    excess = float(np.sum((y_true - mu) ** 2 - mu))
    if excess <= 0:
        logger.info("no overdispersion detected; falling back to Poisson intervals")
        return MAX_DISPERSION
    k = float(np.sum(mu**2) / excess)
    return float(np.clip(k, 1e-3, MAX_DISPERSION))


def interval_bounds(
    mu: np.ndarray,
    dispersion: float,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Central interval from the count distribution around each predicted mean.

    Bounds are integers because the forecast quantity is a count of pickups; a
    lower bound of 3.7 cabs is not something an operator can act on.
    """
    from scipy.stats import nbinom, poisson

    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-9, None)

    if dispersion >= MAX_DISPERSION:
        lower = poisson.ppf(lower_quantile, mu)
        upper = poisson.ppf(upper_quantile, mu)
    else:
        p = dispersion / (dispersion + mu)
        lower = nbinom.ppf(lower_quantile, dispersion, p)
        upper = nbinom.ppf(upper_quantile, dispersion, p)

    lower = np.clip(np.nan_to_num(np.asarray(lower, dtype=np.float64), nan=0.0), 0.0, None)
    upper = np.clip(np.nan_to_num(np.asarray(upper, dtype=np.float64), nan=0.0), 0.0, None)
    return lower, np.maximum(upper, lower)


@dataclass(frozen=True)
class CoverageResult:
    """Measured interval behaviour for one tier."""

    tier: str
    n: int
    nominal: float
    empirical: float
    mean_width: float
    median_width: float
    below_lower: float
    above_upper: float

    @property
    def error(self) -> float:
        return self.empirical - self.nominal

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "n": self.n,
            "nominal_coverage": round(self.nominal, 4),
            "empirical_coverage": round(self.empirical, 4),
            "coverage_error": round(self.error, 4),
            "mean_width": round(self.mean_width, 4),
            "median_width": round(self.median_width, 4),
            "below_lower": round(self.below_lower, 4),
            "above_upper": round(self.above_upper, 4),
        }


def measure_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float = NOMINAL_COVERAGE,
    tier: str = "all",
) -> CoverageResult:
    """Empirical coverage, interval width, and both tail miss rates.

    The tail rates are not decoration. Total coverage alone cannot distinguish a
    correctly centred interval from a one-sided one that reaches its number by
    never missing below — the exact failure that let the previous implementation
    pass its gate while being useless for planning.
    """
    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    return CoverageResult(
        tier=tier,
        n=int(len(y_true)),
        nominal=nominal,
        empirical=float(np.mean(inside)),
        mean_width=float(np.mean(width)),
        median_width=float(np.median(width)),
        below_lower=float(np.mean(y_true < lower)),
        above_upper=float(np.mean(y_true > upper)),
    )


def measure_coverage_by_tier(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tiers: np.ndarray,
    nominal: float = NOMINAL_COVERAGE,
) -> list[CoverageResult]:
    results = [measure_coverage(y_true, lower, upper, nominal, "all")]
    for tier in ("high", "medium", "low"):
        mask = tiers == tier
        if mask.any():
            results.append(measure_coverage(y_true[mask], lower[mask], upper[mask], nominal, tier))
    return results


def check_coverage_gate(
    results: list[CoverageResult],
    gated_tiers: list[str],
    tolerance: float = 0.05,
    tail_tolerance: float = 0.04,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
) -> tuple[bool, list[str]]:
    """Spec §8 gate 4, with a tail check added after the one-sided-interval bug.

    Total coverage is necessary but not sufficient. An interval whose lower bound
    sits at zero never misses below, so it can post 95% coverage while saying
    nothing about the downside — and minimum-staffing decisions taken against
    that bound would be badly wrong. Each tail is therefore checked against its
    own nominal rate.

    The low tier is reported but not gated: demand there is mostly zero, so a
    zero lower bound is genuinely correct and its tail rate is uninformative.
    """
    reasons: list[str] = []
    passed = True
    nominal_below = lower_quantile
    nominal_above = 1.0 - upper_quantile

    for result in results:
        if result.tier not in gated_tiers:
            continue

        if abs(result.error) > tolerance:
            passed = False
            direction = "under" if result.error < 0 else "over"
            reasons.append(
                f"{result.tier}: {result.empirical:.1%} coverage against "
                f"{result.nominal:.0%} nominal — {direction}-covers by {abs(result.error):.1%}"
            )
        else:
            reasons.append(f"{result.tier}: {result.empirical:.1%} coverage, within tolerance")

        if abs(result.below_lower - nominal_below) > tail_tolerance:
            passed = False
            reasons.append(
                f"{result.tier}: {result.below_lower:.1%} of actuals fall below the lower "
                f"bound against {nominal_below:.0%} expected — the interval is not centred "
                f"and the lower bound is not usable for planning"
            )
        if abs(result.above_upper - nominal_above) > tail_tolerance:
            passed = False
            reasons.append(
                f"{result.tier}: {result.above_upper:.1%} of actuals exceed the upper bound "
                f"against {nominal_above:.0%} expected"
            )

    return passed, reasons


def format_coverage(results: list[CoverageResult], horizon: int, dispersion: float) -> str:
    lines = [
        f"\nhorizon {horizon}h — interval coverage (dispersion k={dispersion:.2f})",
        "-" * 78,
        f"{'tier':<9}{'n':>10}{'nominal':>10}{'empirical':>12}"
        f"{'mean width':>12}{'below':>9}{'above':>9}",
        "-" * 78,
    ]
    for r in results:
        lines.append(
            f"{r.tier:<9}{r.n:>10,}{r.nominal:>10.0%}{r.empirical:>12.1%}"
            f"{r.mean_width:>12.2f}{r.below_lower:>9.1%}{r.above_upper:>9.1%}"
        )
    return "\n".join(lines)
