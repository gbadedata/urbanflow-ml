"""Prediction intervals by quantile regression.

Spec §7.3, as corrected. Split conformal prediction assumes exchangeability,
which a time series violates: the calibration residuals come from a different
period than the ones being covered, so nominal 90% intervals under-cover in
practice. Rather than reach for a blocked conformal variant, this fits gradient
boosters directly on the pinball loss at each quantile — no extra machinery, and
honest about being empirical rather than guaranteed.

Coverage is then *measured* on held-out data rather than assumed. An interval
that claims 90% and delivers 71% is worse than no interval, because a dispatcher
plans against it.

Quantile crossing is the one artefact worth handling explicitly. The models are
fitted independently, so nothing stops the 0.95 model predicting below the 0.05
model in a sparse zone. Crossed quantiles are sorted back into order rather than
left to produce a negative-width interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from urbanflow.models.baselines import design_matrix

logger = logging.getLogger(__name__)

#: Nominal interval. 0.05/0.95 gives a 90% central interval.
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95
NOMINAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE


@dataclass
class QuantileModel:
    """A pair of boosters bracketing the central interval."""

    lower: object
    upper: object
    columns: list[str]
    lower_quantile: float = LOWER_QUANTILE
    upper_quantile: float = UPPER_QUANTILE

    @property
    def nominal_coverage(self) -> float:
        return self.upper_quantile - self.lower_quantile

    def predict_interval(self, features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper), clipped at zero and sorted into order."""
        matrix = design_matrix(features, self.columns)
        low = np.asarray(self.lower.predict(matrix), dtype=np.float64)  # type: ignore[attr-defined]
        high = np.asarray(self.upper.predict(matrix), dtype=np.float64)  # type: ignore[attr-defined]

        low = np.clip(low, 0.0, None)
        high = np.clip(high, 0.0, None)

        # Independently fitted quantiles can cross in sparse zones; sorting is
        # the standard repair and keeps the interval non-negative in width.
        crossed = low > high
        if crossed.any():
            logger.info(
                "repaired %s crossed quantile pairs (%.3f%%)",
                f"{crossed.sum():,}",
                100 * crossed.mean(),
            )
            low[crossed], high[crossed] = high[crossed], low[crossed].copy()

        return low, high


def fit_quantiles(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    columns: list[str],
    *,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
    max_iter: int = 300,
    random_state: int = 0,
) -> QuantileModel:
    """Fit both interval bounds on the training block only."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from urbanflow.models.champion import CATEGORICAL_FEATURES

    matrix = design_matrix(features, columns)
    categorical = [c in CATEGORICAL_FEATURES for c in columns]

    def _fit(quantile: float) -> HistGradientBoostingRegressor:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=max_iter,
            learning_rate=0.1,
            max_leaf_nodes=63,
            min_samples_leaf=100,
            l2_regularization=1.0,
            categorical_features=categorical,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )
        model.fit(matrix, target)
        logger.info("quantile %.2f fitted in %d iterations", quantile, model.n_iter_)
        return model

    return QuantileModel(
        lower=_fit(lower_quantile),
        upper=_fit(upper_quantile),
        columns=columns,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )


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
    """Empirical coverage and interval width.

    Tail misses are reported separately: an interval can hit its nominal
    coverage while being badly asymmetric, missing almost entirely on one side,
    which matters operationally because under-supply and over-supply are not
    equally costly.
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
) -> tuple[bool, list[str]]:
    """Spec §8 gate 4: 90% intervals must cover 85-95% on the gated tiers.

    The low tier is reported but not gated. Demand there is mostly zero, so a
    lower bound of zero covers a large share of observations trivially and the
    coverage figure says little about the model.
    """
    reasons: list[str] = []
    passed = True

    for result in results:
        if result.tier not in gated_tiers:
            continue
        if abs(result.error) > tolerance:
            passed = False
            direction = "under" if result.error < 0 else "over"
            reasons.append(
                f"{result.tier}: {result.empirical:.1%} empirical coverage against "
                f"{result.nominal:.0%} nominal — {direction}-covers by "
                f"{abs(result.error):.1%}, outside the ±{tolerance:.0%} tolerance"
            )
        else:
            reasons.append(f"{result.tier}: {result.empirical:.1%} coverage, within tolerance")

    return passed, reasons


def format_coverage(results: list[CoverageResult], horizon: int) -> str:
    lines = [
        f"\nhorizon {horizon}h — interval coverage",
        "-" * 74,
        f"{'tier':<9}{'n':>10}{'nominal':>10}{'empirical':>12}"
        f"{'mean width':>12}{'below':>9}{'above':>9}",
        "-" * 74,
    ]
    for r in results:
        lines.append(
            f"{r.tier:<9}{r.n:>10,}{r.nominal:>10.0%}{r.empirical:>12.1%}"
            f"{r.mean_width:>12.2f}{r.below_lower:>9.1%}{r.above_upper:>9.1%}"
        )
    return "\n".join(lines)
