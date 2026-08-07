"""Tests for prediction intervals.

An interval that claims 90% and delivers 71% is worse than no interval, because
a dispatcher plans against it. So coverage is measured rather than assumed, and
these tests pin the measurement itself before it is trusted on real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanflow.models.intervals import (
    NOMINAL_COVERAGE,
    QuantileModel,
    check_coverage_gate,
    fit_quantiles,
    measure_coverage,
    measure_coverage_by_tier,
)

# --- coverage measurement ----------------------------------------------------


def test_perfect_coverage_measured_as_one() -> None:
    y = np.array([5.0, 10.0, 15.0])
    result = measure_coverage(y, np.zeros(3), np.full(3, 100.0))
    assert result.empirical == 1.0
    assert result.below_lower == 0.0
    assert result.above_upper == 0.0


def test_zero_coverage_measured_as_zero() -> None:
    y = np.array([50.0, 60.0])
    result = measure_coverage(y, np.zeros(2), np.ones(2))
    assert result.empirical == 0.0
    assert result.above_upper == 1.0


def test_bounds_are_inclusive() -> None:
    """A value exactly on the boundary is covered."""
    y = np.array([1.0, 9.0])
    result = measure_coverage(y, np.array([1.0, 1.0]), np.array([9.0, 9.0]))
    assert result.empirical == 1.0


def test_tail_misses_reported_separately() -> None:
    """Coverage can hit its target while being badly asymmetric."""
    y = np.array([0.0, 5.0, 5.0, 20.0])
    lower = np.full(4, 1.0)
    upper = np.full(4, 10.0)
    result = measure_coverage(y, lower, upper)
    assert result.empirical == 0.5
    assert result.below_lower == 0.25
    assert result.above_upper == 0.25


def test_interval_width_reported() -> None:
    result = measure_coverage(np.array([5.0, 5.0]), np.array([0.0, 2.0]), np.array([10.0, 4.0]))
    assert result.mean_width == pytest.approx(6.0)
    assert result.median_width == pytest.approx(6.0)


def test_coverage_error_signed() -> None:
    """Negative error means under-coverage, the dangerous direction."""
    y = np.arange(100.0)
    result = measure_coverage(y, np.full(100, 10.0), np.full(100, 60.0), nominal=0.90)
    assert result.error < 0


def test_coverage_by_tier_splits() -> None:
    y = np.array([1.0, 2.0, 50.0, 60.0])
    lower, upper = np.zeros(4), np.full(4, 10.0)
    tiers = np.array(["low", "low", "high", "high"])

    results = measure_coverage_by_tier(y, lower, upper, tiers)
    by_tier = {r.tier: r for r in results}

    assert by_tier["low"].empirical == 1.0
    assert by_tier["high"].empirical == 0.0
    assert by_tier["all"].empirical == 0.5


# --- the gate ----------------------------------------------------------------


def test_gate_passes_within_tolerance() -> None:
    results = [
        measure_coverage(np.arange(100.0), np.full(100, -1.0), np.full(100, 91.0)),
    ]
    results = [
        type(results[0])(
            tier=tier,
            n=1000,
            nominal=0.90,
            empirical=empirical,
            mean_width=5.0,
            median_width=5.0,
            below_lower=0.05,
            above_upper=0.05,
        )
        for tier, empirical in (("high", 0.89), ("medium", 0.93))
    ]
    passed, reasons = check_coverage_gate(results, ["high", "medium"])
    assert passed
    assert len(reasons) == 2


def test_gate_fails_on_under_coverage() -> None:
    from urbanflow.models.intervals import CoverageResult

    results = [
        CoverageResult("high", 1000, 0.90, 0.71, 5.0, 5.0, 0.20, 0.09),
        CoverageResult("medium", 1000, 0.90, 0.91, 5.0, 5.0, 0.05, 0.04),
    ]
    passed, reasons = check_coverage_gate(results, ["high", "medium"])
    assert not passed
    assert any("under-covers" in r for r in reasons)


def test_gate_fails_on_over_coverage() -> None:
    """Intervals so wide they always contain the truth are uninformative."""
    from urbanflow.models.intervals import CoverageResult

    results = [CoverageResult("high", 1000, 0.90, 0.999, 500.0, 500.0, 0.0, 0.001)]
    passed, reasons = check_coverage_gate(results, ["high"])
    assert not passed
    assert any("over-covers" in r for r in reasons)


def test_low_tier_not_gated() -> None:
    """Mostly-zero demand makes a zero lower bound cover trivially."""
    from urbanflow.models.intervals import CoverageResult

    results = [
        CoverageResult("high", 1000, 0.90, 0.90, 5.0, 5.0, 0.05, 0.05),
        CoverageResult("low", 1000, 0.90, 0.40, 1.0, 1.0, 0.60, 0.0),
    ]
    passed, _ = check_coverage_gate(results, ["high", "medium"])
    assert passed


# --- fitted intervals --------------------------------------------------------


def test_fitted_intervals_approximately_cover() -> None:
    """The central check: a 90% interval should cover roughly 90%."""
    rng = np.random.default_rng(0)
    n = 20000
    recent = rng.poisson(20.0, n).astype(float)
    target = rng.poisson(np.clip(recent, 0.1, None)).astype(float)
    features = {"recent_1": recent}

    model = fit_quantiles(features, target, ["recent_1"], max_iter=80)
    lower, upper = model.predict_interval(features)
    result = measure_coverage(target, lower, upper)

    assert 0.82 < result.empirical < 0.97, f"coverage {result.empirical:.3f} implausible"


def test_intervals_are_ordered_and_non_negative() -> None:
    rng = np.random.default_rng(1)
    recent = rng.poisson(1.0, 4000).astype(float)
    target = rng.poisson(1.0, 4000).astype(float)
    features = {"recent_1": recent}

    model = fit_quantiles(features, target, ["recent_1"], max_iter=50)
    lower, upper = model.predict_interval(features)

    assert np.all(lower >= 0.0), "a negative lower bound is not actionable"
    assert np.all(upper >= lower), "crossed quantiles survived the repair"


def test_crossed_quantiles_are_repaired() -> None:
    """Independently fitted quantiles can cross; sorting is the repair."""

    class _Const:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, matrix: np.ndarray) -> np.ndarray:
            return np.full(len(matrix), self.value)

    model = QuantileModel(lower=_Const(9.0), upper=_Const(2.0), columns=["recent_1"])
    lower, upper = model.predict_interval({"recent_1": np.zeros(5)})

    assert np.all(lower == 2.0)
    assert np.all(upper == 9.0)


def test_wider_intervals_for_noisier_demand() -> None:
    """Interval width should track uncertainty, not be constant."""
    rng = np.random.default_rng(2)
    n = 20000
    level = np.where(rng.random(n) < 0.5, 100.0, 2.0)
    recent = rng.poisson(level).astype(float)
    target = rng.poisson(level).astype(float)
    features = {"recent_1": recent}

    model = fit_quantiles(features, target, ["recent_1"], max_iter=80)
    lower, upper = model.predict_interval(features)
    width = upper - lower

    busy = width[level == 100.0].mean()
    quiet = width[level == 2.0].mean()
    assert busy > quiet, "intervals do not widen with demand"


def test_nominal_coverage_constant() -> None:
    assert pytest.approx(0.90) == NOMINAL_COVERAGE
