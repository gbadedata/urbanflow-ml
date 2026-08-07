"""Tests for prediction intervals.

The previous implementation passed its coverage gate while producing a lower
bound pinned at zero: total coverage hit 94.7% because the interval had become
one-sided, and nothing checked the tails. Several tests here exist specifically
so that failure cannot recur silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from urbanflow.models.intervals import (
    MAX_DISPERSION,
    NOMINAL_COVERAGE,
    CoverageResult,
    check_coverage_gate,
    estimate_dispersion,
    interval_bounds,
    measure_coverage,
    measure_coverage_by_tier,
)

# --- dispersion --------------------------------------------------------------


def test_poisson_data_yields_no_overdispersion() -> None:
    rng = np.random.default_rng(0)
    mu = np.full(50000, 20.0)
    y = rng.poisson(mu).astype(float)
    assert estimate_dispersion(y, mu) > 1000


def test_overdispersed_data_yields_finite_dispersion() -> None:
    """Real demand varies more than Poisson allows; k must detect that."""
    rng = np.random.default_rng(1)
    mu = np.full(50000, 20.0)
    y = rng.poisson(mu * rng.gamma(10, 0.1, 50000)).astype(float)
    k = estimate_dispersion(y, mu)
    assert 1.0 < k < 100.0, f"implausible dispersion estimate {k}"


def test_dispersion_never_negative_or_zero() -> None:
    y = np.array([1.0, 1.0, 1.0])
    assert estimate_dispersion(y, y) == MAX_DISPERSION


# --- interval construction ---------------------------------------------------


def test_bounds_are_ordered_and_non_negative() -> None:
    mu = np.array([0.0, 0.5, 5.0, 200.0])
    lower, upper = interval_bounds(mu, 20.0)
    assert np.all(lower >= 0.0)
    assert np.all(upper >= lower)


def test_bounds_are_integers() -> None:
    """A lower bound of 3.7 cabs is not actionable."""
    lower, upper = interval_bounds(np.array([7.3, 88.2]), 20.0)
    assert np.all(lower == np.floor(lower))
    assert np.all(upper == np.floor(upper))


def test_high_demand_gets_a_positive_lower_bound() -> None:
    """The regression that motivated this rewrite.

    A zone averaging 100 pickups an hour must not be given a lower bound of zero.
    """
    lower, _ = interval_bounds(np.array([100.0]), 20.0)
    assert lower[0] > 20.0, f"lower bound collapsed to {lower[0]}"


def test_sparse_demand_may_have_a_zero_lower_bound() -> None:
    """At 0.3 pickups an hour, zero genuinely is the 5th percentile."""
    lower, _ = interval_bounds(np.array([0.3]), 20.0)
    assert lower[0] == 0.0


def test_intervals_widen_with_demand() -> None:
    lower, upper = interval_bounds(np.array([1.0, 10.0, 100.0]), 20.0)
    widths = upper - lower
    assert widths[0] < widths[1] < widths[2]


def test_overdispersion_widens_intervals() -> None:
    """Smaller k means more unexplained variability and wider intervals."""
    mu = np.array([100.0])
    tight = np.subtract(*reversed(interval_bounds(mu, 1000.0)))
    loose = np.subtract(*reversed(interval_bounds(mu, 5.0)))
    assert loose[0] > tight[0]


def test_poisson_fallback_when_no_overdispersion() -> None:
    lower, upper = interval_bounds(np.array([50.0]), MAX_DISPERSION)
    assert upper[0] > lower[0] > 0


# --- coverage measurement ----------------------------------------------------


def test_perfect_and_zero_coverage() -> None:
    y = np.array([5.0, 10.0])
    assert measure_coverage(y, np.zeros(2), np.full(2, 100.0)).empirical == 1.0
    assert measure_coverage(y, np.full(2, 50.0), np.full(2, 60.0)).empirical == 0.0


def test_bounds_are_inclusive() -> None:
    y = np.array([1.0, 9.0])
    assert measure_coverage(y, np.array([1.0, 1.0]), np.array([9.0, 9.0])).empirical == 1.0


def test_tail_rates_reported_separately() -> None:
    y = np.array([0.0, 5.0, 5.0, 20.0])
    result = measure_coverage(y, np.full(4, 1.0), np.full(4, 10.0))
    assert result.empirical == 0.5
    assert result.below_lower == 0.25
    assert result.above_upper == 0.25


def test_coverage_by_tier_splits() -> None:
    y = np.array([1.0, 2.0, 50.0, 60.0])
    tiers = np.array(["low", "low", "high", "high"])
    by_tier = {r.tier: r for r in measure_coverage_by_tier(y, np.zeros(4), np.full(4, 10.0), tiers)}
    assert by_tier["low"].empirical == 1.0
    assert by_tier["high"].empirical == 0.0


# --- the gate ----------------------------------------------------------------


def _result(tier: str, empirical: float, below: float, above: float) -> CoverageResult:
    return CoverageResult(tier, 1000, 0.90, empirical, 10.0, 10.0, below, above)


def test_gate_passes_on_well_centred_intervals() -> None:
    results = [_result("high", 0.90, 0.05, 0.05), _result("medium", 0.91, 0.04, 0.05)]
    passed, _ = check_coverage_gate(results, ["high", "medium"])
    assert passed


def test_gate_catches_the_one_sided_interval() -> None:
    """The exact failure that shipped: 94.7% coverage, 0.0% below the bound.

    Total coverage alone passes; the tail check is what rejects it.
    """
    results = [_result("high", 0.947, 0.0, 0.053)]
    passed, reasons = check_coverage_gate(results, ["high"], tolerance=0.06)
    assert not passed
    assert any("not usable for planning" in r for r in reasons)


def test_gate_fails_on_under_coverage() -> None:
    results = [_result("high", 0.71, 0.20, 0.09)]
    passed, reasons = check_coverage_gate(results, ["high"])
    assert not passed
    assert any("under-covers" in r for r in reasons)


def test_gate_fails_on_uninformative_wide_intervals() -> None:
    results = [_result("high", 0.999, 0.0, 0.001)]
    passed, reasons = check_coverage_gate(results, ["high"])
    assert not passed
    assert any("over-covers" in r for r in reasons)


def test_low_tier_not_gated() -> None:
    """A zero lower bound is genuinely correct where demand is mostly zero."""
    results = [_result("high", 0.90, 0.05, 0.05), _result("low", 0.40, 0.60, 0.0)]
    passed, _ = check_coverage_gate(results, ["high", "medium"])
    assert passed


def test_nominal_coverage_constant() -> None:
    assert pytest.approx(0.90) == NOMINAL_COVERAGE


# --- end to end --------------------------------------------------------------


def test_intervals_cover_overdispersed_counts() -> None:
    """The whole point: fitted dispersion should restore nominal coverage."""
    rng = np.random.default_rng(3)
    n = 100_000
    mu = np.where(rng.random(n) < 0.3, 110.0, 3.0)
    y = rng.poisson(mu * rng.gamma(20, 1 / 20, n)).astype(float)

    half = n // 2
    k = estimate_dispersion(y[:half], mu[:half])
    lower, upper = interval_bounds(mu[half:], k)
    result = measure_coverage(y[half:], lower, upper)

    assert 0.85 < result.empirical < 0.96, f"coverage {result.empirical:.3f}"
    assert result.below_lower > 0.01, "lower bound is not doing any work"
