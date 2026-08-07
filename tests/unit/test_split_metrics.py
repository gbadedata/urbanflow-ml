"""Tests for the temporal split and evaluation metrics.

The split is where leakage enters quietly: a gap that is too short, or blocks
that touch, produce metrics that look plausible and are optimistic. These tests
pin the boundaries rather than trusting the arithmetic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from urbanflow.config import Config, load_config
from urbanflow.models.metrics import (
    evaluate,
    evaluate_by_tier,
    mae,
    poisson_deviance,
    relative_improvement,
    wape,
    zero_hit_rate,
)
from urbanflow.models.split import (
    CALIBRATE,
    TEST,
    TRAIN,
    SplitBounds,
    compute_bounds,
)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    import yaml

    from urbanflow.config import DEFAULT_CONFIG_PATH

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    for key in ("raw_dir", "validated_dir", "quarantine_dir"):
        raw["data"][key] = str(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


FIRST = datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
LAST = datetime(2025, 6, 30, 23, 0, tzinfo=UTC)


# --- split boundaries --------------------------------------------------------


def test_blocks_are_ordered_and_disjoint(config: Config) -> None:
    bounds = compute_bounds(FIRST, LAST, config)
    assert bounds.train_start < bounds.train_end
    assert bounds.train_end < bounds.calibrate_start
    assert bounds.calibrate_end < bounds.test_start
    assert bounds.test_start < bounds.test_end


def test_test_block_is_the_most_recent_data(config: Config) -> None:
    bounds = compute_bounds(FIRST, LAST, config)
    assert bounds.test_end == LAST + timedelta(hours=1)


def test_test_block_has_full_declared_length(config: Config) -> None:
    """Gaps are carved from earlier blocks so the holdout keeps its length."""
    bounds = compute_bounds(FIRST, LAST, config)
    weeks = (bounds.test_end - bounds.test_start) / timedelta(weeks=1)
    assert weeks == pytest.approx(config.split.holdout_weeks)


def test_gap_is_at_least_the_longest_horizon(config: Config) -> None:
    """Otherwise a target in one block is a feature in another."""
    bounds = compute_bounds(FIRST, LAST, config)
    longest = timedelta(hours=max(config.forecast.horizons_hours))
    assert bounds.calibrate_start - bounds.train_end >= longest
    assert bounds.test_start - bounds.calibrate_end >= longest


def test_block_lookup_assigns_correctly(config: Config) -> None:
    bounds = compute_bounds(FIRST, LAST, config)
    assert bounds.block_for(bounds.train_start) == TRAIN
    assert bounds.block_for(bounds.calibrate_start) == CALIBRATE
    assert bounds.block_for(bounds.test_start) == TEST


def test_gap_hours_belong_to_no_block(config: Config) -> None:
    """A row in the gap must be excluded, not silently absorbed."""
    bounds = compute_bounds(FIRST, LAST, config)
    in_gap = bounds.train_end + timedelta(hours=1)
    assert in_gap < bounds.calibrate_start
    assert bounds.block_for(in_gap) is None


def test_end_bounds_are_exclusive(config: Config) -> None:
    bounds = compute_bounds(FIRST, LAST, config)
    assert bounds.block_for(bounds.train_end) is None
    assert bounds.block_for(bounds.test_end) is None


def test_validate_rejects_touching_blocks() -> None:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    bounds = SplitBounds(
        train_start=now,
        train_end=now + timedelta(days=10),
        calibrate_start=now + timedelta(days=10),
        calibrate_end=now + timedelta(days=20),
        test_start=now + timedelta(days=20),
        test_end=now + timedelta(days=30),
        gap_hours=6,
    )
    with pytest.raises(ValueError, match="gap between train and calibrate"):
        bounds.validate()


def test_validate_rejects_out_of_order_blocks() -> None:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    bounds = SplitBounds(
        train_start=now + timedelta(days=30),
        train_end=now + timedelta(days=40),
        calibrate_start=now,
        calibrate_end=now + timedelta(days=10),
        test_start=now + timedelta(days=10),
        test_end=now + timedelta(days=20),
        gap_hours=6,
    )
    with pytest.raises(ValueError, match="overlaps or follows"):
        bounds.validate()


def test_short_span_raises_rather_than_producing_empty_blocks(config: Config) -> None:
    """Too little data must fail loudly, not yield a zero-row training block."""
    with pytest.raises(ValueError):
        compute_bounds(FIRST, FIRST + timedelta(days=20), config)


# --- metrics -----------------------------------------------------------------


def test_mae_basic() -> None:
    assert mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert mae(np.array([0.0, 0.0]), np.array([1.0, 3.0])) == 2.0


def test_wape_is_normalised_mae() -> None:
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    assert wape(y, p) == pytest.approx(np.sum(np.abs(y - p)) / np.sum(y))


def test_wape_undefined_on_all_zero_actuals() -> None:
    """The reason percentage metrics are not reported for the low tier."""
    assert np.isnan(wape(np.zeros(5), np.ones(5)))


def test_poisson_deviance_zero_for_perfect_prediction() -> None:
    y = np.array([0.0, 1.0, 5.0, 20.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)


def test_poisson_deviance_handles_zero_actuals() -> None:
    """46% of grid cells are zero, so this must not produce NaN."""
    value = poisson_deviance(np.zeros(4), np.array([1.0, 2.0, 0.5, 3.0]))
    assert np.isfinite(value)
    assert value > 0


def test_poisson_deviance_survives_zero_prediction() -> None:
    """A dead zone invites a zero prediction; log(0) must not appear."""
    assert np.isfinite(poisson_deviance(np.array([1.0, 0.0]), np.array([0.0, 0.0])))


def test_poisson_deviance_penalises_underprediction_of_counts() -> None:
    y = np.array([10.0, 10.0])
    assert poisson_deviance(y, np.array([5.0, 5.0])) > poisson_deviance(y, np.array([9.0, 9.0]))


def test_zero_hit_rate() -> None:
    y = np.array([0.0, 0.0, 0.0, 5.0])
    assert zero_hit_rate(y, np.array([0.1, 0.2, 4.0, 5.0])) == pytest.approx(2 / 3)


def test_zero_hit_rate_undefined_without_zeros() -> None:
    assert np.isnan(zero_hit_rate(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_evaluate_returns_all_metrics() -> None:
    result = evaluate(np.array([1.0, 2.0, 0.0]), np.array([1.0, 2.0, 0.5]))
    assert result.n == 3
    assert result.tier == "all"
    assert np.isfinite(result.poisson_deviance)


def test_evaluate_by_tier_splits_correctly() -> None:
    y = np.array([100.0, 100.0, 5.0, 5.0, 0.0, 0.0])
    p = np.array([90.0, 110.0, 4.0, 6.0, 0.0, 1.0])
    tiers = np.array(["high", "high", "medium", "medium", "low", "low"])

    results = evaluate_by_tier(y, p, tiers)
    by_name = {r.tier: r for r in results}

    assert by_name["all"].n == 6
    assert by_name["high"].n == 2
    assert by_name["medium"].n == 2
    assert by_name["low"].n == 2


def test_evaluate_by_tier_omits_absent_tiers() -> None:
    y = np.array([100.0, 100.0])
    tiers = np.array(["high", "high"])
    results = evaluate_by_tier(y, y, tiers)
    assert {r.tier for r in results} == {"all", "high"}


def test_relative_improvement_positive_when_better() -> None:
    """Lower is better for every metric, so improvement is a reduction."""
    assert relative_improvement(champion=9.0, benchmark=10.0) == pytest.approx(0.1)


def test_relative_improvement_negative_when_worse() -> None:
    assert relative_improvement(champion=11.0, benchmark=10.0) == pytest.approx(-0.1)


def test_tier_metrics_serialise_nan_as_null() -> None:
    """WAPE is undefined in dead zones; JSON must not carry NaN."""
    result = evaluate(np.zeros(3), np.ones(3), tier="low")
    assert result.to_dict()["wape"] is None


# --- baseline specification --------------------------------------------------


def test_demand_features_identified() -> None:
    """Count-valued columns need the log transform; calendar columns must not."""
    from urbanflow.models.baselines import is_demand_feature

    for column in ("recent_1", "lag_24h", "roll_mean_168h", "how_median_4w"):
        assert is_demand_feature(column), column
    for column in ("local_hour", "hour_sin", "is_weekend", "zero_rate_168h"):
        assert not is_demand_feature(column), column


def test_design_matrix_log_transforms_only_demand_columns() -> None:
    from urbanflow.models.baselines import design_matrix

    features = {
        "recent_1": np.array([0.0, 99.0]),
        "local_hour": np.array([0.0, 23.0]),
    }
    matrix = design_matrix(features, ["recent_1", "local_hour"])
    assert matrix[:, 0] == pytest.approx([0.0, np.log(100.0)])
    assert matrix[:, 1] == pytest.approx([0.0, 23.0])


def test_design_matrix_handles_zero_demand() -> None:
    """46% of grid cells are zero; log1p(0) must be 0, not -inf."""
    from urbanflow.models.baselines import design_matrix

    matrix = design_matrix({"recent_1": np.zeros(3)}, ["recent_1"])
    assert np.all(np.isfinite(matrix))
    assert matrix.sum() == 0.0


def test_poisson_glm_spans_wide_demand_range() -> None:
    """One global model must cope with zones from 0.1 to 300 pickups an hour.

    Raw counts under a log link make demand enter the exponent multiplicatively,
    which overpredicts small zones badly; the log transform is what makes a
    single pooled model viable.
    """
    from urbanflow.models.baselines import fit_poisson

    rng = np.random.default_rng(0)
    levels = np.concatenate([np.full(2000, 200.0), np.full(2000, 1.0)])
    recent = rng.poisson(levels).astype(float)
    target = rng.poisson(levels).astype(float)

    model = fit_poisson({"recent_1": recent}, target, ["recent_1"])
    predictions = model.predict({"recent_1": recent})

    busy, quiet = predictions[:2000].mean(), predictions[2000:].mean()
    assert 150 < busy < 260, f"busy zone prediction off: {busy}"
    assert 0.4 < quiet < 2.5, f"quiet zone prediction off: {quiet}"


def test_poisson_baseline_rejects_use_before_fitting() -> None:
    from urbanflow.models.baselines import PoissonBaseline

    with pytest.raises(RuntimeError, match="before fitting"):
        PoissonBaseline(name="c", description="").predict({})


def test_volume_bias_detects_systematic_underprediction() -> None:
    """The failure MAE cannot see: consistently forecasting too few pickups."""
    from urbanflow.models.metrics import volume_bias

    y = np.array([10.0, 20.0, 30.0])
    assert volume_bias(y, y) == pytest.approx(1.0)
    assert volume_bias(y, y * 0.8) == pytest.approx(0.8)
    assert volume_bias(y, y * 1.25) == pytest.approx(1.25)


def test_volume_bias_undefined_without_demand() -> None:
    from urbanflow.models.metrics import volume_bias

    assert np.isnan(volume_bias(np.zeros(3), np.ones(3)))


def test_median_estimator_underpredicts_skewed_counts() -> None:
    """Why MAE and deviance disagree on this data.

    MAE is minimised by the conditional median, deviance by the conditional
    mean. For right-skewed counts the median sits below the mean, so a
    median-like predictor posts a strong MAE while under-forecasting volume.
    """
    from urbanflow.models.metrics import mae, volume_bias

    # Rate chosen to match the observed low tier, which averages ~1.56
    # pickups an hour. At higher rates a Poisson median and mean coincide on
    # the integers, so the divergence is specifically a sparse-demand effect —
    # which is where 62% of the grid lives.
    rng = np.random.default_rng(0)
    y = rng.poisson(1.5, size=50000).astype(float)

    median_pred = np.full_like(y, np.median(y))
    mean_pred = np.full_like(y, np.mean(y))

    assert mae(y, median_pred) <= mae(y, mean_pred)
    assert volume_bias(y, median_pred) < 0.75, "median should under-forecast volume"
    assert volume_bias(y, mean_pred) == pytest.approx(1.0, abs=0.01)
