"""Tests for the champion and the promotion gate.

The gate is the mechanism that stops a model shipping on a flattering number, so
it needs to fail for the right reasons. Each criterion gets a test that isolates
it, and the two-way behaviour — keeping the baseline is a pass — is asserted
explicitly rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from urbanflow.config import Config, load_config
from urbanflow.models.champion import (
    CHAMPION,
    evaluate_gate,
    fit_champion,
)
from urbanflow.models.metrics import TierMetrics


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


def _metrics(
    tier: str,
    deviance: float,
    bias: float = 1.0,
    mae: float = 1.0,
    n: int = 1000,
) -> TierMetrics:
    return TierMetrics(
        tier=tier,
        n=n,
        mae=mae,
        rmse=mae,
        poisson_deviance=deviance,
        wape=0.2,
        zero_hit_rate=0.5,
        mean_actual=5.0,
        mean_predicted=5.0 * bias,
        volume_bias=bias,
    )


def _benchmark() -> list[TierMetrics]:
    return [
        _metrics("all", 5.0),
        _metrics("high", 7.0),
        _metrics("medium", 3.4),
        _metrics("low", 5.0),
    ]


# --- gate logic --------------------------------------------------------------


def test_gate_promotes_on_clear_improvement(config: Config) -> None:
    champion = [
        _metrics("all", 2.5),
        _metrics("high", 5.0),
        _metrics("medium", 2.4),
        _metrics("low", 1.4),
    ]
    result = evaluate_gate(config, 1, champion, _benchmark())
    assert result.passed
    assert result.ships == CHAMPION


def test_gate_keeps_baseline_on_marginal_improvement(config: Config) -> None:
    """5% is an improvement and still not enough; the threshold is 10%."""
    champion = [
        _metrics("all", 4.75),
        _metrics("high", 6.65),
        _metrics("medium", 3.23),
        _metrics("low", 4.75),
    ]
    result = evaluate_gate(config, 1, champion, _benchmark())
    assert not result.passed
    assert result.ships == "hour_of_week_median"
    assert any("below the 10% threshold" in r for r in result.reasons)


def test_keeping_the_baseline_is_a_documented_outcome(config: Config) -> None:
    """Spec §8: shipping the simpler system with evidence is a pass."""
    champion = [_metrics("all", 5.0), _metrics("high", 7.0), _metrics("medium", 3.4)]
    result = evaluate_gate(config, 1, champion, _benchmark())
    assert result.ships == "hour_of_week_median"
    assert result.reasons, "a refusal must record why"


def test_gate_fails_on_volume_bias_even_when_deviance_wins(config: Config) -> None:
    """A model may not buy deviance with systematic over-forecasting."""
    champion = [
        _metrics("all", 1.0),
        _metrics("high", 1.0),
        _metrics("medium", 1.0, bias=1.30),
        _metrics("low", 1.0),
    ]
    result = evaluate_gate(config, 1, champion, _benchmark())
    assert not result.passed
    assert result.bias_violations
    assert any("volume bias" in r for r in result.reasons)


def test_gate_fails_on_under_forecasting_too(config: Config) -> None:
    """The bias constraint is two-sided; under-supply is the worse failure."""
    champion = [
        _metrics("all", 1.0),
        _metrics("high", 1.0),
        _metrics("medium", 1.0),
        _metrics("low", 1.0, bias=0.85),
    ]
    result = evaluate_gate(config, 1, champion, _benchmark())
    assert not result.passed
    assert "low (bias 0.850)" in result.bias_violations


def test_ungated_tier_degradation_blocks_promotion(config: Config) -> None:
    """The low tier is not gated for improvement but may not collapse."""
    champion = [
        _metrics("all", 2.0),
        _metrics("high", 5.0),
        _metrics("medium", 2.4),
        _metrics("low", 9.0),
    ]
    result = evaluate_gate(config, 1, champion, _benchmark())
    assert not result.passed
    assert result.degradations


def test_small_tier_regression_is_tolerated(config: Config) -> None:
    """A 5% dip in an ungated tier is within tolerance."""
    champion = [
        _metrics("all", 2.0),
        _metrics("high", 5.0),
        _metrics("medium", 2.4),
        _metrics("low", 5.25),
    ]
    result = evaluate_gate(config, 1, champion, _benchmark())
    assert result.passed


def test_gate_uses_configured_metric(config: Config) -> None:
    assert config.gates.metric == "poisson_deviance"
    result = evaluate_gate(config, 1, _benchmark(), _benchmark())
    assert result.metric == "poisson_deviance"


def test_identical_models_do_not_promote(config: Config) -> None:
    result = evaluate_gate(config, 1, _benchmark(), _benchmark())
    assert not result.passed
    assert result.improvements["high"] == pytest.approx(0.0)


def test_gate_result_serialises(config: Config) -> None:
    result = evaluate_gate(config, 1, _benchmark(), _benchmark())
    payload = result.to_dict()
    assert payload["metric"] == "poisson_deviance"
    assert isinstance(payload["reasons"], list)


# --- champion fitting --------------------------------------------------------


def test_champion_learns_a_daily_pattern() -> None:
    """A model that cannot recover a strong diurnal signal is misconfigured."""
    rng = np.random.default_rng(0)
    n = 6000
    hour = rng.integers(0, 24, n).astype(float)
    level = 20 + 15 * np.sin(2 * np.pi * hour / 24)
    target = rng.poisson(np.clip(level, 0.1, None)).astype(float)
    features = {"local_hour": hour, "recent_1": target}

    model = fit_champion(features, target, ["local_hour", "recent_1"], max_iter=60)
    predictions = model.predict(features)

    peak = predictions[hour == 6].mean()
    trough = predictions[hour == 18].mean()
    assert peak > trough, "diurnal shape not recovered"


def test_champion_predictions_are_non_negative() -> None:
    """Demand cannot be negative; a forecast of -3 cabs is not actionable."""
    rng = np.random.default_rng(1)
    n = 2000
    recent = rng.poisson(1.0, n).astype(float)
    target = rng.poisson(1.0, n).astype(float)
    model = fit_champion({"recent_1": recent}, target, ["recent_1"], max_iter=40)
    assert np.all(model.predict({"recent_1": recent}) >= 0.0)


def test_champion_is_deterministic() -> None:
    """Same seed, same data, same predictions — required for reproducibility."""
    rng = np.random.default_rng(2)
    recent = rng.poisson(3.0, 2000).astype(float)
    target = rng.poisson(3.0, 2000).astype(float)

    first = fit_champion({"recent_1": recent}, target, ["recent_1"], max_iter=40)
    second = fit_champion({"recent_1": recent}, target, ["recent_1"], max_iter=40)
    assert np.allclose(first.predict({"recent_1": recent}), second.predict({"recent_1": recent}))


def test_champion_rejects_use_before_fitting() -> None:
    from urbanflow.models.champion import ChampionModel

    with pytest.raises(RuntimeError, match="before fitting"):
        ChampionModel(name="x", description="").predict({})
