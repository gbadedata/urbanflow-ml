"""Tests for project configuration.

The config file records the decisions the whole build depends on, so the
invariants that matter are tested here rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from urbanflow.config import DEFAULT_CONFIG_PATH, Config, load_config


def test_default_config_loads() -> None:
    config = load_config()
    assert config.data.service == "yellow"
    assert config.forecast.horizons_hours


def test_default_config_starts_after_covid_break() -> None:
    """Spec §5.1: the window must not span the 2020-21 structural break."""
    config = load_config()
    year = int(config.data.start_month.split("-")[0])
    assert year >= 2022, "window starts inside or before the COVID regime break"


def test_gap_covers_longest_horizon() -> None:
    """Spec §7.1: without this, a validation target can fall inside training."""
    config = load_config()
    assert config.split.gap_hours >= max(config.forecast.horizons_hours)


def test_feature_cutoff_is_positive() -> None:
    """Spec §2.1: a zero cutoff means assuming instant data availability."""
    config = load_config()
    assert config.features.feature_cutoff_minutes > 0


def test_unknown_zones_are_excluded() -> None:
    """LocationIDs 264 and 265 are TLC placeholders, not real zones."""
    config = load_config()
    assert 264 in config.zones.excluded_location_ids
    assert 265 in config.zones.excluded_location_ids


def test_demand_tiers_are_ordered() -> None:
    config = load_config()
    assert config.demand_tiers.high_min > config.demand_tiers.medium_min


def test_demand_tiers_use_mean_not_median() -> None:
    """Measured correction to spec §7.2.

    152 of 259 zones average under 1 pickup/hour and their median hour is 0, so a
    median-based rule puts nearly every zone in the low tier.
    """
    config = load_config()
    assert config.demand_tiers.metric == "mean"


def test_demand_tiers_reject_median_metric(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["demand_tiers"]["metric"] = "median"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="mean"):
        load_config(path)


def test_volume_floor_is_set() -> None:
    """43 zones fall below 0.1 pickups/hour and carry 0.02% of trips."""
    config = load_config()
    assert config.zones.min_pickups_per_hour > 0


def test_excluded_zones_within_valid_range() -> None:
    config = load_config()
    for zone in config.zones.excluded_location_ids:
        assert config.zones.min_location_id <= zone <= config.zones.max_location_id


def test_rejects_excluded_zone_outside_range(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["zones"]["excluded_location_ids"] = [1, 9999]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="outside the valid"):
        load_config(path)


def test_rejects_bad_month_format(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["data"]["start_month"] = "2023/01"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError, match="YYYY-MM"):
        load_config(path)


def test_rejects_impossible_month(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["data"]["start_month"] = "2023-13"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValidationError):
        load_config(path)


def test_rejects_gap_shorter_than_horizon(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["split"]["gap_hours"] = 1
    raw["forecast"]["horizons_hours"] = [1, 3, 6]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="gap_hours"):
        load_config(path)


def test_paths_resolve_to_absolute() -> None:
    config = load_config()
    assert config.raw_path.is_absolute()
    assert config.manifest_path.is_absolute()


def test_config_model_rejects_negative_cutoff() -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["features"]["feature_cutoff_minutes"] = -5
    with pytest.raises(ValidationError):
        Config.model_validate(raw)
