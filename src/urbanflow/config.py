"""Typed project configuration.

Every module reads its settings from here rather than hard-coding constants, so
that a single YAML file is the record of what a given experiment assumed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class DataConfig(BaseModel):
    start_month: str
    end_month: str
    service: str = "yellow"
    raw_dir: Path
    manifest_dir: Path
    validated_dir: Path
    quarantine_dir: Path

    @field_validator("start_month", "end_month")
    @classmethod
    def _check_month_format(cls, v: str) -> str:
        parts = v.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            raise ValueError(f"month must be formatted YYYY-MM, got {v!r}")
        year, month = int(parts[0]), int(parts[1])
        if not 2009 <= year <= 2100:
            raise ValueError(f"year out of plausible range: {year}")
        if not 1 <= month <= 12:
            raise ValueError(f"month out of range: {month}")
        return v


class ZonesConfig(BaseModel):
    excluded_location_ids: list[int]
    min_location_id: int = Field(ge=1)
    max_location_id: int = Field(ge=1)
    min_pickups_per_hour: float = Field(ge=0)
    lookup_url: str


class FeaturesConfig(BaseModel):
    feature_cutoff_minutes: int = Field(ge=0)
    lags_hours: list[int]
    rolling_windows_hours: list[int]
    timezone: str


class ForecastConfig(BaseModel):
    horizons_hours: list[int]


class SplitConfig(BaseModel):
    holdout_weeks: int = Field(gt=0)
    gap_hours: int = Field(ge=0)


class GatesConfig(BaseModel):
    """Promotion criteria, fixed before any champion was trained."""

    metric: str
    min_relative_improvement: float = Field(gt=0)
    gated_tiers: list[str]
    max_volume_bias_deviation: float = Field(gt=0)
    max_tier_degradation: float = Field(gt=0)

    @field_validator("metric")
    @classmethod
    def _check_metric(cls, v: str) -> str:
        allowed = {"poisson_deviance", "mae", "rmse"}
        if v not in allowed:
            raise ValueError(f"gate metric must be one of {sorted(allowed)}, got {v!r}")
        return v


class ValidationConfig(BaseModel):
    max_trip_duration_hours: int = Field(gt=0)
    dst_ambiguous_policy: str = "earliest"


class DemandTiersConfig(BaseModel):
    """Tiers are defined on MEAN hourly pickups, not median.

    Measured against 2024 data, 152 of 259 zones average under 1 pickup/hour and
    their median hour is exactly 0, so a median-based rule collapses almost every
    zone into the low tier and erases the distinction.
    """

    metric: str = "mean"
    high_min: float
    medium_min: float

    @field_validator("metric")
    @classmethod
    def _check_metric(cls, v: str) -> str:
        if v != "mean":
            raise ValueError(
                "demand tiers must be defined on mean hourly pickups; median "
                "classifies nearly every zone as low (see configs/config.yaml)"
            )
        return v


class Config(BaseModel):
    data: DataConfig
    zones: ZonesConfig
    features: FeaturesConfig
    forecast: ForecastConfig
    split: SplitConfig
    demand_tiers: DemandTiersConfig
    validation: ValidationConfig
    gates: GatesConfig

    @property
    def raw_path(self) -> Path:
        return _resolve(self.data.raw_dir)

    @property
    def manifest_path(self) -> Path:
        return _resolve(self.data.manifest_dir)

    @property
    def validated_path(self) -> Path:
        return _resolve(self.data.validated_dir)

    @property
    def quarantine_path(self) -> Path:
        return _resolve(self.data.quarantine_dir)

    @property
    def validation_max_duration_hours(self) -> int:
        return self.validation.max_trip_duration_hours

    def check_zone_range(self) -> None:
        """Excluded zones must be inside the declared valid range."""
        out_of_range = [
            z
            for z in self.zones.excluded_location_ids
            if not self.zones.min_location_id <= z <= self.zones.max_location_id
        ]
        if out_of_range:
            raise ValueError(
                f"excluded zone ids fall outside the valid LocationID range: {out_of_range}"
            )

    def check_gap_covers_horizon(self) -> None:
        """Spec §7.1: the train/validation gap must be at least the longest horizon.

        Without this, a validation row's target can fall inside the training window.
        """
        max_horizon = max(self.forecast.horizons_hours)
        if self.split.gap_hours < max_horizon:
            raise ValueError(
                f"split.gap_hours ({self.split.gap_hours}) must be >= the longest "
                f"forecast horizon ({max_horizon})"
            )


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path | None = None) -> Config:
    """Load and validate the project configuration."""
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)
    config = Config.model_validate(raw)
    config.check_gap_covers_horizon()
    config.check_zone_range()
    return config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached accessor for the default configuration."""
    return load_config()
