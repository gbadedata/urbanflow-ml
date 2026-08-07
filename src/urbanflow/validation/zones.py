"""Forecastable-zone determination and demand tiering.

Which zones a model should predict is an empirical question, not a fixed list.
Two zones sat in the raw data with one and four pickups respectively across a
full year; carrying them into an hourly model would inject pure noise into every
macro-averaged metric. This module derives the set from the data and records the
evidence, so the decision can be re-checked when the window moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from urbanflow.config import Config

TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_EXCLUDED = "excluded"


@dataclass(frozen=True)
class ZoneProfile:
    """Observed volume for one zone over the profiling window."""

    zone_id: int
    trips: int
    hours_in_window: int

    @property
    def mean_per_hour(self) -> float:
        return self.trips / self.hours_in_window if self.hours_in_window else 0.0

    def tier(self, config: Config) -> str:
        if self.mean_per_hour < config.zones.min_pickups_per_hour:
            return TIER_EXCLUDED
        if self.mean_per_hour >= config.demand_tiers.high_min:
            return TIER_HIGH
        if self.mean_per_hour >= config.demand_tiers.medium_min:
            return TIER_MEDIUM
        return TIER_LOW


def profile_zones(
    con: duckdb.DuckDBPyConnection, config: Config, validated_glob: str | None = None
) -> list[ZoneProfile]:
    """Count trips per zone and the number of hours spanned.

    Profiling runs on validated data so that rows already rejected for other
    reasons cannot inflate a zone above the volume floor.
    """
    pattern = validated_glob or str(config.validated_path / "trips_*.parquet")
    rows = con.execute(f"""
        SELECT zone_id, COUNT(*) AS trips
        FROM read_parquet('{pattern}')
        GROUP BY zone_id
    """).fetchall()

    span_row = con.execute(f"""
        SELECT date_diff('hour', MIN(pickup_utc), MAX(pickup_utc)) + 1
        FROM read_parquet('{pattern}')
    """).fetchone()
    hours = int(span_row[0]) if span_row and span_row[0] else 0

    return [ZoneProfile(zone_id=int(r[0]), trips=int(r[1]), hours_in_window=hours) for r in rows]


def forecastable_zones(profiles: list[ZoneProfile], config: Config) -> list[int]:
    """Zone IDs that pass both structural and volume screens."""
    excluded = set(config.zones.excluded_location_ids)
    return sorted(
        p.zone_id for p in profiles if p.zone_id not in excluded and p.tier(config) != TIER_EXCLUDED
    )


def write_zone_registry(profiles: list[ZoneProfile], config: Config, path: Path) -> Path:
    """Persist the zone decision with the evidence behind it.

    The registry is committed. It is what lets a reader see why 43 zones were
    dropped without re-running the profiling.
    """
    structural = set(config.zones.excluded_location_ids)
    total_trips = sum(p.trips for p in profiles)

    entries: list[dict[str, object]] = []
    for p in sorted(profiles, key=lambda x: -x.trips):
        tier = TIER_EXCLUDED if p.zone_id in structural else p.tier(config)
        entries.append(
            {
                "zone_id": p.zone_id,
                "trips": p.trips,
                "mean_per_hour": round(p.mean_per_hour, 4),
                "tier": tier,
                "excluded_reason": (
                    "structural"
                    if p.zone_id in structural
                    else ("below_volume_floor" if tier == TIER_EXCLUDED else None)
                ),
            }
        )

    kept = [e for e in entries if e["tier"] != TIER_EXCLUDED]
    kept_ids = {e["zone_id"] for e in kept}
    kept_trips = sum(p.trips for p in profiles if p.zone_id in kept_ids)
    payload = {
        "min_pickups_per_hour": config.zones.min_pickups_per_hour,
        "tier_metric": config.demand_tiers.metric,
        "tier_high_min": config.demand_tiers.high_min,
        "tier_medium_min": config.demand_tiers.medium_min,
        "zones_profiled": len(entries),
        "zones_forecastable": len(kept),
        "trips_retained_share": (round(kept_trips / total_trips, 6) if total_trips else 0.0),
        "tier_counts": {
            tier: sum(1 for e in entries if e["tier"] == tier)
            for tier in (TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_EXCLUDED)
        },
        "zones": entries,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
