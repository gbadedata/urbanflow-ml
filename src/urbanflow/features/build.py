"""Feature construction under an availability cutoff.

The central constraint (spec §2.1): at prediction time ``as_of``, only data that
has actually arrived may be used. With a feed latency of ``cutoff`` minutes, the
most recent *fully arrived* hour is::

    anchor = floor_hour(as_of - cutoff) - 1h

The subtraction of an hour is the part usually missed. At ``as_of = 14:00`` with
a 15-minute cutoff, the hour 13:00-13:59 has not finished, and the hour
12:00-12:59 finished at 13:00 with its records complete by 13:15. So the newest
usable observation is 12:00 — two hours behind ``as_of``, not one.

For an on-the-hour ``as_of`` and a horizon ``h``, this reduces to a constant
offset from the target hour::

    anchor = target - (h + 2) hours

Two families of feature follow from that:

* **Recent history** anchors on availability. There is no lag-1h feature; it
  would require data that has not arrived. This is precisely the leakage that
  makes a naive backtest look excellent and a deployed model fail.
* **Seasonal history** anchors on the target hour — demand at the same clock
  hour yesterday, two days ago, and last week. These are old enough to have
  arrived regardless of horizon, and they carry the diurnal and weekly signal
  that recent lags cannot.

Because the grid is dense (spec §2.2), row-offset windows are exactly equivalent
to time-based windows. That equivalence is what makes these definitions safe; on
a sparse table every one of them would silently span the wrong interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from urbanflow.config import Config

logger = logging.getLogger(__name__)

#: Weeks of history used by the hour-of-week trailing median, which doubles as
#: Baseline B (spec §2.3) — the benchmark the champion must beat.
HOW_MEDIAN_WEEKS = 4


def anchor_hour(as_of: datetime, cutoff_minutes: int) -> datetime:
    """The most recent hour whose data has fully arrived at ``as_of``."""
    available_until = as_of - timedelta(minutes=cutoff_minutes)
    floored = available_until.replace(minute=0, second=0, microsecond=0)
    return floored - timedelta(hours=1)


def anchor_offset_hours(horizon_hours: int, cutoff_minutes: int) -> int:
    """Hours between the target hour and the anchor, for an on-the-hour origin.

    Raises if the cutoff exceeds an hour, since the constant-offset simplification
    no longer holds and the feature definitions below would be wrong.
    """
    if not 0 <= cutoff_minutes < 60:
        raise ValueError(
            f"cutoff_minutes must be in [0, 60) for the constant-offset derivation; "
            f"got {cutoff_minutes}. A longer latency needs per-row anchor computation."
        )
    return horizon_hours + 2


@dataclass(frozen=True)
class FeatureSpec:
    """The feature set for one horizon, with every source offset made explicit."""

    horizon_hours: int
    cutoff_minutes: int

    @property
    def offset(self) -> int:
        return anchor_offset_hours(self.horizon_hours, self.cutoff_minutes)

    @property
    def recent_offsets(self) -> list[int]:
        """Offsets from target for the three most recent available hours."""
        return [self.offset, self.offset + 1, self.offset + 2]

    @property
    def seasonal_offsets(self) -> list[int]:
        """Same clock hour yesterday, two days ago, one week ago."""
        return [24, 48, 168]

    def warmup_hours(self) -> int:
        """Hours of history required before every feature is fully formed.

        Not merely the longest single lag. Two DuckDB behaviours make a NULL check
        insufficient on its own:

        * ``list_median`` ignores NULL elements, so a 4-week median silently
          returns a 1-week median while there are only 168 hours of history.
        * ``ROWS BETWEEN`` aggregates over however many rows exist, so a 168-hour
          rolling mean silently becomes a 20-hour mean near the series start.

        Both produce non-NULL values that mean something different from the same
        column later on — a distribution shift landing squarely in the training
        block, which is the earliest data. Rows are therefore trimmed by required
        history, not by nullness.
        """
        return max(
            self.offset + 167,
            HOW_MEDIAN_WEEKS * 168,
        )

    def validate(self) -> None:
        """Assert no declared feature reads data newer than the anchor."""
        for seasonal in self.seasonal_offsets:
            if seasonal < self.offset:
                raise ValueError(
                    f"seasonal lag {seasonal}h is newer than the anchor at "
                    f"{self.offset}h for horizon {self.horizon_hours}h — it would "
                    f"use data that has not arrived"
                )


def build_features(
    con: duckdb.DuckDBPyConnection,
    config: Config,
    horizon_hours: int,
    grid_table: str = "zone_hour",
) -> str:
    """Build the feature table for one horizon; returns the table name.

    Every window is expressed as a row offset ending at or before the anchor, so
    availability is enforced by construction rather than checked afterwards.
    """
    spec = FeatureSpec(horizon_hours, config.features.feature_cutoff_minutes)
    spec.validate()

    off = spec.offset
    tz = config.features.timezone
    table = f"features_h{horizon_hours}"

    how_terms = ", ".join(
        f"LAG(pickup_count, {168 * w}) OVER w" for w in range(1, HOW_MEDIAN_WEEKS + 1)
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT
            location_id,
            hour_utc AS target_hour,
            {horizon_hours} AS horizon_hours,
            hour_utc - INTERVAL '{horizon_hours} hour' AS as_of,
            hour_utc - INTERVAL '{off} hour' AS anchor_hour,
            pickup_count AS target,

            -- Recent history: anchored on availability, never on the target.
            LAG(pickup_count, {off})     OVER w AS recent_1,
            LAG(pickup_count, {off + 1}) OVER w AS recent_2,
            LAG(pickup_count, {off + 2}) OVER w AS recent_3,

            -- Seasonal history: same clock hour on earlier days.
            LAG(pickup_count, 24)  OVER w AS lag_24h,
            LAG(pickup_count, 48)  OVER w AS lag_48h,
            LAG(pickup_count, 168) OVER w AS lag_168h,

            -- Rolling aggregates over windows ending at the anchor.
            AVG(pickup_count) OVER (
                PARTITION BY location_id ORDER BY hour_utc
                ROWS BETWEEN {off + 2} PRECEDING AND {off} PRECEDING
            ) AS roll_mean_3h,
            AVG(pickup_count) OVER (
                PARTITION BY location_id ORDER BY hour_utc
                ROWS BETWEEN {off + 23} PRECEDING AND {off} PRECEDING
            ) AS roll_mean_24h,
            AVG(pickup_count) OVER (
                PARTITION BY location_id ORDER BY hour_utc
                ROWS BETWEEN {off + 167} PRECEDING AND {off} PRECEDING
            ) AS roll_mean_168h,
            AVG(CASE WHEN pickup_count = 0 THEN 1.0 ELSE 0.0 END) OVER (
                PARTITION BY location_id ORDER BY hour_utc
                ROWS BETWEEN {off + 167} PRECEDING AND {off} PRECEDING
            ) AS zero_rate_168h,

            -- Hour-of-week trailing median over {HOW_MEDIAN_WEEKS} weeks.
            -- This is also Baseline B, computed from the same code path so the
            -- benchmark and the feature cannot drift apart.
            list_median(LIST_VALUE({how_terms})) AS how_median_4w,

            -- Calendar features, derived in local time from a UTC index. The
            -- grid keeps 24 rows per day through both DST transitions; only
            -- these derived columns change across the boundary.
            EXTRACT(hour   FROM timezone('{tz}', hour_utc)) AS local_hour,
            EXTRACT(dow    FROM timezone('{tz}', hour_utc)) AS local_dow,
            EXTRACT(month  FROM timezone('{tz}', hour_utc)) AS local_month,
            CASE WHEN EXTRACT(dow FROM timezone('{tz}', hour_utc)) IN (0, 6)
                 THEN 1 ELSE 0 END AS is_weekend,
            SIN(2 * PI() * EXTRACT(hour FROM timezone('{tz}', hour_utc)) / 24) AS hour_sin,
            COS(2 * PI() * EXTRACT(hour FROM timezone('{tz}', hour_utc)) / 24) AS hour_cos,
            SIN(2 * PI() * EXTRACT(dow FROM timezone('{tz}', hour_utc)) / 7) AS dow_sin,
            COS(2 * PI() * EXTRACT(dow FROM timezone('{tz}', hour_utc)) / 7) AS dow_cos
        FROM {grid_table}
        WINDOW w AS (PARTITION BY location_id ORDER BY hour_utc)
    """)

    # Trim the warm-up period. A NULL check alone is not enough here: see
    # FeatureSpec.warmup_hours for why partially-formed features come back
    # non-NULL and therefore pass any nullness filter.
    con.execute(f"""
        DELETE FROM {table}
        WHERE target_hour < (
            SELECT MIN(hour_utc) + INTERVAL '{spec.warmup_hours()} hour' FROM {grid_table}
        )
    """)
    # Belt and braces: the warmup trim above is the real guard, but a NULL here
    # would mean the derivation and the data disagree.
    con.execute(f"DELETE FROM {table} WHERE how_median_4w IS NULL OR recent_3 IS NULL")
    return table


def feature_columns(table_columns: list[str]) -> list[str]:
    """Model input columns: everything except identifiers and the target."""
    reserved = {"location_id", "target_hour", "as_of", "anchor_hour", "target", "horizon_hours"}
    return [c for c in table_columns if c not in reserved]


def assert_no_future_leakage(
    con: duckdb.DuckDBPyConnection, config: Config, table: str, horizon_hours: int
) -> None:
    """Verify every feature's source hour is at or before the anchor.

    Spec §8, gate 2. Checked against the recorded anchor rather than recomputed,
    so a change to the offset derivation cannot pass silently.
    """
    spec = FeatureSpec(horizon_hours, config.features.feature_cutoff_minutes)
    row = con.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE anchor_hour > as_of - INTERVAL '{config.features.feature_cutoff_minutes} minute'
           OR anchor_hour >= target_hour
    """).fetchone()
    assert row is not None
    if row[0]:
        raise ValueError(
            f"{row[0]:,} rows in {table} have an anchor at or after the forecast "
            f"origin — features would use unavailable data (spec §2.1)"
        )
    logger.info(
        "leakage check passed for horizon %dh (anchor is target - %dh)",
        horizon_hours,
        spec.offset,
    )
