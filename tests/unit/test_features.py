"""Tests for the feature builder.

The availability cutoff is the spec's headline correction, so the tests that
matter most are the ones asserting no feature reads data that has not arrived.
A backtest that violates this looks excellent and cannot be deployed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from urbanflow.config import Config, load_config
from urbanflow.duck import connect
from urbanflow.features.build import (
    FeatureSpec,
    anchor_hour,
    anchor_offset_hours,
    assert_no_future_leakage,
    build_features,
    feature_columns,
)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    import yaml

    from urbanflow.config import DEFAULT_CONFIG_PATH

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["data"]["raw_dir"] = str(tmp_path)
    raw["data"]["validated_dir"] = str(tmp_path / "validated")
    raw["data"]["quarantine_dir"] = str(tmp_path / "quarantine")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    return connect()


def _seed_grid(con: duckdb.DuckDBPyConnection, hours: int = 1200, zones: int = 2) -> None:
    """A dense grid with a known, reproducible demand pattern."""
    zone_list = ", ".join(str(160 + z) for z in range(zones))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE zone_hour AS
        WITH h AS (
            SELECT UNNEST(generate_series(
                TIMESTAMPTZ '2024-01-01 00:00:00+00',
                TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL '{hours - 1} hour',
                INTERVAL '1 hour')) AS hour_utc
        ), z AS (SELECT UNNEST([{zone_list}]) AS location_id)
        SELECT z.location_id, h.hour_utc,
               CAST(10 + (EXTRACT(hour FROM h.hour_utc) * 2) AS INTEGER) AS pickup_count
        FROM z CROSS JOIN h
    """)


# --- anchor arithmetic -------------------------------------------------------


def test_anchor_is_two_hours_back_on_the_hour() -> None:
    """At 14:00 with 15 minutes of latency, 12:00 is the newest complete hour.

    13:00-13:59 has not finished; 12:00-12:59 finished at 13:00 with records
    complete by 13:15.
    """
    as_of = datetime(2024, 6, 1, 14, 0, tzinfo=UTC)
    assert anchor_hour(as_of, 15) == datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def test_anchor_mid_hour_origin() -> None:
    as_of = datetime(2024, 6, 1, 14, 30, tzinfo=UTC)
    assert anchor_hour(as_of, 15) == datetime(2024, 6, 1, 13, 0, tzinfo=UTC)


def test_anchor_with_zero_latency_still_excludes_incomplete_hour() -> None:
    """Even with instant delivery, the current hour is not yet complete."""
    as_of = datetime(2024, 6, 1, 14, 0, tzinfo=UTC)
    assert anchor_hour(as_of, 0) == datetime(2024, 6, 1, 13, 0, tzinfo=UTC)


def test_anchor_offset_grows_with_horizon() -> None:
    assert anchor_offset_hours(1, 15) == 3
    assert anchor_offset_hours(6, 15) == 8


def test_anchor_offset_rejects_latency_over_an_hour() -> None:
    """The constant-offset derivation is only valid below one hour."""
    with pytest.raises(ValueError, match="per-row anchor"):
        anchor_offset_hours(1, 90)


def test_spec_has_no_lag_1h_feature() -> None:
    """A lag-1h feature is impossible under any positive latency."""
    spec = FeatureSpec(horizon_hours=1, cutoff_minutes=15)
    assert min(spec.recent_offsets) >= 3
    assert 1 not in spec.recent_offsets


def test_spec_validates_seasonal_lags_are_available() -> None:
    for horizon in (1, 3, 6):
        FeatureSpec(horizon, 15).validate()


def test_spec_rejects_seasonal_lag_newer_than_anchor() -> None:
    """A 24h seasonal lag is unavailable at a 23h horizon."""
    spec = FeatureSpec(horizon_hours=23, cutoff_minutes=15)
    with pytest.raises(ValueError, match="has not arrived"):
        spec.validate()


# --- built features ----------------------------------------------------------


def test_leakage_check_passes(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=1)
    assert_no_future_leakage(con, config, table, horizon_hours=1)


def test_anchor_column_matches_derivation(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=3)
    row = con.execute(f"""
        SELECT target_hour, as_of, anchor_hour FROM {table} LIMIT 1
    """).fetchone()
    assert row is not None
    target, as_of, anchor = row
    assert as_of == target - timedelta(hours=3)
    assert anchor == anchor_hour(as_of, config.features.feature_cutoff_minutes)


def test_recent_feature_equals_anchor_hour_demand(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """recent_1 must be the observed count at the anchor, not near it."""
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=1)
    row = con.execute(f"""
        SELECT f.recent_1, g.pickup_count
        FROM {table} f
        JOIN zone_hour g
          ON g.location_id = f.location_id AND g.hour_utc = f.anchor_hour
        LIMIT 5
    """).fetchall()
    assert row
    assert all(r[0] == r[1] for r in row)


def test_seasonal_lag_equals_same_hour_previous_day(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=1)
    rows = con.execute(f"""
        SELECT f.lag_24h, g.pickup_count
        FROM {table} f
        JOIN zone_hour g
          ON g.location_id = f.location_id
         AND g.hour_utc = f.target_hour - INTERVAL '24 hour'
        LIMIT 5
    """).fetchall()
    assert rows
    assert all(r[0] == r[1] for r in rows)


def test_horizon_shifts_the_anchor_but_not_seasonal_lags(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """Longer horizons see staler recent history, identical seasonal history."""
    _seed_grid(con)
    t1 = build_features(con, config, horizon_hours=1)
    con.execute(f"CREATE OR REPLACE TEMP TABLE keep_h1 AS SELECT * FROM {t1}")
    t6 = build_features(con, config, horizon_hours=6)

    row = con.execute(f"""
        SELECT a.anchor_hour, b.anchor_hour, a.lag_168h, b.lag_168h
        FROM keep_h1 a JOIN {t6} b
          ON a.location_id = b.location_id AND a.target_hour = b.target_hour
        LIMIT 1
    """).fetchone()
    assert row is not None
    assert row[1] == row[0] - timedelta(hours=5)
    assert row[2] == row[3]


def test_no_feature_reads_beyond_the_anchor(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """The invariant, stated directly: nothing may come from after the anchor."""
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=1)
    bad = con.execute(f"""
        SELECT COUNT(*) FROM {table} WHERE anchor_hour >= as_of
    """).fetchone()
    assert bad is not None and bad[0] == 0


def test_rolling_window_excludes_the_target(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """A rolling mean that includes its own target is the classic leak."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE zone_hour AS
        WITH h AS (
            SELECT UNNEST(generate_series(
                TIMESTAMPTZ '2024-01-01 00:00:00+00',
                TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL '1199 hour',
                INTERVAL '1 hour')) AS hour_utc
        )
        SELECT 161 AS location_id, hour_utc,
               CASE WHEN hour_utc = TIMESTAMPTZ '2024-02-10 12:00:00+00'
                    THEN 9999 ELSE 5 END AS pickup_count
        FROM h
    """)
    table = build_features(con, config, horizon_hours=1)
    row = con.execute(f"""
        SELECT roll_mean_3h, roll_mean_24h FROM {table}
        WHERE target_hour = TIMESTAMPTZ '2024-02-10 12:00:00+00'
    """).fetchone()
    assert row is not None
    assert row[0] == 5.0, "3h rolling mean absorbed its own target"
    assert row[1] == 5.0, "24h rolling mean absorbed its own target"


def test_hour_of_week_median_is_baseline_b(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """Baseline B and the feature share one code path so they cannot diverge."""
    _seed_grid(con, hours=1500)
    table = build_features(con, config, horizon_hours=1)
    rows = con.execute(f"""
        SELECT how_median_4w, lag_168h FROM {table} LIMIT 20
    """).fetchall()
    assert rows
    # The seeded pattern repeats weekly, so the median matches last week exactly.
    assert all(r[0] == r[1] for r in rows)


def test_how_median_uses_full_four_weeks(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """Regression: list_median ignores NULLs.

    Before the warm-up trim, rows in weeks 2-4 received a median computed from
    one, two or three observations while still returning a non-NULL value — so a
    nullness filter passed them straight into the training block.
    """
    _seed_grid(con, hours=1500)
    table = build_features(con, config, horizon_hours=1)
    row = con.execute(f"""
        SELECT MIN(target_hour) FROM {table}
    """).fetchone()
    grid_start = con.execute("SELECT MIN(hour_utc) FROM zone_hour").fetchone()
    assert row is not None and grid_start is not None
    assert row[0] >= grid_start[0] + timedelta(hours=4 * 168)


def test_rolling_window_is_never_partial(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """A 168h mean over a constant series must equal that constant exactly.

    A partially-formed window would still average to the constant, so the series
    is made non-constant early to expose truncation.
    """
    con.execute("""
        CREATE OR REPLACE TEMP TABLE zone_hour AS
        WITH h AS (
            SELECT UNNEST(generate_series(
                TIMESTAMPTZ '2024-01-01 00:00:00+00',
                TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL '2999 hour',
                INTERVAL '1 hour')) AS hour_utc
        )
        SELECT 161 AS location_id, hour_utc, 7 AS pickup_count FROM h
    """)
    table = build_features(con, config, horizon_hours=1)
    row = con.execute(f"""
        SELECT MIN(roll_mean_168h), MAX(roll_mean_168h), MIN(how_median_4w) FROM {table}
    """).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(7.0)
    assert row[1] == pytest.approx(7.0)
    assert row[2] == pytest.approx(7.0)


def test_warmup_hours_covers_the_longest_lookback() -> None:
    spec = FeatureSpec(horizon_hours=6, cutoff_minutes=15)
    assert spec.warmup_hours() >= 4 * 168
    assert spec.warmup_hours() >= spec.offset + 167


def test_rows_without_full_history_are_dropped(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    _seed_grid(con, hours=1200)
    table = build_features(con, config, horizon_hours=1)
    nulls = con.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE recent_1 IS NULL OR lag_168h IS NULL OR how_median_4w IS NULL
    """).fetchone()
    assert nulls is not None and nulls[0] == 0


def test_hour_of_week_median_always_uses_four_full_weeks(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """Regression: list_median ignores NULLs.

    Rows in weeks 2-4 once received a one-, two- or three-week median under a
    name promising four, so Baseline B meant something different at the start of
    the series than everywhere else.
    """
    _seed_grid(con, hours=1200, zones=1)
    start = con.execute("SELECT MIN(hour_utc) FROM zone_hour").fetchone()
    assert start is not None
    table = build_features(con, config, horizon_hours=1)

    first = con.execute(f"SELECT MIN(target_hour) FROM {table}").fetchone()
    assert first is not None
    assert first[0] >= start[0] + timedelta(hours=168 * 4)


def test_trim_removes_exactly_the_incomplete_history(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """672 hours per zone, not 168."""
    _seed_grid(con, hours=2000, zones=3)
    grid_rows = con.execute("SELECT COUNT(*) FROM zone_hour").fetchone()
    assert grid_rows is not None
    table = build_features(con, config, horizon_hours=1)
    kept = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert kept is not None
    assert grid_rows[0] - kept[0] == 3 * 168 * 4


def test_helper_column_not_exposed_as_feature(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=1)
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
    assert "how_weeks_present" not in cols


def test_calendar_features_use_local_time(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """2024-06-01 16:00 UTC is noon in New York, not 16:00."""
    _seed_grid(con, hours=5000)
    table = build_features(con, config, horizon_hours=1)
    row = con.execute(f"""
        SELECT local_hour FROM {table}
        WHERE target_hour = TIMESTAMPTZ '2024-06-01 16:00:00+00' LIMIT 1
    """).fetchone()
    assert row is not None and row[0] == 12


def test_weekend_flag_matches_local_day(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _seed_grid(con, hours=1200)
    table = build_features(con, config, horizon_hours=1)
    rows = con.execute(f"""
        SELECT local_dow, is_weekend FROM {table} LIMIT 500
    """).fetchall()
    assert all((r[1] == 1) == (r[0] in (0, 6)) for r in rows)


def test_cyclical_encodings_are_bounded(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=1)
    row = con.execute(f"""
        SELECT MIN(hour_sin), MAX(hour_sin), MIN(dow_cos), MAX(dow_cos) FROM {table}
    """).fetchone()
    assert row is not None
    assert all(-1.0001 <= v <= 1.0001 for v in row)


def test_target_is_not_a_feature(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """Spec §7.4: the target must never appear in the model input columns."""
    _seed_grid(con)
    table = build_features(con, config, horizon_hours=1)
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
    assert "target" not in feature_columns(cols)
    assert "target_hour" not in feature_columns(cols)
    assert "recent_1" in feature_columns(cols)
