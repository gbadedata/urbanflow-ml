"""Tests for the dense zone-hour grid.

The density assertion is the release blocker (spec §8, gate 1): the grid must
contain exactly zones x hours rows, with no tolerance. Everything else here
guards against ways that invariant can hold while the contents are still wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from urbanflow.config import Config, load_config
from urbanflow.duck import connect, session_timezone
from urbanflow.features.grid import (
    build_grid,
    detect_city_wide_zero_hours,
    zone_hour_coverage,
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


def _write_trips(config: Config, trips: list[tuple[datetime, int]]) -> Path:
    """Write a validated-trips parquet from (pickup_utc, zone_id) pairs."""
    config.validated_path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "pickup_utc": pa.array([t[0] for t in trips], type=pa.timestamp("us", tz="UTC")),
            "pickup_ts": pa.array([t[0] for t in trips], type=pa.timestamp("us")),
            "zone_id": pa.array([t[1] for t in trips], type=pa.int32()),
            "vendor_id": pa.array([1] * len(trips), type=pa.int32()),
        }
    )
    path = config.validated_path / "trips_2024-03.parquet"
    pq.write_table(table, path)
    return path


BASE = datetime(2024, 3, 5, 0, 0, tzinfo=UTC)


def test_grid_is_dense(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """Two zones over a 5-hour span must yield exactly 10 rows.

    Only three of those hours have any trips at all.
    """
    _write_trips(
        config,
        [(BASE, 161), (BASE + timedelta(hours=2), 161), (BASE + timedelta(hours=4), 237)],
    )
    stats = build_grid(con, config, [161, 237])

    assert stats.zones == 2
    assert stats.hours == 5
    assert stats.rows == 10
    assert stats.is_dense
    assert stats.rows == stats.expected_rows


def test_missing_hours_become_zeros_not_absent_rows(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """The core correction: absence of a trip record is a zero, not a gap."""
    _write_trips(config, [(BASE, 161), (BASE + timedelta(hours=3), 161)])
    build_grid(con, config, [161])

    rows = con.execute("SELECT hour_utc, pickup_count FROM zone_hour ORDER BY hour_utc").fetchall()
    assert [r[1] for r in rows] == [1, 0, 0, 1]


def test_total_pickups_preserved(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """Aggregation must not lose or invent demand."""
    trips = [(BASE + timedelta(hours=h % 4), 161) for h in range(20)]
    _write_trips(config, trips)
    stats = build_grid(con, config, [161])
    assert stats.total_pickups == 20


def test_trips_in_same_hour_are_summed(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _write_trips(config, [(BASE, 161), (BASE + timedelta(minutes=17), 161), (BASE, 161)])
    build_grid(con, config, [161])
    count = con.execute("SELECT pickup_count FROM zone_hour WHERE hour_utc = ?", [BASE]).fetchone()
    assert count is not None and count[0] == 3


def test_hours_are_truncated_to_the_hour(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _write_trips(config, [(BASE + timedelta(minutes=43, seconds=12), 161)])
    build_grid(con, config, [161])
    rows = con.execute("SELECT hour_utc FROM zone_hour").fetchall()
    assert all(r[0].minute == 0 and r[0].second == 0 for r in rows)


def test_excluded_zones_absent_from_grid(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """Trips in a non-forecastable zone must not create grid rows."""
    _write_trips(config, [(BASE, 161), (BASE, 264)])
    build_grid(con, config, [161])

    zones = con.execute("SELECT DISTINCT location_id FROM zone_hour").fetchall()
    assert [z[0] for z in zones] == [161]


def test_zone_with_no_trips_still_gets_full_row_set(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """A forecastable zone that saw nothing is all zeros, not missing."""
    _write_trips(config, [(BASE, 161), (BASE + timedelta(hours=2), 161)])
    stats = build_grid(con, config, [161, 237])

    assert stats.rows == 6
    counts = con.execute(
        "SELECT SUM(pickup_count) FROM zone_hour WHERE location_id = 237"
    ).fetchone()
    assert counts is not None and counts[0] == 0


def test_city_wide_zero_hour_detected(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """An hour with zero demand across every zone is a pipeline failure."""
    _write_trips(
        config,
        [
            (BASE, 161),
            (BASE, 237),
            (BASE + timedelta(hours=2), 161),
            (BASE + timedelta(hours=2), 237),
        ],
    )
    build_grid(con, config, [161, 237])

    gaps = detect_city_wide_zero_hours(con)
    assert len(gaps) == 1
    assert gaps[0] == BASE + timedelta(hours=1)


def test_no_city_wide_gap_when_one_zone_active(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """A quiet hour in one zone is normal and must not be flagged."""
    _write_trips(
        config, [(BASE, 161), (BASE + timedelta(hours=1), 237), (BASE + timedelta(hours=2), 161)]
    )
    build_grid(con, config, [161, 237])
    assert detect_city_wide_zero_hours(con) == []


def test_zero_share_reported(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _write_trips(config, [(BASE, 161), (BASE + timedelta(hours=3), 161)])
    stats = build_grid(con, config, [161])
    assert stats.zero_rows == 2
    assert stats.zero_share == pytest.approx(0.5)


def test_coverage_ranks_sparsest_zone_first(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _write_trips(
        config,
        [(BASE + timedelta(hours=h), 161) for h in range(4)] + [(BASE, 237)],
    )
    build_grid(con, config, [161, 237])
    coverage = zone_hour_coverage(con)
    assert coverage[0][0] == 237, "the sparsest zone should sort first"


def test_grid_spans_utc_dst_transition_uniformly(
    con: duckdb.DuckDBPyConnection, config: Config
) -> None:
    """Storing UTC removes the 23/25-hour day problem entirely.

    2024-03-10 is the US spring-forward date; in local time that day has 23
    hours, but the UTC grid must still produce exactly 24 rows per zone per day.
    """
    day = datetime(2024, 3, 10, 0, 0, tzinfo=UTC)
    _write_trips(config, [(day, 161), (day + timedelta(hours=23), 161)])
    stats = build_grid(con, config, [161])
    assert stats.hours == 24
    assert stats.rows == 24


def test_session_timezone_is_pinned_to_utc(con: duckdb.DuckDBPyConnection) -> None:
    """Regression: DuckDB defaults to the host timezone.

    Built on a Europe/London machine the grid reported four city-wide zero
    hours, two of them on UK clock-change dates that have nothing to do with New
    York taxis. Identical code in a UTC CI runner produced different output.
    """
    assert session_timezone(con) == "UTC"


def test_grid_is_timezone_independent(config: Config) -> None:
    """The same inputs must produce the same grid regardless of host zone."""
    import os

    _write_trips(config, [(BASE, 161), (BASE + timedelta(hours=3), 161)])

    results = []
    original = os.environ.get("TZ")
    try:
        for tz in ("UTC", "Europe/London", "America/New_York"):
            os.environ["TZ"] = tz
            local_con = connect()
            stats = build_grid(local_con, config, [161])
            results.append((stats.rows, stats.total_pickups, stats.zero_rows))
            local_con.close()
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original

    assert len(set(results)) == 1, f"grid varied by host timezone: {results}"


def test_dst_fold_hour_classified_as_artifact() -> None:
    """The empty UTC hour at autumn fall-back is a conversion artifact."""
    from urbanflow.features.grid import classify_zero_hour

    # 2024-11-03 05:00 UTC is 01:00 EDT; 06:00 UTC is 01:00 EST — the fold.
    assert classify_zero_hour(datetime(2024, 11, 3, 5, 0, tzinfo=UTC)) == "dst_fold_artifact"


def test_ordinary_empty_hour_classified_as_gap() -> None:
    from urbanflow.features.grid import classify_zero_hour

    assert classify_zero_hour(datetime(2024, 6, 15, 12, 0, tzinfo=UTC)) == "ingestion_gap"


def test_uk_clock_change_is_not_an_artifact() -> None:
    """A UK transition must never be treated as meaningful for NYC data."""
    from urbanflow.features.grid import classify_zero_hour

    assert classify_zero_hour(datetime(2024, 10, 27, 1, 0, tzinfo=UTC)) == "ingestion_gap"


def test_empty_zone_list_raises(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    _write_trips(config, [(BASE, 161)])
    with pytest.raises(ValueError, match="no forecastable zones"):
        build_grid(con, config, [])


def test_no_matching_trips_raises(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    """Building a grid from zones with no data is a configuration error."""
    _write_trips(config, [(BASE, 161)])
    with pytest.raises(ValueError, match="no trips"):
        build_grid(con, config, [999])
