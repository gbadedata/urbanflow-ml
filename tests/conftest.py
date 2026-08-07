"""Synthetic TLC fixtures.

Tests run against generated Parquet rather than real trip data so that CI needs
no network and no 1.5GB download. The generators deliberately reproduce the two
schema eras observed in the real snapshot, because the whole point of the
contract layer is surviving that difference.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Observed in the real data: 2023-01 shipped int64 zone IDs and a lowercase
# airport_fee; from 2023-02 zones are int32 and the column is Airport_fee.
ERA_2023_01 = "era_2023_01"
ERA_MODERN = "era_modern"


def _write(path: Path, rows: list[dict[str, object]], era: str) -> Path:
    pickup = pa.array([r["pickup"] for r in rows], type=pa.timestamp("us"))
    dropoff = pa.array([r["dropoff"] for r in rows], type=pa.timestamp("us"))
    distance = pa.array([r.get("distance", 1.5) for r in rows], type=pa.float64())

    if era == ERA_2023_01:
        zone = pa.array([r["zone"] for r in rows], type=pa.int64())
        vendor = pa.array([1] * len(rows), type=pa.int64())
        extra_name, extra = "airport_fee", pa.array([0.0] * len(rows), type=pa.float64())
    else:
        zone = pa.array([r["zone"] for r in rows], type=pa.int32())
        vendor = pa.array([1] * len(rows), type=pa.int32())
        extra_name, extra = "Airport_fee", pa.array([0.0] * len(rows), type=pa.float64())

    table = pa.table(
        {
            "VendorID": vendor,
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff,
            "PULocationID": zone,
            "DOLocationID": zone,
            "trip_distance": distance,
            "fare_amount": pa.array([10.0] * len(rows), type=pa.float64()),
            "total_amount": pa.array([r.get("total", 12.0) for r in rows], type=pa.float64()),
            extra_name: extra,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _row(
    pickup: datetime | None,
    zone: int | None,
    minutes: int = 12,
    distance: float = 1.5,
    total: float = 12.0,
) -> dict[str, object]:
    dropoff = None
    if pickup is not None:
        from datetime import timedelta

        dropoff = pickup + timedelta(minutes=minutes)
    return {
        "pickup": pickup,
        "dropoff": dropoff,
        "zone": zone,
        "distance": distance,
        "total": total,
    }


@pytest.fixture
def clean_month(tmp_path: Path) -> Path:
    """A month of unproblematic rows in the modern schema."""
    rows = [_row(datetime(2024, 3, 5, h, 30), 161) for h in range(24)]
    return _write(tmp_path / "yellow_tripdata_2024-03.parquet", rows, ERA_MODERN)


@pytest.fixture
def legacy_month(tmp_path: Path) -> Path:
    """The 2023-01 schema era: int64 zones, lowercase airport_fee."""
    rows = [_row(datetime(2023, 1, 5, h, 30), 161) for h in range(24)]
    return _write(tmp_path / "yellow_tripdata_2023-01.parquet", rows, ERA_2023_01)


@pytest.fixture
def dirty_month(tmp_path: Path) -> Path:
    """One row for each failure mode the contract declares.

    Dated March 2024 so that the spring-forward gap (2024-03-10 02:00-02:59
    America/New_York) falls inside the nominal month.
    """
    rows = [
        _row(datetime(2024, 3, 5, 9, 0), 161),  # valid
        _row(datetime(2024, 3, 5, 10, 0), 237),  # valid, different zone
        _row(None, 161),  # NULL_PICKUP_TS
        _row(datetime(2024, 3, 5, 11, 0), None),  # NULL_ZONE
        _row(datetime(2024, 3, 5, 12, 0), 999),  # ZONE_OUT_OF_RANGE
        _row(datetime(2024, 3, 5, 13, 0), 264),  # ZONE_EXCLUDED
        _row(datetime(2002, 1, 1, 0, 0), 161),  # TS_OUT_OF_MONTH
        _row(datetime(2024, 3, 10, 2, 30), 161),  # DST_NONEXISTENT
        _row(datetime(2024, 3, 5, 14, 0), 161, minutes=-30),  # NEGATIVE_DURATION
        _row(datetime(2024, 3, 5, 15, 0), 161, minutes=60 * 30),  # IMPLAUSIBLE_DURATION
    ]
    return _write(tmp_path / "yellow_tripdata_2024-03.parquet", rows, ERA_MODERN)


@pytest.fixture
def duplicate_month(tmp_path: Path) -> Path:
    """Three identical valid rows plus one distinct row."""
    dup = _row(datetime(2024, 3, 5, 9, 0), 161)
    rows = [dup, dict(dup), dict(dup), _row(datetime(2024, 3, 5, 10, 0), 161)]
    return _write(tmp_path / "yellow_tripdata_2024-03.parquet", rows, ERA_MODERN)


@pytest.fixture
def concurrent_trips_month(tmp_path: Path) -> Path:
    """Distinct trips that collide on pickup second, zone and vendor.

    Routine in a busy zone — Midtown sees a pickup every few seconds. These are
    real demand and must survive deduplication; an earlier signature that ignored
    dropoff, distance and fare collapsed them and truncated peak volume.
    """
    ts = datetime(2024, 3, 5, 9, 0)
    rows = [
        _row(ts, 161, minutes=10, distance=1.2, total=11.0),
        _row(ts, 161, minutes=25, distance=4.8, total=28.0),
        _row(ts, 161, minutes=10, distance=3.3, total=19.5),
    ]
    return _write(tmp_path / "yellow_tripdata_2024-03.parquet", rows, ERA_MODERN)


@pytest.fixture
def dst_fold_month(tmp_path: Path) -> Path:
    """The autumn fold: 2024-11-03 01:30 local occurs twice, both legitimate."""
    rows = [
        _row(datetime(2024, 11, 3, 1, 30), 161),
        _row(datetime(2024, 11, 3, 1, 30), 237),
        _row(datetime(2024, 11, 3, 3, 0), 161),
    ]
    return _write(tmp_path / "yellow_tripdata_2024-11.parquet", rows, ERA_MODERN)
