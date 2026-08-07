"""Row-level validation and quarantine.

Runs one month at a time through DuckDB so that 103M rows never need to fit in
memory. Every row is assigned a reason code or passes; failing rows are written
out with their code attached and counted, because a validation step that drops
rows without an audit trail is indistinguishable from a bug (spec §6.1).

Timestamps are the subtle part. TLC publishes naive local wall-clock times, so
converting to UTC has to reckon with both DST transitions:

* Spring forward — 02:00-02:59 local does not exist. Rows there are a data error
  and are quarantined as DST_NONEXISTENT rather than silently shifted.
* Autumn fold — 01:00-01:59 local happens twice. Both readings are legitimate
  clock times, so these are resolved to the earlier (daylight) one by policy and
  kept. Quarantining them would discard an hour of real demand twice a year.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb

from urbanflow.config import Config
from urbanflow.validation.contract import DataContract

logger = logging.getLogger(__name__)


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Single-value query result.

    ``fetchone`` is typed as optional; an aggregate always returns a row, so a
    None here means the query was malformed rather than empty.
    """
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"expected one row from: {sql}")
    return int(row[0] or 0)


@dataclass
class MonthValidationResult:
    """Audit record for one month of source data."""

    month: str
    rows_in: int
    rows_out: int
    reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def rows_quarantined(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def quarantine_rate(self) -> float:
        return self.rows_quarantined / self.rows_in if self.rows_in else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "month": self.month,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_quarantined": self.rows_quarantined,
            "quarantine_rate": round(self.quarantine_rate, 6),
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }


def _month_bounds(month: str) -> tuple[str, str]:
    """Half-open [start, end) bounds for a 'YYYY-MM' month."""
    year, mon = (int(p) for p in month.split("-"))
    start = date(year, mon, 1)
    end = date(year + 1, 1, 1) if mon == 12 else date(year, mon + 1, 1)
    return start.isoformat(), end.isoformat()


def _reason_case_sql(config: Config) -> str:
    """SQL CASE expression assigning the first matching reason code to each row.

    Ordering matters: a row with a null timestamp cannot also be meaningfully
    tested for month membership, so the cheapest and most fundamental checks
    come first and later checks assume earlier ones passed.
    """
    excluded = ", ".join(str(z) for z in config.zones.excluded_location_ids)
    max_hours = config.validation_max_duration_hours
    return f"""
        CASE
            WHEN pickup_ts IS NULL THEN 'NULL_PICKUP_TS'
            WHEN zone_id IS NULL THEN 'NULL_ZONE'
            WHEN zone_id < {config.zones.min_location_id}
              OR zone_id > {config.zones.max_location_id} THEN 'ZONE_OUT_OF_RANGE'
            WHEN zone_id IN ({excluded}) THEN 'ZONE_EXCLUDED'
            WHEN pickup_ts < TIMESTAMP '{{start}}' 
              OR pickup_ts >= TIMESTAMP '{{end}}' THEN 'TS_OUT_OF_MONTH'
            WHEN pickup_utc IS NULL THEN 'DST_NONEXISTENT'
            WHEN dropoff_ts IS NOT NULL AND dropoff_ts < pickup_ts
                THEN 'NEGATIVE_DURATION'
            WHEN dropoff_ts IS NOT NULL
             AND date_diff('hour', pickup_ts, dropoff_ts) > {max_hours}
                THEN 'IMPLAUSIBLE_DURATION'
            ELSE NULL
        END
    """


def validate_month(
    con: duckdb.DuckDBPyConnection,
    config: Config,
    month: str,
    *,
    write: bool = True,
) -> MonthValidationResult:
    """Validate one monthly file, writing validated rows and quarantine."""
    raw_path = config.raw_path / f"{config.data.service}_tripdata_{month}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"no raw file for {month} at {raw_path}")

    start, end = _month_bounds(month)
    tz = config.features.timezone

    # Cast to the canonical contract types here — this is where the 2023-01
    # schema era (int64 zones, double passenger_count) is reconciled with later
    # files. The cast is explicit so the migration is visible in the code.
    #
    # trip_distance and total_amount are read for the duplicate signature ONLY.
    # They are never written to the validated corpus and never reach a model
    # (spec §7.4); their sole job is distinguishing two genuinely different trips
    # that happen to share a pickup second from one record submitted twice.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW raw_month AS
        SELECT
            CAST(tpep_pickup_datetime AS TIMESTAMP)  AS pickup_ts,
            CAST(tpep_dropoff_datetime AS TIMESTAMP) AS dropoff_ts,
            CAST(PULocationID AS INTEGER)            AS zone_id,
            CAST(VendorID AS INTEGER)                AS vendor_id,
            CAST(trip_distance AS DOUBLE)            AS dup_distance,
            CAST(total_amount AS DOUBLE)             AS dup_total
        FROM read_parquet('{raw_path}')
    """)

    # A local time that does not survive a round trip through the timezone is one
    # that does not exist (the spring gap). DuckDB maps such times forward, so the
    # round trip returns a different wall clock and the mismatch is the detector.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW tz_month AS
        SELECT
            *,
            CASE
                WHEN timezone('{tz}', timezone('{tz}', pickup_ts)) = pickup_ts
                THEN timezone('{tz}', pickup_ts)
                ELSE NULL
            END AS pickup_utc
        FROM raw_month
    """)

    reason_sql = _reason_case_sql(config).format(start=start, end=end)
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW judged AS
        SELECT *, {reason_sql} AS reason
        FROM tz_month
    """)

    rows_in = _scalar(con, "SELECT COUNT(*) FROM judged")

    reason_rows = con.execute("""
        SELECT reason, COUNT(*) AS n
        FROM judged WHERE reason IS NOT NULL
        GROUP BY reason ORDER BY n DESC
    """).fetchall()
    reason_counts = {str(r[0]): int(r[1]) for r in reason_rows}

    # Duplicates are resolved and counted by one operation, so the count and the
    # written output cannot disagree. An earlier version counted duplicates on one
    # signature and wrote DISTINCT on another; 1.65M rows then disappeared from the
    # real corpus with no reason code attached, which the conservation check caught.
    #
    # The signature is a full-row match. Deduplicating on (pickup_ts, zone, vendor)
    # alone is wrong for demand forecasting: a busy zone sees a pickup every few
    # seconds, so two distinct cabs sharing a pickup second is routine, and
    # collapsing them truncates real demand precisely where volume is highest.
    con.execute("""
        CREATE OR REPLACE TEMP VIEW deduped AS
        SELECT * EXCLUDE (dup_rank) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY pickup_ts, dropoff_ts, zone_id, vendor_id,
                             dup_distance, dup_total
            ) AS dup_rank
            FROM judged WHERE reason IS NULL
        ) WHERE dup_rank = 1
    """)

    passed = _scalar(con, "SELECT COUNT(*) FROM judged WHERE reason IS NULL")
    kept = _scalar(con, "SELECT COUNT(*) FROM deduped")
    if passed != kept:
        reason_counts["DUPLICATE"] = passed - kept

    if write:
        validated_dir = config.validated_path
        quarantine_dir = config.quarantine_path
        validated_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        con.execute(f"""
            COPY (
                SELECT pickup_utc, pickup_ts, zone_id, vendor_id FROM deduped
            ) TO '{validated_dir / f"trips_{month}.parquet"}' (FORMAT PARQUET)
        """)
        con.execute(f"""
            COPY (
                SELECT pickup_ts, dropoff_ts, zone_id, vendor_id, reason
                FROM judged WHERE reason IS NOT NULL
            ) TO '{quarantine_dir / f"quarantine_{month}.parquet"}' (FORMAT PARQUET)
        """)
        out_file = validated_dir / f"trips_{month}.parquet"
        rows_out = _scalar(con, f"SELECT COUNT(*) FROM read_parquet('{out_file}')")
    else:
        rows_out = kept

    return MonthValidationResult(
        month=month,
        rows_in=rows_in,
        rows_out=rows_out,
        reason_counts=reason_counts,
    )


def check_file_against_contract(path: Path, contract: DataContract) -> list[str]:
    """Schema-level gate, run before any row is read."""
    from urbanflow.validation.contract import parquet_schema

    return contract.check_schema(parquet_schema(path))
