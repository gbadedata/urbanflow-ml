"""Dense zone-hour grid construction.

The single most consequential transformation in the project, and the one most
often got wrong: trip records contain no row for an hour with no pickups, so a
`GROUP BY zone, hour` silently omits them. Every lag, rolling window and error
metric computed on that output is then biased, worst in exactly the low-volume
zones where the model is weakest.

The grid is therefore a cross-join of forecastable zones with every hour in
range, left-joined to observed counts and filled with zero. Absence of a trip
record is evidence of zero demand, not absence of evidence.

Hours are generated in UTC, which makes the series uniform: there is no 23- or
25-hour day to special-case. DST only affects locally-derived calendar features,
which are computed downstream and never change the grid's shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from urbanflow.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridStats:
    """Shape and integrity summary for a built grid."""

    zones: int
    hours: int
    rows: int
    zero_rows: int
    total_pickups: int
    city_wide_zero_hours: int
    start_hour: datetime
    end_hour: datetime

    @property
    def expected_rows(self) -> int:
        return self.zones * self.hours

    @property
    def is_dense(self) -> bool:
        return self.rows == self.expected_rows

    @property
    def zero_share(self) -> float:
        return self.zero_rows / self.rows if self.rows else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "zones": self.zones,
            "hours": self.hours,
            "rows": self.rows,
            "expected_rows": self.expected_rows,
            "is_dense": self.is_dense,
            "zero_rows": self.zero_rows,
            "zero_share": round(self.zero_share, 6),
            "total_pickups": self.total_pickups,
            "city_wide_zero_hours": self.city_wide_zero_hours,
            "start_hour": self.start_hour.isoformat(),
            "end_hour": self.end_hour.isoformat(),
        }


def build_grid(
    con: duckdb.DuckDBPyConnection,
    config: Config,
    zone_ids: list[int],
    validated_glob: str | None = None,
) -> GridStats:
    """Build the dense grid as a DuckDB table named ``zone_hour``.

    Returns shape statistics; the caller is expected to assert density rather
    than trust it.
    """
    if not zone_ids:
        raise ValueError("no forecastable zones — run `urbanflow zone-report` first")

    pattern = validated_glob or str(config.validated_path / "trips_*.parquet")
    zone_list = ", ".join(str(z) for z in zone_ids)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW trips AS
        SELECT date_trunc('hour', pickup_utc) AS hour_utc, zone_id
        FROM read_parquet('{pattern}')
        WHERE zone_id IN ({zone_list})
    """)

    bounds = con.execute("SELECT MIN(hour_utc), MAX(hour_utc) FROM trips").fetchone()
    if bounds is None or bounds[0] is None:
        raise ValueError("validated data contains no trips for the forecastable zones")
    start_hour, end_hour = bounds

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE zone_hour AS
        WITH hours AS (
            SELECT UNNEST(generate_series(
                TIMESTAMPTZ '{start_hour}',
                TIMESTAMPTZ '{end_hour}',
                INTERVAL '1 hour'
            )) AS hour_utc
        ),
        zones AS (
            SELECT UNNEST([{zone_list}]) AS location_id
        ),
        grid AS (
            SELECT z.location_id, h.hour_utc
            FROM zones z CROSS JOIN hours h
        ),
        observed AS (
            SELECT zone_id AS location_id, hour_utc, COUNT(*) AS n
            FROM trips GROUP BY 1, 2
        )
        SELECT
            g.location_id,
            g.hour_utc,
            CAST(COALESCE(o.n, 0) AS INTEGER) AS pickup_count
        FROM grid g
        LEFT JOIN observed o
          ON o.location_id = g.location_id AND o.hour_utc = g.hour_utc
    """)

    row = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT location_id), COUNT(DISTINCT hour_utc),
               SUM(CASE WHEN pickup_count = 0 THEN 1 ELSE 0 END), SUM(pickup_count)
        FROM zone_hour
    """).fetchone()
    assert row is not None

    gaps = detect_city_wide_zero_hours(con)

    return GridStats(
        zones=int(row[1]),
        hours=int(row[2]),
        rows=int(row[0]),
        zero_rows=int(row[3] or 0),
        total_pickups=int(row[4] or 0),
        city_wide_zero_hours=len(gaps),
        start_hour=start_hour,
        end_hour=end_hour,
    )


def detect_city_wide_zero_hours(con: duckdb.DuckDBPyConnection) -> list[datetime]:
    """Hours where every forecastable zone recorded zero pickups.

    New York does not stop hailing cabs for an entire hour, so such an hour is
    never observed demand. Two distinct causes produce it:

    * **DST fold artifact.** At the autumn transition local 01:00-01:59 occurs
      twice, spanning two UTC hours, but ambiguous local times all resolve to one
      of them. The other UTC hour is left empty by the conversion, not by the
      city. Expect exactly one per year.
    * **Ingestion gap.** Anything else — an upstream publication failure.

    Both are excluded from training; the distinction matters because the first is
    expected and the second is a defect worth chasing.
    """
    rows = con.execute("""
        SELECT hour_utc FROM zone_hour
        GROUP BY hour_utc HAVING SUM(pickup_count) = 0
        ORDER BY hour_utc
    """).fetchall()
    return [r[0] for r in rows]


def classify_zero_hour(hour: datetime, timezone_name: str = "America/New_York") -> str:
    """Label a city-wide zero hour as a DST artifact or a genuine gap.

    The fold produces two UTC instants sharing one local wall-clock hour with
    different offsets. Either may be the one left empty by the conversion, so
    both neighbours are checked; looking only backwards misses the earlier half
    of the pair, which is the half that is actually emptied.
    """
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(timezone_name)
    local = hour.astimezone(tz)
    for delta in (-1, 1):
        neighbour = (hour + timedelta(hours=delta)).astimezone(tz)
        if local.hour == neighbour.hour and local.utcoffset() != neighbour.utcoffset():
            return "dst_fold_artifact"
    return "ingestion_gap"


def zone_hour_coverage(con: duckdb.DuckDBPyConnection) -> list[tuple[int, float]]:
    """Per-zone share of hours with zero demand.

    A zone at or near 100% is a candidate the volume floor should have caught;
    a sudden change over time suggests a zone entering or leaving service.
    """
    rows = con.execute("""
        SELECT location_id,
               SUM(CASE WHEN pickup_count = 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
        FROM zone_hour GROUP BY location_id ORDER BY 2 DESC
    """).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]
