"""Database access.

Deliberately thin. The serving path needs three things: a window of demand
history for a zone, the zone registry, and the set of hours excluded from
training. Everything else the API returns comes from the model bundle.

Queries are written in portable SQL so the same code runs against PostgreSQL in
deployment and SQLite in tests, which keeps the tested path and the served path
the same one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import Engine, create_engine, text

DEFAULT_URL = "postgresql+psycopg://urbanflow:urbanflow@localhost:5432/urbanflow"


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or database_url(), pool_pre_ping=True)


def split_statements(sql: str) -> list[str]:
    """Split a DDL script into executable statements.

    Line comments are removed first: a naive split on ";" turns the file's
    leading comment block into a statement of pure prose, which the driver
    reports as "incomplete input" — an error that says nothing about the cause.
    """
    lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def apply_schema(engine: Engine, schema_path: Path) -> None:
    """Run the DDL. Idempotent — every statement is CREATE IF NOT EXISTS."""
    with engine.begin() as conn:
        for statement in split_statements(schema_path.read_text()):
            conn.execute(text(statement))


@dataclass(frozen=True)
class ZoneRecord:
    location_id: int
    zone_name: str | None
    borough: str | None
    demand_tier: str | None
    is_forecastable: bool


class DemandRepository:
    """Read access to the dense grid and zone registry."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def forecastable_zones(self) -> list[ZoneRecord]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT location_id, zone_name, borough, demand_tier, is_forecastable
                    FROM taxi_zones WHERE is_forecastable = :flag
                    ORDER BY location_id
                """),
                {"flag": True},
            ).fetchall()
        return [ZoneRecord(int(r[0]), r[1], r[2], r[3], bool(r[4])) for r in rows]

    def zone_exists(self, location_id: int) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM taxi_zones WHERE location_id = :zone AND is_forecastable = :flag"
                ),
                {"zone": location_id, "flag": True},
            ).fetchone()
        return row is not None

    def latest_hour(self) -> datetime | None:
        """Most recent hour present in the grid.

        Drivers differ on timestamp handling — PostgreSQL returns a datetime,
        SQLite a string — so the value is normalised here rather than leaving
        each caller to discover the difference.
        """
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT MAX(hour_utc) FROM zone_hour")).fetchone()
        if not row or row[0] is None:
            return None
        value = row[0]
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        assert isinstance(value, datetime)
        return value

    def history(self, location_id: int, target_hour: datetime, hours: int) -> np.ndarray:
        """Demand for the `hours` hours ending at `target_hour`, newest first.

        Returned so that index k is k hours before the target, matching the
        ``LAG(pickup_count, k)`` convention the training features use.

        Missing rows are an error rather than a zero fill: the grid is dense by
        construction (spec §2.2), so a gap means the database is incomplete, and
        silently substituting zeros would fabricate the very demand history the
        forecast depends on.
        """
        earliest = target_hour - timedelta(hours=hours - 1)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT hour_utc, pickup_count FROM zone_hour
                    WHERE location_id = :zone
                      AND hour_utc >= :earliest AND hour_utc <= :target
                    ORDER BY hour_utc DESC
                """),
                {"zone": location_id, "earliest": earliest, "target": target_hour},
            ).fetchall()

        if len(rows) < hours:
            raise LookupError(
                f"zone {location_id} has {len(rows)} of {hours} required hours ending "
                f"{target_hour.isoformat()}; the grid is incomplete for this request"
            )
        return np.array([r[1] for r in rows], dtype=np.float64)
