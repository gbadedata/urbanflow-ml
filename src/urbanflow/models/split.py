"""Three-block temporal split.

Spec §7.1 and §6.2. Three blocks, not two, because fitting calibration and
selecting the abstention threshold on the same data overfits the threshold and
makes the coverage-accuracy curve optimistic.

    train      earliest    model fitting
    calibrate  middle      model selection, calibration, threshold
    test       latest      read once, at release

Two rules make the split honest:

* **Temporal, never random.** A random split lets the model see next week while
  predicting last week, which is not a situation it will ever face.
* **A gap between blocks.** Without one, the last training rows and the first
  validation rows share seasonal lags drawn from the same hours. The gap is at
  least the longest horizon so a target in one block cannot be a feature in
  another.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import duckdb

from urbanflow.config import Config

TRAIN = "train"
CALIBRATE = "calibrate"
TEST = "test"


@dataclass(frozen=True)
class SplitBounds:
    """Half-open [start, end) bounds for each block."""

    train_start: datetime
    train_end: datetime
    calibrate_start: datetime
    calibrate_end: datetime
    test_start: datetime
    test_end: datetime
    gap_hours: int

    def block_for(self, hour: datetime) -> str | None:
        if self.train_start <= hour < self.train_end:
            return TRAIN
        if self.calibrate_start <= hour < self.calibrate_end:
            return CALIBRATE
        if self.test_start <= hour < self.test_end:
            return TEST
        return None

    def validate(self) -> None:
        """Assert the blocks are ordered, disjoint and properly separated."""
        if not self.train_start < self.train_end <= self.calibrate_start:
            raise ValueError("train block overlaps or follows calibrate")
        if not self.calibrate_start < self.calibrate_end <= self.test_start:
            raise ValueError("calibrate block overlaps or follows test")
        if not self.test_start < self.test_end:
            raise ValueError("test block is empty")

        gap = timedelta(hours=self.gap_hours)
        if self.calibrate_start - self.train_end < gap:
            raise ValueError(
                f"gap between train and calibrate is "
                f"{self.calibrate_start - self.train_end}, needs {gap}"
            )
        if self.test_start - self.calibrate_end < gap:
            raise ValueError(
                f"gap between calibrate and test is "
                f"{self.test_start - self.calibrate_end}, needs {gap}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "train": [self.train_start.isoformat(), self.train_end.isoformat()],
            "calibrate": [self.calibrate_start.isoformat(), self.calibrate_end.isoformat()],
            "test": [self.test_start.isoformat(), self.test_end.isoformat()],
            "gap_hours": self.gap_hours,
        }


def compute_bounds(first_hour: datetime, last_hour: datetime, config: Config) -> SplitBounds:
    """Derive block boundaries from the data span and configuration.

    The test block is the most recent ``holdout_weeks``; the calibrate block is
    the same length immediately before it. Gaps are carved out of the *earlier*
    block so the frozen test period keeps its full declared length.
    """
    holdout = timedelta(weeks=config.split.holdout_weeks)
    gap = timedelta(hours=max(config.split.gap_hours, max(config.forecast.horizons_hours)))
    end = last_hour + timedelta(hours=1)

    test_start = end - holdout
    calibrate_end = test_start - gap
    calibrate_start = calibrate_end - holdout + gap
    train_end = calibrate_start - gap

    bounds = SplitBounds(
        train_start=first_hour,
        train_end=train_end,
        calibrate_start=calibrate_start,
        calibrate_end=calibrate_end,
        test_start=test_start,
        test_end=end,
        gap_hours=int(gap.total_seconds() // 3600),
    )
    bounds.validate()
    return bounds


def assign_blocks(con: duckdb.DuckDBPyConnection, table: str, bounds: SplitBounds) -> str:
    """Add a ``block`` column to a feature table; returns the table name.

    Rows falling in a gap are labelled NULL and excluded from every block, which
    is the point of the gap.
    """
    con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS block VARCHAR")
    con.execute(f"""
        UPDATE {table} SET block = CASE
            WHEN target_hour >= TIMESTAMPTZ '{bounds.train_start}'
             AND target_hour <  TIMESTAMPTZ '{bounds.train_end}'      THEN '{TRAIN}'
            WHEN target_hour >= TIMESTAMPTZ '{bounds.calibrate_start}'
             AND target_hour <  TIMESTAMPTZ '{bounds.calibrate_end}'  THEN '{CALIBRATE}'
            WHEN target_hour >= TIMESTAMPTZ '{bounds.test_start}'
             AND target_hour <  TIMESTAMPTZ '{bounds.test_end}'       THEN '{TEST}'
            ELSE NULL
        END
    """)
    return table


def block_counts(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, int]:
    rows = con.execute(f"""
        SELECT COALESCE(block, 'gap') AS b, COUNT(*) FROM {table} GROUP BY 1
    """).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


def assert_no_temporal_overlap(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Every hour in an earlier block must precede every hour in a later one."""
    rows = con.execute(f"""
        SELECT block, MIN(target_hour), MAX(target_hour)
        FROM {table} WHERE block IS NOT NULL GROUP BY block
    """).fetchall()
    spans = {str(r[0]): (r[1], r[2]) for r in rows}

    order = [b for b in (TRAIN, CALIBRATE, TEST) if b in spans]
    for earlier, later in zip(order, order[1:], strict=False):
        if spans[earlier][1] >= spans[later][0]:
            raise ValueError(
                f"{earlier} block ends at {spans[earlier][1]} but {later} starts at "
                f"{spans[later][0]} — blocks overlap in time"
            )
