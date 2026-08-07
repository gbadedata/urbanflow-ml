"""DuckDB connection management.

DuckDB resolves TIMESTAMPTZ operations — ``date_trunc``, ``generate_series``,
display, and comparison — against the *session* timezone, which defaults to the
host machine's local zone. That makes the pipeline's output depend on where it
runs: a developer in Europe/London and a CI runner in UTC would produce
different grids from identical inputs, and the discrepancy appears only around
DST boundaries where it is easiest to misread as a data problem.

Every connection is therefore pinned to UTC. New York local time is still
obtained where it is genuinely needed, but always by naming the zone explicitly
rather than by inheriting it from the environment.
"""

from __future__ import annotations

import duckdb

SESSION_TIMEZONE = "UTC"


def connect(database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with a deterministic session timezone."""
    con = duckdb.connect(database)
    con.execute(f"SET TimeZone='{SESSION_TIMEZONE}'")
    return con


def session_timezone(con: duckdb.DuckDBPyConnection) -> str:
    """Report the connection's effective timezone, for assertions."""
    row = con.execute("SELECT current_setting('TimeZone')").fetchone()
    if row is None:
        raise RuntimeError("could not read session timezone")
    return str(row[0])
