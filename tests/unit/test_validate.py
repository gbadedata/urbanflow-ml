"""Tests for row-level validation and quarantine.

Each rule gets a row in the dirty-month fixture, so a rule that stops firing
fails a test rather than quietly passing bad data through. The DST cases are
separated because they are the ones most likely to be got wrong silently: a
nonexistent time shifted forward looks like valid data forever after.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from urbanflow.config import Config, load_config
from urbanflow.validation.validate import _month_bounds, validate_month


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Config pointed at a temporary directory tree."""
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
    return duckdb.connect()


def test_month_bounds_regular_month() -> None:
    assert _month_bounds("2024-03") == ("2024-03-01", "2024-04-01")


def test_month_bounds_december_rolls_year() -> None:
    """December's upper bound is the following January, not month 13."""
    assert _month_bounds("2024-12") == ("2024-12-01", "2025-01-01")


def test_month_bounds_february_leap_year() -> None:
    start, end = _month_bounds("2024-02")
    assert (start, end) == ("2024-02-01", "2024-03-01")
    assert (date.fromisoformat(end) - date.fromisoformat(start)).days == 29


def test_clean_month_passes_everything(
    con: duckdb.DuckDBPyConnection, config: Config, clean_month: Path
) -> None:
    result = validate_month(con, config, "2024-03")
    assert result.rows_in == 24
    assert result.rows_out == 24
    assert result.reason_counts == {}
    assert result.quarantine_rate == 0.0


def test_legacy_era_validates_after_cast(
    con: duckdb.DuckDBPyConnection, config: Config, legacy_month: Path
) -> None:
    """The 2023-01 int64/double era must survive the canonical cast."""
    result = validate_month(con, config, "2023-01")
    assert result.rows_in == 24
    assert result.rows_out == 24


def test_every_rule_fires(
    con: duckdb.DuckDBPyConnection, config: Config, dirty_month: Path
) -> None:
    result = validate_month(con, config, "2024-03")
    assert result.rows_in == 10
    assert result.rows_out == 2, "only the two clean rows should survive"

    for code in (
        "NULL_PICKUP_TS",
        "NULL_ZONE",
        "ZONE_OUT_OF_RANGE",
        "ZONE_EXCLUDED",
        "TS_OUT_OF_MONTH",
        "DST_NONEXISTENT",
        "NEGATIVE_DURATION",
        "IMPLAUSIBLE_DURATION",
    ):
        assert result.reason_counts.get(code) == 1, f"{code} did not fire exactly once"


def test_rows_are_conserved(
    con: duckdb.DuckDBPyConnection, config: Config, dirty_month: Path
) -> None:
    """Nothing may vanish: every input row is either output or quarantined."""
    result = validate_month(con, config, "2024-03")
    assert result.rows_out + sum(result.reason_counts.values()) == result.rows_in


def test_reason_codes_are_declared_in_contract(
    con: duckdb.DuckDBPyConnection, config: Config, dirty_month: Path
) -> None:
    """A reason code the contract does not declare is undocumented behaviour."""
    from urbanflow.validation.contract import load_contract

    result = validate_month(con, config, "2024-03")
    assert set(result.reason_counts) <= load_contract().rule_codes


def test_quarantine_file_carries_reasons(
    con: duckdb.DuckDBPyConnection, config: Config, dirty_month: Path
) -> None:
    """Spec §6.1: rejected rows are traceable to the rule that rejected them."""
    validate_month(con, config, "2024-03")
    path = config.quarantine_path / "quarantine_2024-03.parquet"
    assert path.exists()

    rows = con.execute(
        f"SELECT reason, COUNT(*) FROM read_parquet('{path}') GROUP BY reason"
    ).fetchall()
    assert len(rows) == 8
    assert all(r[0] is not None for r in rows)


def test_duplicates_removed_and_counted(
    con: duckdb.DuckDBPyConnection, config: Config, duplicate_month: Path
) -> None:
    result = validate_month(con, config, "2024-03")
    assert result.rows_in == 4
    assert result.rows_out == 2
    assert result.reason_counts["DUPLICATE"] == 2


def test_concurrent_trips_are_not_deduplicated(
    con: duckdb.DuckDBPyConnection, config: Config, concurrent_trips_month: Path
) -> None:
    """Real demand must survive: same second, same zone, same vendor, different trips."""
    result = validate_month(con, config, "2024-03")
    assert result.rows_in == 3
    assert result.rows_out == 3, "distinct trips collapsed into one — demand truncated"
    assert "DUPLICATE" not in result.reason_counts


def test_conservation_holds_with_near_duplicates(
    con: duckdb.DuckDBPyConnection, config: Config, concurrent_trips_month: Path
) -> None:
    """Regression: rows once vanished between the duplicate count and the write.

    The count used one signature and the output another, so 1.65M rows left the
    real corpus with no reason code. Count and write now derive from one operation.
    """
    result = validate_month(con, config, "2024-03")
    assert result.rows_out + sum(result.reason_counts.values()) == result.rows_in


def test_dst_fold_rows_are_kept(
    con: duckdb.DuckDBPyConnection, config: Config, dst_fold_month: Path
) -> None:
    """The autumn fold hour is real demand and must not be quarantined."""
    result = validate_month(con, config, "2024-11")
    assert result.rows_out == 3
    assert "DST_NONEXISTENT" not in result.reason_counts


def test_validated_output_has_utc_timestamp(
    con: duckdb.DuckDBPyConnection, config: Config, clean_month: Path
) -> None:
    validate_month(con, config, "2024-03")
    path = config.validated_path / "trips_2024-03.parquet"
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
    assert "pickup_utc" in cols
    assert "zone_id" in cols


def test_validated_output_excludes_prohibited_columns(
    con: duckdb.DuckDBPyConnection, config: Config, clean_month: Path
) -> None:
    """Spec §7.4: fare and dropoff fields must not reach the modelling corpus."""
    from urbanflow.validation.contract import load_contract

    validate_month(con, config, "2024-03")
    path = config.validated_path / "trips_2024-03.parquet"
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
    assert not cols & load_contract().prohibited_columns


def test_dry_run_writes_nothing(
    con: duckdb.DuckDBPyConnection, config: Config, dirty_month: Path
) -> None:
    result = validate_month(con, config, "2024-03", write=False)
    assert result.rows_out == 2
    assert not (config.validated_path / "trips_2024-03.parquet").exists()


def test_missing_file_raises(con: duckdb.DuckDBPyConnection, config: Config) -> None:
    with pytest.raises(FileNotFoundError):
        validate_month(con, config, "2019-07")
