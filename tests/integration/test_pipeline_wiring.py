"""End-to-end wiring checks.

Every component was individually tested and the champion still silently failed to
run: a CLI patch renamed the output file but never passed the flag through, so
the command reported success having fitted nothing. Unit tests cannot see that —
the defect lived entirely in the glue.

These tests exercise the seam between the CLI and the model layer, asserting the
key names the CLI indexes are the ones the evaluator actually produces.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import yaml

from urbanflow.config import DEFAULT_CONFIG_PATH, Config, load_config
from urbanflow.duck import connect
from urbanflow.features.build import build_features, feature_columns
from urbanflow.models.baselines import BASELINE_A, BASELINE_B, BASELINE_C
from urbanflow.models.champion import CHAMPION, evaluate_gate
from urbanflow.models.evaluate import evaluate_baselines
from urbanflow.models.split import assign_blocks, compute_bounds


@pytest.fixture
def config(tmp_path: Path) -> Config:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    for key in ("raw_dir", "validated_dir", "quarantine_dir"):
        raw["data"][key] = str(tmp_path)
    raw["split"]["holdout_weeks"] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture
def prepared(config: Config) -> tuple[duckdb.DuckDBPyConnection, str, list[str]]:
    """A small grid carried all the way to a split feature table."""
    con = connect()
    con.execute("""
        CREATE TEMP TABLE zone_hour AS
        WITH h AS (
            SELECT UNNEST(generate_series(
                TIMESTAMPTZ '2024-01-01 00:00:00+00',
                TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL '3999 hour',
                INTERVAL '1 hour')) AS hour_utc
        ), z AS (SELECT UNNEST([161, 237]) AS location_id)
        SELECT z.location_id, h.hour_utc,
               CAST(GREATEST(0, 30 + 20 * SIN(2 * PI() * EXTRACT(hour FROM h.hour_utc) / 24)
                    + (CASE WHEN z.location_id = 237 THEN -28 ELSE 0 END)) AS INTEGER)
               AS pickup_count
        FROM z CROSS JOIN h
    """)
    table = build_features(con, config, horizon_hours=1)
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
    features = feature_columns(cols)

    span = con.execute(f"SELECT MIN(target_hour), MAX(target_hour) FROM {table}").fetchone()
    assert span is not None
    assign_blocks(con, table, compute_bounds(span[0], span[1], config))
    return con, table, features


def test_baselines_produce_the_expected_keys(
    config: Config, prepared: tuple[duckdb.DuckDBPyConnection, str, list[str]]
) -> None:
    con, table, features = prepared
    results = evaluate_baselines(con, config, table, features, {161: "high", 237: "medium"})
    assert set(results) == {BASELINE_A, BASELINE_B, BASELINE_C}


def test_champion_flag_actually_fits_a_champion(
    config: Config, prepared: tuple[duckdb.DuckDBPyConnection, str, list[str]]
) -> None:
    """Regression: the flag once renamed the report without fitting anything."""
    con, table, features = prepared
    results = evaluate_baselines(
        con, config, table, features, {161: "high", 237: "medium"}, with_champion=True
    )
    assert CHAMPION in results, "champion requested but not present in results"
    assert results[CHAMPION], "champion produced no metrics"


def test_champion_absent_when_not_requested(
    config: Config, prepared: tuple[duckdb.DuckDBPyConnection, str, list[str]]
) -> None:
    con, table, features = prepared
    results = evaluate_baselines(con, config, table, features, {161: "high", 237: "medium"})
    assert CHAMPION not in results


def test_gate_consumes_the_keys_the_cli_indexes(
    config: Config, prepared: tuple[duckdb.DuckDBPyConnection, str, list[str]]
) -> None:
    """The CLI looks up results[CHAMPION] and results[BASELINE_B] by name."""
    con, table, features = prepared
    results = evaluate_baselines(
        con, config, table, features, {161: "high", 237: "medium"}, with_champion=True
    )
    gate = evaluate_gate(config, 1, results[CHAMPION], results[BASELINE_B])
    assert gate.ships in (CHAMPION, BASELINE_B)
    assert gate.reasons


def test_cli_exposes_the_champion_option() -> None:
    """The option must reach the evaluator, not merely rename the output."""
    import inspect

    from urbanflow.cli import baselines_cmd

    params = inspect.signature(baselines_cmd).parameters
    assert "champion" in params
    source = inspect.getsource(baselines_cmd)
    assert "with_champion=champion" in source, "flag not passed through to the evaluator"


def test_max_iter_reaches_the_champion(
    config: Config, prepared: tuple[duckdb.DuckDBPyConnection, str, list[str]]
) -> None:
    """Same seam that broke before: an option must change behaviour, not just names."""
    import inspect

    from urbanflow.cli import baselines_cmd

    source = inspect.getsource(baselines_cmd)
    assert "max_iter=max_iter" in source, "max_iter not passed through to the evaluator"


def test_intervals_command_wired() -> None:
    import inspect

    from urbanflow.cli import intervals_cmd

    params = inspect.signature(intervals_cmd).parameters
    assert "max_iter" in params
    assert "frozen_test" in params
    source = inspect.getsource(intervals_cmd)
    assert "check_coverage_gate" in source, "coverage gate never applied"
    assert "max_iter=max_iter" in source
