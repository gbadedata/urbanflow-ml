"""API contract tests.

Run against a real SQLite database and real fitted models rather than mocks, so
the tested path is the served path. Mocked repositories would have passed
happily while the SQL was wrong.

The error cases matter as much as the happy one: an unknown zone, an
unsupported horizon, and an origin without enough history are ordinary
conditions a client must be able to distinguish, not faults.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from urbanflow.models.champion import fit_champion
from urbanflow.models.persist import ModelBundle

HORIZONS = (1, 3, 6)
ZONE = 161
GRID_HOURS = 1500
GRID_START = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


@pytest.fixture
def database(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A populated SQLite database with a dense grid for one zone."""
    from urbanflow.config import PROJECT_ROOT
    from urbanflow.serving.repository import split_statements

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    schema = (PROJECT_ROOT / "sql" / "schema.sql").read_text()

    # The API reads only taxi_zones and zone_hour, so only those are created
    # here. The forecasts and model_runs tables use PostgreSQL interval
    # arithmetic that SQLite cannot parse; they are exercised by `load-db`
    # against a real PostgreSQL instance rather than emulated badly here.
    schema = re.sub(r"COMMENT ON[^;]*;", "", schema, flags=re.S)
    statements = [
        s
        for s in split_statements(schema)
        if ("taxi_zones" in s or "zone_hour" in s) and "forecasts" not in s
    ]
    statements = [
        s.replace("TIMESTAMPTZ", "TIMESTAMP")
        .replace("DOUBLE PRECISION", "REAL")
        .replace("date_trunc('hour', hour_utc) = hour_utc", "1 = 1")
        for s in statements
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))

    rng = np.random.default_rng(0)
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO taxi_zones (location_id, borough, zone_name, is_forecastable,
                    exclusion_reason, demand_tier, mean_per_hour, trips_observed)
                VALUES (:id, 'Manhattan', 'Midtown Center', 1, NULL, 'high', 50.0, 75000)
            """),
            {"id": ZONE},
        )
        conn.execute(
            text("""
                INSERT INTO taxi_zones (location_id, borough, zone_name, is_forecastable,
                    exclusion_reason, demand_tier, mean_per_hour, trips_observed)
                VALUES (264, 'Unknown', 'NA', 0, 'structural', 'excluded', 0.0, 0)
            """)
        )
        rows = [
            {
                "z": ZONE,
                "h": GRID_START + timedelta(hours=i),
                "c": int(rng.poisson(30 + 20 * np.sin(2 * np.pi * (i % 24) / 24))),
            }
            for i in range(GRID_HOURS)
        ]
        conn.execute(
            text("INSERT INTO zone_hour (location_id, hour_utc, pickup_count) VALUES (:z, :h, :c)"),
            rows,
        )
    return engine


@pytest.fixture
def models(tmp_path: Path) -> Path:
    """Real fitted bundles, one per horizon."""
    from urbanflow.config import get_config

    config = get_config()
    rng = np.random.default_rng(1)
    n = 4000
    recent = rng.poisson(30.0, n).astype(float)
    target = rng.poisson(30.0, n).astype(float)
    columns = ["recent_1", "local_hour"]
    features = {"recent_1": recent, "local_hour": rng.integers(0, 24, n).astype(float)}

    directory = tmp_path / "models"
    for horizon in HORIZONS:
        model = fit_champion(features, target, columns, max_iter=30)
        ModelBundle(
            model=model,
            horizon_hours=horizon,
            feature_columns=columns,
            feature_cutoff_minutes=config.features.feature_cutoff_minutes,
            anchor_offset_hours=horizon + 2,
            dispersion=25.0,
            snapshot_id="yellow-test-000000000000",
            train_start=GRID_START.isoformat(),
            train_end=(GRID_START + timedelta(hours=1000)).isoformat(),
        ).save(directory)
    return directory


@pytest.fixture
def client(database, models: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from urbanflow.serving import app as app_module

    monkeypatch.setattr(app_module, "MODEL_DIR", models)
    monkeypatch.setattr(app_module, "make_engine", lambda *a, **k: database)
    with TestClient(app_module.app) as test_client:
        yield test_client


VALID_AS_OF = GRID_START + timedelta(hours=GRID_HOURS - 20)


def _forecast(client, **kwargs):  # type: ignore[no-untyped-def]
    params = {"zone_id": ZONE, "as_of": VALID_AS_OF.isoformat(), "horizon": 1}
    params.update(kwargs)
    return client.get("/v1/forecast", params=params)


# --- health and readiness ----------------------------------------------------


def test_health_does_not_touch_the_database(client) -> None:  # type: ignore[no-untyped-def]
    """Liveness must not fail because a dependency is slow."""
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_dependencies(client) -> None:  # type: ignore[no-untyped-def]
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert "3 loaded" in body["checks"]["models"]
    assert body["checks"]["database"] == "reachable"


# --- forecast ----------------------------------------------------------------


def test_forecast_returns_the_full_contract(client) -> None:  # type: ignore[no-untyped-def]
    body = _forecast(client).json()
    for key in (
        "zone_id",
        "as_of",
        "target_hour",
        "horizon_hours",
        "forecast",
        "lower_p05",
        "upper_p95",
        "model_version",
        "feature_cutoff_minutes",
        "anchor_hour",
        "generated_at",
    ):
        assert key in body, f"missing {key}"


def test_forecast_is_non_negative_and_bracketed(client) -> None:  # type: ignore[no-untyped-def]
    body = _forecast(client).json()
    assert body["forecast"] >= 0
    assert body["lower_p05"] <= body["upper_p95"]


def test_target_hour_matches_horizon(client) -> None:  # type: ignore[no-untyped-def]
    for horizon in HORIZONS:
        body = _forecast(client, horizon=horizon).json()
        as_of = datetime.fromisoformat(body["as_of"])
        target = datetime.fromisoformat(body["target_hour"])
        assert (target - as_of).total_seconds() / 3600 == horizon


def test_anchor_precedes_the_origin(client) -> None:  # type: ignore[no-untyped-def]
    """The availability rule, visible in the response (spec §2.1)."""
    for horizon in HORIZONS:
        body = _forecast(client, horizon=horizon).json()
        anchor = datetime.fromisoformat(body["anchor_hour"])
        as_of = datetime.fromisoformat(body["as_of"])
        assert anchor < as_of, f"anchor {anchor} is not before origin {as_of}"


def test_response_declares_the_cutoff(client) -> None:  # type: ignore[no-untyped-def]
    """A consumer must be able to see what latency the number assumed."""
    assert _forecast(client).json()["feature_cutoff_minutes"] == 15


def test_model_version_present(client) -> None:  # type: ignore[no-untyped-def]
    assert _forecast(client).json()["model_version"].startswith("h1-")


# --- error contracts ---------------------------------------------------------


def test_unknown_zone_returns_404(client) -> None:  # type: ignore[no-untyped-def]
    response = _forecast(client, zone_id=99999)
    assert response.status_code == 404
    assert "not forecastable" in response.json()["detail"]


def test_excluded_zone_returns_404(client) -> None:  # type: ignore[no-untyped-def]
    """A zone below the volume floor is knowable, not servable."""
    assert _forecast(client, zone_id=264).status_code == 404


def test_unsupported_horizon_returns_400(client) -> None:  # type: ignore[no-untyped-def]
    response = _forecast(client, horizon=12)
    assert response.status_code == 400
    assert "available horizons" in response.json()["detail"]


def test_origin_without_history_returns_422(client) -> None:  # type: ignore[no-untyped-def]
    """Better an explicit refusal than a forecast built on partial history."""
    response = _forecast(client, as_of=(GRID_START + timedelta(hours=5)).isoformat())
    assert response.status_code == 422
    assert "required hours" in response.json()["detail"]


def test_malformed_timestamp_returns_422(client) -> None:  # type: ignore[no-untyped-def]
    assert _forecast(client, as_of="not-a-date").status_code == 422


def test_missing_parameter_returns_422(client) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/v1/forecast", params={"zone_id": ZONE}).status_code == 422


# --- metadata endpoints ------------------------------------------------------


def test_zones_lists_only_forecastable(client) -> None:  # type: ignore[no-untyped-def]
    zones = client.get("/v1/zones").json()
    ids = {z["location_id"] for z in zones}
    assert ZONE in ids
    assert 264 not in ids, "an excluded zone must not be advertised"


def test_model_endpoint_exposes_provenance(client) -> None:  # type: ignore[no-untyped-def]
    body = client.get("/v1/model").json()
    assert set(body["horizons"]) == {str(h) for h in HORIZONS}
    first = body["horizons"]["1"]
    assert first["snapshot_id"] == "yellow-test-000000000000"
    assert first["feature_cutoff_minutes"] == 15
    assert "dispersion" in first


def test_openapi_schema_generated(client) -> None:  # type: ignore[no-untyped-def]
    schema = client.get("/openapi.json").json()
    assert "/v1/forecast" in schema["paths"]
