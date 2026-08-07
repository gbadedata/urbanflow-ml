"""FastAPI serving layer.

Endpoints return the model version and the availability cutoff alongside every
forecast, so a consumer can tell which model produced a number and under what
assumption. Errors are deterministic 4xx with actionable messages rather than a
500 and a stack trace: an unknown zone, an unsupported horizon, and an `as_of`
without enough history are all normal conditions, not faults.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from urbanflow.config import PROJECT_ROOT, get_config
from urbanflow.features.build import FeatureSpec
from urbanflow.models.intervals import interval_bounds
from urbanflow.models.persist import ModelBundle, load_all
from urbanflow.serving.features import (
    build_feature_vector,
    required_history_hours,
    to_arrays,
)
from urbanflow.serving.repository import DemandRepository, make_engine

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))


class ForecastResponse(BaseModel):
    zone_id: int
    as_of: datetime
    target_hour: datetime
    horizon_hours: int
    forecast: float = Field(description="Expected pickups in the target hour.")
    lower_p05: float
    upper_p95: float
    model_version: str
    feature_cutoff_minutes: int = Field(
        description="Feed latency assumed. Features use no data newer than as_of minus this."
    )
    anchor_hour: datetime = Field(
        description="Most recent hour whose demand informed this forecast."
    )
    generated_at: datetime


class ZoneResponse(BaseModel):
    location_id: int
    zone_name: str | None
    borough: str | None
    demand_tier: str | None


class ModelResponse(BaseModel):
    horizons: dict[int, dict[str, Any]]
    zones_forecastable: int
    latest_grid_hour: datetime | None


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, str]


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and open the pool once, not per request."""
    config = get_config()
    state["config"] = config
    state["engine"] = make_engine()
    state["repository"] = DemandRepository(state["engine"])

    bundles = load_all(MODEL_DIR)
    for bundle in bundles.values():
        # Refuse a model trained under a different availability assumption
        # rather than serving predictions whose features mean something else.
        bundle.check_compatible(config.features.feature_cutoff_minutes)
    state["models"] = bundles
    logger.info("loaded %d champion models from %s", len(bundles), MODEL_DIR)
    yield
    state.clear()


app = FastAPI(
    title="UrbanFlow ML",
    version="0.1.0",
    description="Hourly taxi-demand forecasts by NYC zone, with measured intervals.",
    lifespan=lifespan,
)


def _bundle(horizon: int) -> ModelBundle:
    models: dict[int, ModelBundle] = state.get("models", {})
    if horizon not in models:
        raise HTTPException(
            status_code=400,
            detail=(f"horizon {horizon}h is not served; available horizons are {sorted(models)}"),
        )
    return models[horizon]


@app.get("/v1/forecast", response_model=ForecastResponse)
def forecast(
    zone_id: int = Query(description="TLC taxi zone LocationID."),
    as_of: datetime = Query(description="Forecast origin, UTC."),
    horizon: int = Query(description="Hours ahead: 1, 3 or 6."),
) -> ForecastResponse:
    config = state["config"]
    repository: DemandRepository = state["repository"]
    bundle = _bundle(horizon)

    if not repository.zone_exists(zone_id):
        raise HTTPException(
            status_code=404,
            detail=(
                f"zone {zone_id} is not forecastable; it is either unknown or below "
                f"the volume floor. See /v1/zones."
            ),
        )

    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)
    target_hour = as_of.replace(minute=0, second=0, microsecond=0)
    from datetime import timedelta

    target_hour = target_hour + timedelta(hours=horizon)

    spec = FeatureSpec(horizon, bundle.feature_cutoff_minutes)
    hours = required_history_hours(spec)
    try:
        history = repository.history(zone_id, target_hour, hours)
    except LookupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    features = build_feature_vector(history, target_hour, spec, config.features.timezone)
    arrays = to_arrays(features, bundle.feature_columns)

    mu = float(bundle.model.predict(arrays)[0])
    lower, upper = interval_bounds(np.array([mu]), bundle.dispersion)

    return ForecastResponse(
        zone_id=zone_id,
        as_of=as_of,
        target_hour=target_hour,
        horizon_hours=horizon,
        forecast=round(mu, 3),
        lower_p05=float(lower[0]),
        upper_p95=float(upper[0]),
        model_version=bundle.model_version,
        feature_cutoff_minutes=bundle.feature_cutoff_minutes,
        anchor_hour=target_hour - timedelta(hours=spec.offset),
        generated_at=datetime.now(UTC),
    )


@app.get("/v1/zones", response_model=list[ZoneResponse])
def zones() -> list[ZoneResponse]:
    repository: DemandRepository = state["repository"]
    return [
        ZoneResponse(
            location_id=z.location_id,
            zone_name=z.zone_name,
            borough=z.borough,
            demand_tier=z.demand_tier,
        )
        for z in repository.forecastable_zones()
    ]


@app.get("/v1/model", response_model=ModelResponse)
def model_info() -> ModelResponse:
    repository: DemandRepository = state["repository"]
    models: dict[int, ModelBundle] = state.get("models", {})
    return ModelResponse(
        horizons={h: b.metadata() for h, b in models.items()},
        zones_forecastable=len(repository.forecastable_zones()),
        latest_grid_hour=repository.latest_hour(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Process liveness only — deliberately does not touch the database."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Dependency readiness: models loaded, database reachable, grid populated."""
    checks: dict[str, str] = {}

    models = state.get("models", {})
    checks["models"] = f"{len(models)} loaded" if models else "none loaded"

    try:
        repository: DemandRepository = state["repository"]
        latest = repository.latest_hour()
        checks["database"] = "reachable"
        checks["grid"] = f"latest hour {latest.isoformat()}" if latest else "empty"
    except Exception as exc:  # noqa: BLE001 - readiness must report, not raise
        checks["database"] = f"unreachable: {type(exc).__name__}"
        checks["grid"] = "unknown"

    ok = bool(models) and checks["database"] == "reachable" and "latest hour" in checks["grid"]
    return ReadyResponse(ready=ok, checks=checks)
