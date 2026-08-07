# UrbanFlow ML

Production taxi-demand forecasting and decision API over NYC TLC yellow-taxi trip records.

Predicts hourly pickup demand per taxi zone at +1h, +3h and +6h horizons, with measured
prediction intervals and a benchmark the model has to actually beat.

**Status:** Phase 1 in progress — ingestion and provenance complete. No results yet;
this README will carry measured numbers only once they are reproducibly captured
in the repository.

---

## Stated assumptions

These are the assumptions the whole build rests on. They are stated here because
leaving them implicit is the difference between a forecasting system and a
backtest that could never be deployed.

**Feature availability.** TLC publishes monthly, months in arrears, so a lag-1h
feature could never exist at prediction time in a real deployment. This project
assumes an operator-internal trip feed with **15 minutes of latency**, simulated
from historical TLC data. No feature may use data newer than
`as_of - FEATURE_CUTOFF_MINUTES`; this is enforced in the feature builder and
covered by a unit test, not left as a convention.

**Data window.** January 2023 onward. Earlier data is excluded deliberately: the
2020–21 period is a structural break (volumes fell sharply and the hour-of-week
shape changed), and yellow-taxi share has declined secularly against for-hire
vehicles, so older absolute levels do not transfer.

**Dense zone-hour grid.** Zero-pickup hours produce no rows in trip data. The
canonical table is a full cross-join of forecastable zones × hours with missing
counts materialised as genuine zeros, because a naive `GROUP BY` silently drops
them and biases every lag and metric downstream — worst in exactly the
low-volume zones where the model is weakest.

---

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
# Install dependencies
uv sync --all-groups

# Inspect the resolved configuration
uv run urbanflow config-show

# Start PostgreSQL
cp .env.example .env
docker compose up -d

# Download the configured window and write a snapshot manifest
uv run urbanflow ingest

# Run the test suite
uv run pytest
```

Ingestion is idempotent — re-running skips files already present and readable.
Roughly 1.5 GB for the default 30-month window.

---

## Provenance

Every experiment references a **snapshot ID**, derived from the SHA-256 checksums
of the files it covers. Two people who download the same window get the same ID;
a silently revised upstream file produces a different one. Manifests live in
`data/manifests/` and are committed. Raw Parquet is not.

```bash
uv run urbanflow manifest-show <snapshot_id>
```

---

## Repository layout

```
configs/            # locked project decisions (window, zones, cutoff, horizons)
data_contracts/     # schema and validation contracts
src/urbanflow/
  ingest/           # source discovery, download, snapshot manifests
  validation/       # data contract enforcement, quarantine
  features/         # dense grid, lags, calendar — availability cutoff enforced here
  models/           # baselines and champion
  serving/          # FastAPI application
  monitoring/       # drift and realised error
sql/                # grid construction and analytical queries
tests/{unit,integration,contract}/
reports/            # data quality, model card, evaluation
```

---

## Model plan

| Tier | Model | Role |
|------|-------|------|
| A | Seasonal naive (t−168h) | Sanity floor. Reported, never the gate. |
| B | Hour-of-week trailing median, 4-week window | **The benchmark the champion must beat.** |
| C | `PoissonRegressor` | Interpretable statistical baseline |
| Champion | `HistGradientBoostingRegressor(loss="poisson")` | Must beat B by ≥10% MAE or B ships |

Baseline B is deliberately strong. A benchmark that cannot be failed teaches
nothing, and if the champion does not clear it, that result gets written up
rather than buried.

Metrics are reported by demand tier rather than uniformly — percentage errors are
meaningless where median hourly demand is below 2 pickups, so those zones are
evaluated on Poisson deviance and interval coverage instead.

---

## Licence

MIT
