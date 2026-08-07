# UrbanFlow ML

Production taxi-demand forecasting and decision API over NYC TLC yellow-taxi trip records.

Predicts hourly pickup demand per taxi zone at +1h, +3h and +6h horizons, with measured
prediction intervals and a benchmark the model has to actually beat.

**Status:** Phase 1 in progress — ingestion, provenance and validation complete.
No model results yet; this README will carry measured numbers only once they are
reproducibly captured in the repository.

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

**Forecastable zones.** Not every TLC LocationID is a zone a model should
predict. Six are excluded structurally (264 and 265 are "unknown" and "outside
NYC" placeholders; Newark Airport sits outside yellow-cab pickup jurisdiction at
~16 pickups/day; the three island IDs recorded 0, 0 and 4 pickups in all of
2024). A further volume floor of 0.1 pickups/hour removes 43 zones that together
carry 0.02% of trips — one recorded a single pickup across the year. Sparsity
above that floor is handled by tiering rather than deletion.

**Demand tiers are defined on mean, not median.** The spec originally used median
hourly pickups; measured against real data that is unworkable, because 152 of 259
zones average under 1 pickup/hour and their median hour is exactly 0. Tiers are
therefore computed on the mean: high (>=20/hr, 47 zones), medium (2-20/hr, 31),
low (0.1-2/hr, 138). High and medium carry 98.17% of trips and are what the
acceptance gates measure; low tier is reported but not gated.

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

# Check every file against the data contract
uv run urbanflow contract-check

# Validate, quarantine and write the quality report
uv run urbanflow validate

# Profile zone volumes and write the forecastable-zone registry
uv run urbanflow zone-report

# Run the test suite
uv run pytest
```

Ingestion is idempotent — re-running skips files already present and readable.
Roughly 1.5 GB for the default 30-month window.

---

## Schema eras

The 2023-01 to 2025-06 window spans two TLC schema revisions:

| Boundary | Change | Impact |
|---|---|---|
| 2023-01 -> 2023-02 | `PULocationID` int64 -> int32, `passenger_count` double -> int64, `airport_fee` -> `Airport_fee` | Real migration; handled by explicit cast to the canonical contract type |
| 2024-12 -> 2025-01 | `cbd_congestion_fee` added (congestion pricing) | Benign; column is not consumed |

The ingest manifest compares whole-schema fingerprints and reports this window as
inconsistent, which is correct but too blunt to act on. The data contract in
`data_contracts/` instead asserts that the four consumed columns exist with a
compatible type *family*, so an added fee column or a widened integer does not
raise a false alarm while a retyped zone ID would.

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
