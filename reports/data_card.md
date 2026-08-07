# Data card: NYC TLC yellow taxi corpus

| Field | Value |
|---|---|
| Source | New York City Taxi and Limousine Commission trip records |
| Access | Public monthly Parquet files, no authentication |
| Snapshot | `yellow-2023-01-2025-06-26c6cf6a154f` |
| Window | January 2023 through June 2025, 30 monthly files |
| Raw records | 103,563,330 |
| Download size | Approximately 1.5 GB |
| Licence | Public data published by a municipal agency |

---

## 1. Provenance

Every experiment references a snapshot identifier derived from the SHA-256
checksums of the files it covers, combined with the service and window. Two
people downloading the same window produce the same identifier. A silently
revised upstream file produces a different one.

Manifests record, per file: source URL, SHA-256, byte size, row count, schema
fingerprint and retrieval timestamp. They are committed to version control.

```bash
uv run urbanflow manifest-show yellow-2023-01-2025-06-26c6cf6a154f
```

The Commission states that submitted trip data may contain accuracy or
completeness issues, and revises published files over time. The manifest exists
so that a result can be traced to specific bytes rather than to a filename.

---

## 2. Window selection

| Hazard | Effect | Response |
|---|---|---|
| COVID period | Volumes fell roughly 90%, hour-of-week shape changed | Exclude 2020 and 2021 |
| Secular decline | Yellow taxi share falling against for-hire vehicles | Start 2023 so levels transfer |

More data is not better when it spans a structural break. A model trained across
one learns an average of two regimes and represents neither.

---

## 3. Schema changes

The 30 files do not share a schema. Three eras exist within the window.

| Period | Schema characteristics |
|---|---|
| 2023-01 | `PULocationID` int64, `passenger_count` double, column named `airport_fee` |
| 2023-02 to 2024-12 | `PULocationID` int32, `passenger_count` int64, column named `Airport_fee` |
| 2025-01 onward | As above, plus `cbd_congestion_fee` added when congestion pricing began |

Only the first boundary retypes a consumed column. The second adds an unused fee
column.

Handling is explicit: the data contract requires consumed columns to match by
**type family** rather than exact type, so an integer widening passes while a
zone identifier shipped as text would fail. Casting to canonical types happens in
the validation SQL so the migration is visible in code.

---

## 4. Fields consumed

| Column | Type family | Role | Nullable |
|---|---|---|---|
| `tpep_pickup_datetime` | timestamp | Time index and target source | No |
| `PULocationID` | integer | Prediction unit | No |
| `tpep_dropoff_datetime` | timestamp | Quality control only | Yes |
| `VendorID` | integer | Provenance and duplicate signature | Yes |

### Prohibited fields

`DOLocationID`, `fare_amount`, `tip_amount`, `tolls_amount`, `total_amount`,
`payment_type`, `extra`, `mta_tax`, `improvement_surcharge`,
`congestion_surcharge`, `airport_fee`, `Airport_fee`, `cbd_congestion_fee`,
`trip_distance`, `RatecodeID`, `store_and_fwd_flag`.

All are known only after a trip completes. `trip_distance` and `total_amount` are
read during validation for duplicate detection and never reach a model, which is
asserted by test.

---

## 5. Validation results

| Reason code | Rows | Share |
|---|---|---|
| `ZONE_EXCLUDED` | 601,691 | 0.5810% |
| `NEGATIVE_DURATION` | 3,011 | 0.0029% |
| `TS_OUT_OF_MONTH` | 1,258 | 0.0012% |
| `IMPLAUSIBLE_DURATION` | 525 | 0.0005% |
| `DUPLICATE` | 10 | 0.00001% |
| `NULL_PICKUP_TS` | 0 | |
| `NULL_ZONE` | 0 | |
| `ZONE_OUT_OF_RANGE` | 0 | |
| `DST_NONEXISTENT` | 0 | |
| **Total quarantined** | **606,495** | **0.5856%** |
| **Passed** | **102,956,835** | **99.4144%** |

Quarantined rows are written with their reason code attached, never dropped. The
pipeline asserts `rows_out + sum(reason_counts) == rows_in` per month and exits
non-zero otherwise.

### Notes on individual codes

**`TS_OUT_OF_MONTH`** confirms that published files genuinely contain rows
timestamped years from the month they claim to cover.

**`DUPLICATE` at 10 rows** is the important figure. The intuitive signature for
this dataset, matching on pickup time, zone and vendor, removes about 3 million
rows, and almost all of them are real trips. Midtown sees a pickup roughly every
18 seconds, so two cabs from one vendor collecting passengers in the same zone
during the same second is routine. The signature used is a full row match
including distance and total amount.

**`DST_NONEXISTENT` at zero** is expected. Meters record real elapsed time, so
timestamps in the spring-forward gap should not occur. The check exists because
without it the database silently shifts such times forward and they look valid
permanently.

---

## 6. Zone selection

265 location identifiers exist. 259 have any observed pickups. 217 are
forecastable.

### Structural exclusions

| ID | Zone | 2024 pickups | Reason |
|---|---|---|---|
| 264 | Unknown | 122,200 | Placeholder, not a location |
| 265 | Outside of NYC | 24,001 | Placeholder, not a location |
| 1 | Newark Airport | 5,940 | Outside yellow cab pickup jurisdiction |
| 103 | Governor's Island group | 0 | No pickups |
| 104 | Governor's Island group | 0 | No pickups |
| 105 | Governor's Island group | 4 | Four pickups in a year |

### Volume floor

| Floor (pickups per hour) | Zones retained | Trips retained |
|---|---|---|
| 0 | 259 | 100.00% |
| **0.1 (chosen)** | **216** | **99.98%** |
| 0.5 | 137 | 99.50% |
| 1.0 | 107 | 99.06% |
| 2.0 | 78 | 98.17% |

Full evidence per zone is in `reports/zone_registry.json`.

---

## 7. Derived corpus

### Dense zone-hour grid

| Property | Value |
|---|---|
| Zones | 217 |
| Hours | 21,887 |
| Rows | 4,749,479 |
| Zero rows | 2,180,828 (45.92%) |
| Total pickups | 102,929,915 |
| Span | 2023-01-01 05:00 UTC to 2025-07-01 03:00 UTC |

Trip records contain no row for an hour with no pickups. The grid materialises
those hours as genuine zeros through a cross join, because absence of a trip
record is evidence of zero demand rather than absence of evidence. Density is a
release blocker with zero tolerance.

### City-wide zero hours

| Hour | Classification |
|---|---|
| 2023-11-05 05:00 UTC | Daylight saving fold artifact |
| 2024-11-03 05:00 UTC | Daylight saving fold artifact |

**Zero ingestion gaps.** An hour where every zone reports zero is never a quiet
night in New York. The fold artifact arises because ambiguous local times all
resolve to one of the two UTC hours the folded local hour spans, leaving the
other empty by conversion rather than by the city.

### Feature tables

| Property | Value |
|---|---|
| Rows per horizon | 4,603,655 |
| Features | 19 |
| Warm-up trimmed | 672 hours per zone (145,824 rows) |

---

## 8. Known biases and limitations

| Issue | Detail |
|---|---|
| Yellow taxis only | For-hire vehicles carry the majority of modern trips and are excluded |
| Declining mode share | Absolute demand levels shift over the window |
| Geographic concentration | 47 of 217 zones carry most volume; the long tail is sparse |
| Excluded zones | 43 zones have no forecast, carrying 0.02% of trips |
| Source revisions | The Commission revises published files; a snapshot identifier change signals this |
| Reported quality issues | The Commission states submitted data may be inaccurate or incomplete |
| No weather or events | Both plausibly explain part of the unexplained variance |
| Manhattan orientation | The dataset reflects where yellow taxis operate, not where demand exists citywide |

The final row matters for interpretation. This corpus describes yellow taxi
pickups, which is a biased sample of mobility demand, concentrated where yellow
taxis are permitted and economical to operate. It should not be read as a measure
of transport need.

---

## 9. Reproducing this corpus

```bash
uv run urbanflow ingest           # 15 to 40 minutes
uv run urbanflow contract-check   # seconds
uv run urbanflow validate         # 5 to 15 minutes
uv run urbanflow zone-report      # under a minute
uv run urbanflow build-grid       # 1 to 3 minutes
uv run urbanflow build-features   # 1 to 2 minutes
```

Every stage is idempotent and writes an audit artefact. If the snapshot
identifier differs from the one recorded here, the upstream files have been
revised and figures in this document may no longer match.
