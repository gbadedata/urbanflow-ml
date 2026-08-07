-- UrbanFlow ML — PostgreSQL schema
--
-- The grid is the canonical analytical table: one row per forecastable zone per
-- UTC hour, with zero-demand hours materialised as genuine zeros rather than
-- absent rows (spec §2.2).
--
-- Timestamps are stored as TIMESTAMPTZ in UTC. Local calendar features are
-- derived at feature-build time, not stored, so that a DST transition changes
-- the derived column and never the grid's row count.

CREATE TABLE IF NOT EXISTS taxi_zones (
    location_id      INTEGER PRIMARY KEY,
    borough          TEXT,
    zone_name        TEXT,
    service_zone     TEXT,
    is_forecastable  BOOLEAN     NOT NULL,
    exclusion_reason TEXT,
    demand_tier      TEXT,
    mean_per_hour    DOUBLE PRECISION,
    trips_observed   BIGINT,
    CONSTRAINT taxi_zones_tier_valid
        CHECK (demand_tier IN ('high', 'medium', 'low', 'excluded')),
    -- A forecastable zone must have a tier; an excluded one must have a reason.
    CONSTRAINT taxi_zones_decision_documented
        CHECK (
            (is_forecastable AND demand_tier <> 'excluded' AND exclusion_reason IS NULL)
            OR (NOT is_forecastable AND exclusion_reason IS NOT NULL)
        )
);

COMMENT ON TABLE taxi_zones IS
    'TLC zone registry with the forecastability decision and its evidence.';
COMMENT ON COLUMN taxi_zones.exclusion_reason IS
    'structural | below_volume_floor — NULL when the zone is forecastable.';

CREATE TABLE IF NOT EXISTS zone_hour (
    location_id   INTEGER     NOT NULL REFERENCES taxi_zones (location_id),
    hour_utc      TIMESTAMPTZ NOT NULL,
    pickup_count  INTEGER     NOT NULL,
    PRIMARY KEY (location_id, hour_utc),
    CONSTRAINT zone_hour_count_non_negative CHECK (pickup_count >= 0),
    -- The grid is hourly; a timestamp with a non-zero minute or second means the
    -- aggregation truncated wrongly.
    CONSTRAINT zone_hour_aligned
        CHECK (date_trunc('hour', hour_utc) = hour_utc)
);

COMMENT ON TABLE zone_hour IS
    'Dense grid: one row per forecastable zone per UTC hour. Zeros are real data.';

CREATE INDEX IF NOT EXISTS zone_hour_hour_idx ON zone_hour (hour_utc);
CREATE INDEX IF NOT EXISTS zone_hour_zone_idx ON zone_hour (location_id);

-- Hours excluded from training and evaluation because the source itself failed.
-- A city-wide all-zero hour is an ingestion gap, not a quiet night; conflating
-- the two teaches the model that demand sometimes vanishes everywhere at once.
CREATE TABLE IF NOT EXISTS excluded_hours (
    hour_utc   TIMESTAMPTZ PRIMARY KEY,
    reason     TEXT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE excluded_hours IS
    'Hours where the pipeline, not the city, produced zero demand.';

CREATE TABLE IF NOT EXISTS model_runs (
    run_id          TEXT PRIMARY KEY,
    snapshot_id     TEXT        NOT NULL,
    model_name      TEXT        NOT NULL,
    horizon_hours   INTEGER     NOT NULL,
    trained_at      TIMESTAMPTZ NOT NULL,
    train_start     TIMESTAMPTZ NOT NULL,
    train_end       TIMESTAMPTZ NOT NULL,
    feature_cutoff_minutes INTEGER NOT NULL,
    metrics         JSONB
);

COMMENT ON COLUMN model_runs.feature_cutoff_minutes IS
    'Recorded per run: a model trained under a different availability assumption '
    'is not comparable to one trained under this spec (§2.1).';

CREATE TABLE IF NOT EXISTS forecasts (
    run_id        TEXT        NOT NULL REFERENCES model_runs (run_id),
    location_id   INTEGER     NOT NULL REFERENCES taxi_zones (location_id),
    as_of         TIMESTAMPTZ NOT NULL,
    target_hour   TIMESTAMPTZ NOT NULL,
    horizon_hours INTEGER     NOT NULL,
    prediction    DOUBLE PRECISION NOT NULL,
    lower_p05     DOUBLE PRECISION,
    upper_p95     DOUBLE PRECISION,
    generated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, location_id, as_of, horizon_hours),
    CONSTRAINT forecasts_interval_ordered
        CHECK (lower_p05 IS NULL OR upper_p95 IS NULL OR lower_p05 <= upper_p95),
    CONSTRAINT forecasts_horizon_consistent
        CHECK (target_hour = as_of + (horizon_hours * INTERVAL '1 hour'))
);

CREATE INDEX IF NOT EXISTS forecasts_target_idx ON forecasts (target_hour);
