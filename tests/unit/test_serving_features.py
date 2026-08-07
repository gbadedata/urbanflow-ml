"""Training/serving feature parity.

The single most valuable test in the serving layer. Training computes features
with SQL window functions over the full grid; serving computes them in numpy from
a point lookup. If those two drift apart, the model receives inputs that mean
something different from what it learned, and nothing about the system looks
broken — predictions are still produced, the API still returns 200, and accuracy
quietly decays.

So rather than testing the serving path against hand-written expectations, these
tests run the real training builder over a synthetic grid and assert the serving
path reproduces its output exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from urbanflow.config import DEFAULT_CONFIG_PATH, Config, load_config
from urbanflow.duck import connect
from urbanflow.features.build import FeatureSpec, build_features
from urbanflow.serving.features import (
    build_feature_vector,
    required_history_hours,
    to_arrays,
)

#: Columns both paths must agree on, to the last decimal.
COMPARED = [
    "recent_1",
    "recent_2",
    "recent_3",
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "roll_mean_3h",
    "roll_mean_24h",
    "roll_mean_168h",
    "zero_rate_168h",
    "how_median_4w",
    "local_hour",
    "local_dow",
    "local_month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    for key in ("raw_dir", "validated_dir", "quarantine_dir"):
        raw["data"][key] = str(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture
def grid_and_features(config: Config) -> tuple[np.ndarray, list[dict[str, float]], datetime]:
    """A synthetic grid carried through the real training builder.

    Demand is irregular on purpose: a smooth pattern would let two different
    definitions agree by coincidence.
    """
    con = connect()
    con.execute("""
        CREATE TEMP TABLE zone_hour AS
        WITH h AS (
            SELECT UNNEST(generate_series(
                TIMESTAMPTZ '2024-01-01 00:00:00+00',
                TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL '2999 hour',
                INTERVAL '1 hour')) AS hour_utc
        )
        SELECT 161 AS location_id, hour_utc,
               CAST(ABS(HASH(hour_utc)) % 40 AS INTEGER) AS pickup_count
        FROM h
    """)
    rows = con.execute("""
        SELECT hour_utc, pickup_count FROM zone_hour ORDER BY hour_utc
    """).fetchall()
    counts = np.array([r[1] for r in rows], dtype=np.float64)
    start = rows[0][0]
    return counts, [], start


def _training_row(
    config: Config, horizon: int, target_hour: datetime
) -> tuple[dict[str, float], np.ndarray, datetime]:
    """Run the training builder and return one row plus the raw grid."""
    con = connect()
    con.execute("""
        CREATE TEMP TABLE zone_hour AS
        WITH h AS (
            SELECT UNNEST(generate_series(
                TIMESTAMPTZ '2024-01-01 00:00:00+00',
                TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL '2999 hour',
                INTERVAL '1 hour')) AS hour_utc
        )
        SELECT 161 AS location_id, hour_utc,
               CAST(ABS(HASH(hour_utc)) % 40 AS INTEGER) AS pickup_count
        FROM h
    """)
    table = build_features(con, config, horizon_hours=horizon)
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
    row = con.execute(
        f"SELECT * FROM {table} WHERE target_hour = ? LIMIT 1", [target_hour]
    ).fetchone()
    assert row is not None, f"no training row at {target_hour}"
    training = dict(zip(cols, row, strict=True))

    grid = con.execute("SELECT hour_utc, pickup_count FROM zone_hour ORDER BY hour_utc").fetchall()
    counts = np.array([g[1] for g in grid], dtype=np.float64)
    return training, counts, grid[0][0]


def _history_for(counts: np.ndarray, grid_start: datetime, target: datetime) -> np.ndarray:
    """Slice history so index k is k hours before the target."""
    index = int((target - grid_start).total_seconds() // 3600)
    return counts[: index + 1][::-1]


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_serving_matches_training_exactly(config: Config, horizon: int) -> None:
    """The parity assertion. Any divergence here is training/serving skew."""
    target = datetime(2024, 3, 20, 14, 0, tzinfo=UTC)
    training, counts, grid_start = _training_row(config, horizon, target)
    history = _history_for(counts, grid_start, target)

    spec = FeatureSpec(horizon, config.features.feature_cutoff_minutes)
    serving = build_feature_vector(history, target, spec, config.features.timezone)

    for column in COMPARED:
        assert serving[column] == pytest.approx(float(training[column]), rel=1e-9), (
            f"{column} differs at horizon {horizon}h: "
            f"serving={serving[column]} training={training[column]}"
        )


@pytest.mark.parametrize("hour", [0, 6, 14, 23])
def test_parity_holds_across_the_day(config: Config, hour: int) -> None:
    """Calendar features are derived differently in each path; check several."""
    target = datetime(2024, 4, 20, hour, 0, tzinfo=UTC)
    training, counts, grid_start = _training_row(config, 1, target)
    history = _history_for(counts, grid_start, target)

    spec = FeatureSpec(1, config.features.feature_cutoff_minutes)
    serving = build_feature_vector(history, target, spec, config.features.timezone)

    for column in ("local_hour", "local_dow", "is_weekend", "hour_sin", "dow_cos"):
        assert serving[column] == pytest.approx(float(training[column]), rel=1e-9), column


def test_parity_holds_across_a_dst_boundary(config: Config) -> None:
    """Local calendar features shift at the transition; both paths must shift alike."""
    target = datetime(2024, 3, 10, 12, 0, tzinfo=UTC)
    training, counts, grid_start = _training_row(config, 1, target)
    history = _history_for(counts, grid_start, target)

    spec = FeatureSpec(1, config.features.feature_cutoff_minutes)
    serving = build_feature_vector(history, target, spec, config.features.timezone)
    assert serving["local_hour"] == pytest.approx(float(training["local_hour"]))


# --- serving-path specifics --------------------------------------------------


def test_target_hour_is_never_read() -> None:
    """Index 0 is the target. Reading it is the leak the cutoff exists to stop."""
    spec = FeatureSpec(1, 15)
    history = np.ones(required_history_hours(spec) + 10)
    history[0] = 999_999.0

    features = build_feature_vector(history, datetime(2024, 6, 1, 12, tzinfo=UTC), spec)
    assert all(v < 1000 for k, v in features.items() if k.startswith(("recent", "lag", "roll")))


def test_insufficient_history_raises() -> None:
    """Better a clear error than a forecast built on partial history."""
    spec = FeatureSpec(1, 15)
    with pytest.raises(ValueError, match="need .* hours of history"):
        build_feature_vector(np.ones(100), datetime(2024, 6, 1, tzinfo=UTC), spec)


def test_required_history_covers_four_weeks() -> None:
    assert required_history_hours(FeatureSpec(1, 15)) >= 4 * 168


def test_longer_horizons_need_no_less_history() -> None:
    assert required_history_hours(FeatureSpec(6, 15)) >= required_history_hours(FeatureSpec(1, 15))


def test_to_arrays_shapes_for_the_model() -> None:
    features = {"a": 1.0, "b": 2.0}
    arrays = to_arrays(features, ["a", "b"])
    assert arrays["a"].shape == (1,)
    assert arrays["a"][0] == 1.0


def test_to_arrays_rejects_missing_columns() -> None:
    """A silently absent feature would shift every column after it."""
    with pytest.raises(KeyError, match="missing columns"):
        to_arrays({"a": 1.0}, ["a", "b"])
