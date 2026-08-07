"""Feature construction at serving time.

The training builder computes features with window functions across the whole
grid; serving must produce the identical vector for a single zone and hour from
a point lookup. Two implementations of one definition is how a model silently
degrades in production — the classic training/serving skew — so both paths draw
their offsets from :class:`~urbanflow.features.build.FeatureSpec` rather than
restating them, and ``tests/unit/test_serving_features.py`` asserts the two
agree numerically on real rows.

The history array passed in is indexed by hours *before the target*: index 0 is
the target hour itself (never read), index k is k hours earlier. That mirrors the
``LAG(pickup_count, k)`` calls in the training SQL, so the two definitions can be
compared by eye as well as by test.
"""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from urbanflow.features.build import HOW_MEDIAN_WEEKS, FeatureSpec


#: Hours of history a forecast needs. Four weeks for the trailing median, plus
#: whatever the longest rolling window reaches back to.
def required_history_hours(spec: FeatureSpec) -> int:
    return max(spec.offset + 168, HOW_MEDIAN_WEEKS * 168) + 1


def build_feature_vector(
    history: np.ndarray,
    target_hour: datetime,
    spec: FeatureSpec,
    timezone: str = "America/New_York",
) -> dict[str, float]:
    """Compute one feature row.

    ``history[k]`` is the pickup count k hours before ``target_hour``. Index 0 is
    the target itself and is never read — reading it would be the leak the whole
    availability rule exists to prevent.
    """
    needed = required_history_hours(spec)
    if len(history) < needed:
        raise ValueError(
            f"need {needed} hours of history for a {spec.horizon_hours}h forecast, "
            f"got {len(history)}"
        )

    off = spec.offset
    local = target_hour.astimezone(ZoneInfo(timezone))

    def window_mean(start: int, length: int) -> float:
        """Mean over `length` hours ending at `start` hours before target."""
        return float(np.mean(history[start : start + length]))

    week_lags = [history[168 * w] for w in range(1, HOW_MEDIAN_WEEKS + 1)]

    features = {
        # Recent history, anchored on availability.
        "recent_1": float(history[off]),
        "recent_2": float(history[off + 1]),
        "recent_3": float(history[off + 2]),
        # Seasonal history, anchored on the target hour.
        "lag_24h": float(history[24]),
        "lag_48h": float(history[48]),
        "lag_168h": float(history[168]),
        # Rolling aggregates over windows ending at the anchor.
        "roll_mean_3h": window_mean(off, 3),
        "roll_mean_24h": window_mean(off, 24),
        "roll_mean_168h": window_mean(off, 168),
        "zero_rate_168h": float(np.mean(history[off : off + 168] == 0)),
        "how_median_4w": float(np.median(week_lags)),
        # Calendar features, derived in local time from a UTC index.
        "local_hour": float(local.hour),
        "local_dow": float((local.weekday() + 1) % 7),  # match SQL DOW: Sunday = 0
        "local_month": float(local.month),
        "is_weekend": 1.0 if (local.weekday() + 1) % 7 in (0, 6) else 0.0,
        "hour_sin": math.sin(2 * math.pi * local.hour / 24),
        "hour_cos": math.cos(2 * math.pi * local.hour / 24),
        "dow_sin": math.sin(2 * math.pi * ((local.weekday() + 1) % 7) / 7),
        "dow_cos": math.cos(2 * math.pi * ((local.weekday() + 1) % 7) / 7),
    }
    return features


def to_arrays(features: dict[str, float], columns: list[str]) -> dict[str, np.ndarray]:
    """Shape a single feature row for the model's batch interface."""
    missing = [c for c in columns if c not in features]
    if missing:
        raise KeyError(f"serving features are missing columns the model expects: {missing}")
    return {c: np.array([features[c]], dtype=np.float64) for c in columns}
