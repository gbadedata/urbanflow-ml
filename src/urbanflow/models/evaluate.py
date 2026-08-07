"""Baseline evaluation.

Fits Baseline C on the train block and evaluates all three baselines on a chosen
block. The default is the calibrate block: the test block is frozen and read
once, at release (spec §6.2), so reading it requires an explicit request.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import numpy as np

from urbanflow.config import Config
from urbanflow.models.baselines import (
    FittedBaseline,
    fit_poisson,
    hour_of_week_median,
    seasonal_naive,
)
from urbanflow.models.champion import fit_champion
from urbanflow.models.metrics import TierMetrics, evaluate_by_tier
from urbanflow.models.split import CALIBRATE, TRAIN

logger = logging.getLogger(__name__)


def load_zone_tiers(registry_path: Path) -> dict[int, str]:
    """Map zone id to demand tier from the committed registry."""
    registry = json.loads(registry_path.read_text())
    return {z["zone_id"]: z["tier"] for z in registry["zones"] if z["tier"] != "excluded"}


def load_block(
    con: duckdb.DuckDBPyConnection,
    table: str,
    block: str,
    columns: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load one block as feature arrays, target and zone ids.

    Columns are selected explicitly rather than with ``SELECT *`` so that adding
    a column to the feature table cannot silently change what a model consumes.
    """
    selected = ", ".join(["location_id", "target", *columns])
    frame = con.execute(
        f"SELECT {selected} FROM {table} WHERE block = ? ORDER BY target_hour, location_id",
        [block],
    ).df()

    features = {c: frame[c].to_numpy(dtype=np.float64) for c in columns}
    target = frame["target"].to_numpy(dtype=np.float64)
    zones = frame["location_id"].to_numpy(dtype=np.int64)
    return features, target, zones


def evaluate_baselines(
    con: duckdb.DuckDBPyConnection,
    config: Config,
    table: str,
    feature_cols: list[str],
    zone_tiers: dict[int, str],
    eval_block: str = CALIBRATE,
    with_champion: bool = False,
    max_iter: int = 400,
) -> dict[str, list[TierMetrics]]:
    """Fit on train, evaluate every baseline on ``eval_block``."""
    train_features, train_target, _ = load_block(con, table, TRAIN, feature_cols)
    logger.info("train block: %s rows", f"{len(train_target):,}")

    eval_features, eval_target, eval_zones = load_block(con, table, eval_block, feature_cols)
    logger.info("%s block: %s rows", eval_block, f"{len(eval_target):,}")

    tiers = np.array([zone_tiers.get(int(z), "low") for z in eval_zones])

    models: list[FittedBaseline] = [seasonal_naive(), hour_of_week_median()]
    logger.info("fitting Poisson GLM on %s rows", f"{len(train_target):,}")
    models.append(fit_poisson(train_features, train_target, feature_cols))
    if with_champion:
        logger.info("fitting champion GBM on %s rows", f"{len(train_target):,}")
        models.append(fit_champion(train_features, train_target, feature_cols, max_iter=max_iter))

    results: dict[str, list[TierMetrics]] = {}
    for model in models:
        predictions = np.clip(model.predict(eval_features), 0.0, None)
        results[model.name] = evaluate_by_tier(eval_target, predictions, tiers)
        logger.info("evaluated %s", model.name)

    return results


def format_results(results: dict[str, list[TierMetrics]], horizon: int) -> str:
    """A comparison table, tier by tier."""
    lines = [f"\nhorizon {horizon}h", "=" * 82]
    header = f"{'model':<22}{'tier':<9}{'n':>10}{'MAE':>9}{'deviance':>11}{'WAPE':>9}{'bias':>8}"
    lines.append(header)
    lines.append("-" * 82)

    for name, metrics in results.items():
        for m in metrics:
            wape_str = "  n/a" if np.isnan(m.wape) else f"{m.wape:>8.3f}"
            bias_str = "  n/a" if np.isnan(m.volume_bias) else f"{m.volume_bias:>7.3f}"
            lines.append(
                f"{name:<22}{m.tier:<9}{m.n:>10,}{m.mae:>9.3f}"
                f"{m.poisson_deviance:>11.3f}{wape_str:>9}{bias_str:>8}"
            )
        lines.append("-" * 82)
    return "\n".join(lines)
