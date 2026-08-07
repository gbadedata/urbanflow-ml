"""Champion model and promotion gate.

``HistGradientBoostingRegressor`` with Poisson loss. The loss matches the target
— non-negative counts whose variance rises with their mean — and matches the
gate metric, so the model is optimising the thing it is judged on rather than a
proxy.

Promotion is decided by :func:`evaluate_gate` against Baseline B, using the
criteria fixed in ``reports/gate_decision.md`` before this module was first run.
The gate is genuinely two-way: shipping the baseline with the evidence is a pass,
and the code says so rather than treating it as a failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from urbanflow.config import Config
from urbanflow.models.baselines import FittedBaseline, design_matrix
from urbanflow.models.metrics import TierMetrics, relative_improvement

logger = logging.getLogger(__name__)

CHAMPION = "gbm_poisson"

#: Calendar columns are genuinely categorical: hour 23 is adjacent to hour 0, and
#: a split on "month <= 6" is meaningless. The cyclical sin/cos encodings remain
#: available alongside, so the model can use whichever representation helps.
CATEGORICAL_FEATURES = ("local_hour", "local_dow", "local_month")


@dataclass
class ChampionModel(FittedBaseline):
    """Fitted gradient booster."""

    model: object = None
    columns: list[str] = field(default_factory=list)

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ChampionModel used before fitting")
        matrix = design_matrix(features, self.columns)
        predictions = self.model.predict(matrix)  # type: ignore[attr-defined]
        return np.clip(np.asarray(predictions, dtype=np.float64), 0.0, None)


def fit_champion(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    columns: list[str],
    *,
    random_state: int = 0,
    max_iter: int = 400,
) -> ChampionModel:
    """Fit the champion on the training block only.

    Demand columns are log1p-transformed for the same reason as the GLM: it makes
    one pooled model span zones differing by three orders of magnitude. Trees are
    invariant to monotone transforms of a single feature, so this does not change
    what splits are reachable, but it does keep the two models comparable on
    identical inputs.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    matrix = design_matrix(features, columns)
    categorical = [c in CATEGORICAL_FEATURES for c in columns]

    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=max_iter,
        learning_rate=0.1,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=1.0,
        categorical_features=categorical,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )
    model.fit(matrix, target)
    logger.info("champion fitted in %d boosting iterations", model.n_iter_)

    return ChampionModel(
        name=CHAMPION,
        description="HistGradientBoostingRegressor, Poisson loss, log1p demand features.",
        model=model,
        columns=columns,
    )


@dataclass(frozen=True)
class GateResult:
    """Outcome of the promotion decision for one horizon."""

    horizon_hours: int
    metric: str
    passed: bool
    ships: str
    improvements: dict[str, float]
    bias_violations: list[str]
    degradations: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon_hours": self.horizon_hours,
            "metric": self.metric,
            "passed": self.passed,
            "ships": self.ships,
            "improvements": {k: round(v, 4) for k, v in self.improvements.items()},
            "bias_violations": self.bias_violations,
            "degradations": self.degradations,
            "reasons": self.reasons,
        }


def _metric_value(metrics: TierMetrics, name: str) -> float:
    return {
        "poisson_deviance": metrics.poisson_deviance,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
    }[name]


def evaluate_gate(
    config: Config,
    horizon_hours: int,
    champion: list[TierMetrics],
    benchmark: list[TierMetrics],
) -> GateResult:
    """Apply the promotion criteria fixed in reports/gate_decision.md."""
    gates = config.gates
    champ = {m.tier: m for m in champion}
    bench = {m.tier: m for m in benchmark}

    improvements: dict[str, float] = {}
    reasons: list[str] = []
    bias_violations: list[str] = []
    degradations: list[str] = []

    for tier in champ:
        if tier not in bench:
            continue
        improvements[tier] = relative_improvement(
            _metric_value(champ[tier], gates.metric),
            _metric_value(bench[tier], gates.metric),
        )

    # 1. Required improvement on every gated tier.
    cleared = True
    for tier in gates.gated_tiers:
        if tier not in improvements:
            reasons.append(f"gated tier {tier!r} missing from results")
            cleared = False
            continue
        if improvements[tier] < gates.min_relative_improvement:
            cleared = False
            reasons.append(
                f"{tier}: {gates.metric} improved {improvements[tier]:.1%}, "
                f"below the {gates.min_relative_improvement:.0%} threshold"
            )

    # 2. Volume bias is a hard constraint in every tier, gated or not.
    for tier, metrics in champ.items():
        if tier == "all" or np.isnan(metrics.volume_bias):
            continue
        deviation = abs(metrics.volume_bias - 1.0)
        if deviation > gates.max_volume_bias_deviation:
            bias_violations.append(f"{tier} (bias {metrics.volume_bias:.3f})")
            cleared = False
            reasons.append(
                f"{tier}: volume bias {metrics.volume_bias:.3f} exceeds the "
                f"±{gates.max_volume_bias_deviation:.0%} tolerance"
            )

    # 3. No tier may get materially worse, including ungated ones.
    for tier, improvement in improvements.items():
        if tier == "all":
            continue
        if improvement < -gates.max_tier_degradation:
            degradations.append(f"{tier} ({improvement:.1%})")
            cleared = False
            reasons.append(
                f"{tier}: {gates.metric} degraded {abs(improvement):.1%}, beyond the "
                f"{gates.max_tier_degradation:.0%} tolerance"
            )

    if cleared:
        reasons.append(
            f"champion clears the gate on {', '.join(gates.gated_tiers)} "
            f"with no bias violation or tier degradation"
        )

    return GateResult(
        horizon_hours=horizon_hours,
        metric=gates.metric,
        passed=cleared,
        # Shipping the baseline with evidence is a pass, not a failure (spec §8).
        ships=CHAMPION if cleared else "hour_of_week_median",
        improvements=improvements,
        bias_violations=bias_violations,
        degradations=degradations,
        reasons=reasons,
    )


def format_gate(result: GateResult) -> str:
    verdict = "PROMOTE" if result.passed else "KEEP BASELINE"
    lines = [
        f"\nhorizon {result.horizon_hours}h — {verdict} (ships: {result.ships})",
        f"  gate metric: {result.metric}",
    ]
    for tier, improvement in sorted(result.improvements.items()):
        lines.append(f"  {tier:<8} {improvement:>+7.1%} vs Baseline B")
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)
