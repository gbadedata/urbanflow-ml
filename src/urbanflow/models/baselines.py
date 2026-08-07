"""The baseline ladder.

Spec §2.3, as corrected. Three baselines, and the promotion gate is measured
against B rather than A:

    A  seasonal naive        demand at the same hour last week
    B  hour-of-week median   4-week trailing median for that zone and hour
    C  Poisson regression    interpretable statistical model on the feature set

A is reported as a sanity floor but is deliberately not the gate. It is a single
noisy observation, so a gradient booster beats it by a wide margin almost
regardless of quality — a benchmark that cannot be failed measures nothing.

B is cheap, strong, and genuinely hard to beat in stable zones. If the champion
cannot clear it, that is a finding worth publishing rather than a failure worth
hiding.

A and B need no fitting: both read straight off feature columns that the feature
builder already computed under the availability cutoff. Reusing those columns
rather than recomputing them is deliberate — the benchmark and the feature
cannot drift apart if they are the same expression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINE_A = "seasonal_naive"
BASELINE_B = "hour_of_week_median"
BASELINE_C = "poisson_regression"

#: Feature columns the unfitted baselines read directly.
BASELINE_A_COLUMN = "lag_168h"
BASELINE_B_COLUMN = "how_median_4w"

#: Prefixes of features measured in pickups. PoissonRegressor uses a log link, so
#: a prediction is exp(b.x). Feeding raw counts in makes demand enter
#: multiplicatively in the exponent — a zone at 200 pickups/hour contributes
#: exp(200b), which saturates or explodes depending on the sign, and the fitted
#: coefficient ends up dominated by the largest zones.
#:
#: Applying log1p first makes the model exp(b.log(1+x)) = (1+x)^b — a power law
#: in demand, which is the right shape for a count process and lets one global
#: model span zones from 0.1 to 300 pickups an hour.
DEMAND_FEATURE_PREFIXES = ("recent_", "lag_", "roll_mean_", "how_median_")


@dataclass
class FittedBaseline:
    """A baseline that can produce predictions for a feature matrix."""

    name: str
    description: str

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ColumnBaseline(FittedBaseline):
    """A baseline whose prediction is an existing feature column."""

    column: str = ""

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        if self.column not in features:
            raise KeyError(f"baseline {self.name} needs column {self.column!r}")
        return np.asarray(features[self.column], dtype=np.float64)


def seasonal_naive() -> ColumnBaseline:
    return ColumnBaseline(
        name=BASELINE_A,
        description="Demand at the same zone and hour one week earlier.",
        column=BASELINE_A_COLUMN,
    )


def hour_of_week_median() -> ColumnBaseline:
    return ColumnBaseline(
        name=BASELINE_B,
        description=(
            "Trailing 4-week median for this zone and hour-of-week. "
            "The benchmark the champion must beat (spec §8, gate 3)."
        ),
        column=BASELINE_B_COLUMN,
    )


def is_demand_feature(column: str) -> bool:
    """True when a feature is measured in pickups and needs a log transform."""
    return column.startswith(DEMAND_FEATURE_PREFIXES)


def design_matrix(features: dict[str, np.ndarray], columns: list[str]) -> np.ndarray:
    """Assemble the model matrix, log1p-transforming demand-magnitude columns."""
    parts = []
    for column in columns:
        values = np.asarray(features[column], dtype=np.float64)
        parts.append(np.log1p(np.clip(values, 0.0, None)) if is_demand_feature(column) else values)
    return np.column_stack(parts)


@dataclass
class PoissonBaseline(FittedBaseline):
    """Poisson GLM over the full feature set.

    A count model rather than least squares: the target is a non-negative count
    whose variance rises with its mean, which is what Poisson assumes and what
    squared error does not.
    """

    model: object = None
    columns: list[str] | None = None

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None or self.columns is None:
            raise RuntimeError("PoissonBaseline used before fitting")
        predictions = self.model.predict(  # type: ignore[attr-defined]
            design_matrix(features, self.columns)
        )
        return np.asarray(predictions, dtype=np.float64)


def fit_poisson(
    features: dict[str, np.ndarray], target: np.ndarray, columns: list[str]
) -> PoissonBaseline:
    """Fit a Poisson GLM on the training block only."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    matrix = design_matrix(features, columns)
    # Scaling is fitted here, on training data only (spec §7.4).
    model = make_pipeline(
        StandardScaler(),
        PoissonRegressor(alpha=1e-4, max_iter=500),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(matrix, target)
        except ConvergenceWarning as exc:
            raise RuntimeError(
                f"Poisson GLM did not converge: {exc}. A non-converged baseline "
                "cannot be used as a comparison point."
            ) from exc

    return PoissonBaseline(
        name=BASELINE_C,
        description=(
            "Poisson GLM over the full feature set, demand columns log1p-transformed "
            "and scaled in-fold."
        ),
        model=model,
        columns=columns,
    )
