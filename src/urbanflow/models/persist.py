"""Model artefact persistence.

A served model is only trustworthy if you can say what produced it. Each bundle
carries the snapshot ID, the feature list in order, the availability cutoff it
was trained under, and the dispersion parameter — so a prediction can be traced
back to the data and assumptions behind it, and so a model trained under a
different cutoff can be rejected rather than silently served (spec §11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_FORMAT_VERSION = 1


@dataclass
class ModelBundle:
    """A champion, its interval parameter, and everything needed to trust it."""

    model: Any
    horizon_hours: int
    feature_columns: list[str]
    feature_cutoff_minutes: int
    anchor_offset_hours: int
    dispersion: float
    snapshot_id: str
    train_start: str
    train_end: str
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    format_version: int = MODEL_FORMAT_VERSION
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        """Stable identifier combining horizon, snapshot and training date."""
        stamp = self.trained_at[:10].replace("-", "")
        return f"h{self.horizon_hours}-{self.snapshot_id[-12:]}-{stamp}"

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "format_version": self.format_version,
            "horizon_hours": self.horizon_hours,
            "feature_columns": self.feature_columns,
            "feature_cutoff_minutes": self.feature_cutoff_minutes,
            "anchor_offset_hours": self.anchor_offset_hours,
            "dispersion": self.dispersion,
            "snapshot_id": self.snapshot_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }

    def save(self, directory: Path) -> Path:
        """Write the model and a human-readable sidecar."""
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"champion_h{self.horizon_hours}.joblib"
        joblib.dump(self, path)
        (directory / f"champion_h{self.horizon_hours}.json").write_text(
            json.dumps(self.metadata(), indent=2)
        )
        return path

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        import joblib

        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} does not contain a ModelBundle")
        if bundle.format_version != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"{path} has format version {bundle.format_version}, expected "
                f"{MODEL_FORMAT_VERSION} — retrain rather than serving it"
            )
        return bundle

    def check_compatible(self, feature_cutoff_minutes: int) -> None:
        """Refuse to serve a model trained under a different availability rule.

        A model trained assuming 15 minutes of feed latency will silently
        mispredict if served where the latency is different — the features would
        mean something other than what it learned (spec §2.1).
        """
        if self.feature_cutoff_minutes != feature_cutoff_minutes:
            raise ValueError(
                f"model was trained with a {self.feature_cutoff_minutes}-minute feature "
                f"cutoff but the runtime is configured for {feature_cutoff_minutes} — "
                f"its features would not mean the same thing"
            )


def load_all(directory: Path) -> dict[int, ModelBundle]:
    """Load every champion in a directory, keyed by horizon."""
    bundles: dict[int, ModelBundle] = {}
    for path in sorted(directory.glob("champion_h*.joblib")):
        bundle = ModelBundle.load(path)
        bundles[bundle.horizon_hours] = bundle
    return bundles
