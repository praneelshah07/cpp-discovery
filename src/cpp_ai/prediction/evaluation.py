"""Model evaluation: cross-validation, metrics, calibration, comparison.

Provides honest performance estimation via stratified k-fold cross-validation
and the calibration-curve data needed to check that predicted probabilities mean
what they say. All metrics are computed from held-out folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from ..core.exceptions import ValidationError
from .base import MODEL_REGISTRY, EstimatorFactory

logger = logging.getLogger(__name__)

# Metric name -> function(y_true, y_prob, y_pred). Probability-based metrics
# ignore y_pred; label-based ones ignore y_prob.
_METRICS: dict[str, Callable[..., float]] = {
    "roc_auc": lambda yt, yp, yd: float(roc_auc_score(yt, yp)),
    "pr_auc": lambda yt, yp, yd: float(average_precision_score(yt, yp)),
    "accuracy": lambda yt, yp, yd: float(accuracy_score(yt, yd)),
    "f1": lambda yt, yp, yd: float(f1_score(yt, yd, zero_division=0)),
    "mcc": lambda yt, yp, yd: float(matthews_corrcoef(yt, yd)),
    "brier": lambda yt, yp, yd: float(brier_score_loss(yt, yp)),
}


@dataclass
class CVReport:
    """Cross-validation results for one model."""

    model_name: str
    folds: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> dict[str, tuple[float, float]]:
        """Per-metric (mean, std) across folds."""
        out: dict[str, tuple[float, float]] = {}
        for metric in _METRICS:
            values = [f[metric] for f in self.folds if not np.isnan(f[metric])]
            if values:
                out[metric] = (float(np.mean(values)), float(np.std(values)))
        return out

    def summary_row(self) -> dict[str, float]:
        """Flat ``{metric_mean: value}`` row, handy for comparison tables."""
        row: dict[str, float] = {}
        for metric, (mean, std) in self.summary().items():
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        return row


@dataclass(frozen=True)
class CalibrationCurve:
    """Reliability-curve data for plotting predicted vs. observed frequency."""

    prob_predicted: npt.NDArray[np.float64]
    prob_observed: npt.NDArray[np.float64]
    brier_score: float


def _validate_binary(y: npt.NDArray[np.int_]) -> None:
    classes = set(np.unique(y).tolist())
    if not classes <= {0, 1}:
        raise ValidationError(f"Labels must be binary 0/1; got classes {sorted(classes)}.")
    if len(classes) < 2:
        raise ValidationError("Need both classes (0 and 1) present to evaluate.")


def cross_validate(
    factory: EstimatorFactory,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    *,
    model_name: str = "model",
    n_splits: int = 5,
    seed: int = 42,
) -> CVReport:
    """Stratified k-fold CV, returning per-fold held-out metrics."""
    y = np.asarray(y).astype(int)
    _validate_binary(y)
    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValidationError("Not enough samples in the minority class for 2-fold CV.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    report = CVReport(model_name=model_name)
    for train_idx, test_idx in skf.split(X, y):
        est = factory()
        est.fit(X[train_idx], y[train_idx])
        prob = est.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = y[test_idx]
        report.folds.append(
            {name: fn(yt, prob, pred) for name, fn in _METRICS.items()}
        )
    return report


def compare_models(
    models: Sequence[str] | Mapping[str, EstimatorFactory] | None,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, CVReport]:
    """Cross-validate several models and return their reports keyed by name.

    ``models`` may be a list of registered model names, a mapping of
    name->factory, or ``None`` to compare every registered model.
    """
    if models is None:
        factories = {name: MODEL_REGISTRY.get(name) for name in MODEL_REGISTRY.names()}
    elif isinstance(models, Mapping):
        factories = dict(models)
    else:
        factories = {name: MODEL_REGISTRY.get(name) for name in models}

    reports: dict[str, CVReport] = {}
    for name, factory in factories.items():
        logger.info("Cross-validating %s", name)
        reports[name] = cross_validate(
            factory, X, y, model_name=name, n_splits=n_splits, seed=seed
        )
    return reports


def calibration_curve_data(
    y_true: npt.NDArray[np.int_],
    y_prob: npt.NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> CalibrationCurve:
    """Reliability-curve points and Brier score for calibration assessment."""
    y_true = np.asarray(y_true).astype(int)
    _validate_binary(y_true)
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return CalibrationCurve(
        prob_predicted=predicted,
        prob_observed=observed,
        brier_score=float(brier_score_loss(y_true, y_prob)),
    )
