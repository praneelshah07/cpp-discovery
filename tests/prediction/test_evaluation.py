"""Tests for cross-validation, comparison, and calibration utilities."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.prediction import (
    FeatureBuilder,
    calibration_curve_data,
    compare_models,
    cross_validate,
    make_synthetic_cpp_dataset,
)
from cpp_ai.prediction.base import MODEL_REGISTRY

_BLOCKS = ["charge", "composition", "zscales"]


def _Xy(n: int = 80, seed: int = 1):
    peptides, y = make_synthetic_cpp_dataset(n_per_class=n, seed=seed)
    X = FeatureBuilder(descriptor_blocks=_BLOCKS).fit_transform(peptides)
    return X, y


def test_cross_validate_returns_folds_and_summary() -> None:
    X, y = _Xy()
    report = cross_validate(MODEL_REGISTRY.get("random_forest"), X, y, n_splits=5)
    assert len(report.folds) == 5
    summary = report.summary()
    assert 0.0 <= summary["roc_auc"][0] <= 1.0


def test_learnable_signal_beats_chance() -> None:
    X, y = _Xy(n=100, seed=2)
    report = cross_validate(MODEL_REGISTRY.get("random_forest"), X, y, n_splits=5)
    # synthetic data has real (noisy) signal; AUC should be clearly above 0.5
    assert report.summary()["roc_auc"][0] > 0.6


def test_summary_row_is_flat() -> None:
    X, y = _Xy()
    row = cross_validate(MODEL_REGISTRY.get("logistic_regression"), X, y).summary_row()
    assert "roc_auc_mean" in row and "roc_auc_std" in row


def test_non_binary_labels_rejected() -> None:
    X, _ = _Xy()
    y_bad = np.array([2] * X.shape[0])
    with pytest.raises(ValidationError):
        cross_validate(MODEL_REGISTRY.get("random_forest"), X, y_bad)


def test_single_class_rejected() -> None:
    X, _ = _Xy()
    y_one = np.zeros(X.shape[0], dtype=int)
    with pytest.raises(ValidationError):
        cross_validate(MODEL_REGISTRY.get("random_forest"), X, y_one)


def test_compare_models_returns_report_per_model() -> None:
    X, y = _Xy()
    reports = compare_models(["random_forest", "logistic_regression"], X, y, n_splits=3)
    assert set(reports) == {"random_forest", "logistic_regression"}
    for rep in reports.values():
        assert rep.summary()["roc_auc"][0] >= 0.0


def test_calibration_curve_data() -> None:
    X, y = _Xy()
    rng = np.random.default_rng(0)
    # a noisy-but-correlated probability so bins are populated
    probs = np.clip(y * 0.6 + rng.uniform(0, 0.4, size=len(y)), 0, 1)
    cc = calibration_curve_data(y, probs, n_bins=5)
    assert cc.prob_predicted.shape == cc.prob_observed.shape
    assert 0.0 <= cc.brier_score <= 1.0
