"""Tests for prediction base types, the model zoo, and the synthetic dataset."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.prediction import (
    ENSEMBLE_MODELS,
    MODEL_REGISTRY,
    PredictionResult,
    binary_entropy,
    make_synthetic_cpp_dataset,
)


# --- base ---
def test_binary_entropy_extremes_and_middle() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.1) == pytest.approx(binary_entropy(0.9))


def test_prediction_result_derivation() -> None:
    r = PredictionResult.from_probability(
        sequence="KWKL", probability=0.8, model_name="rf", epistemic_std=0.1
    )
    assert r.predicted_label == 1
    assert r.confidence == pytest.approx(0.8)
    assert 0 < r.predictive_entropy < 1
    assert r.epistemic_std == pytest.approx(0.1)


def test_prediction_result_threshold() -> None:
    r = PredictionResult.from_probability(
        sequence="AAAA", probability=0.4, model_name="rf"
    )
    assert r.predicted_label == 0
    assert r.confidence == pytest.approx(0.6)


# --- models ---
def test_core_models_registered() -> None:
    for name in ("random_forest", "svm", "mlp", "logistic_regression"):
        assert name in MODEL_REGISTRY


def test_random_forest_is_ensemble() -> None:
    assert "random_forest" in ENSEMBLE_MODELS


def test_factory_returns_fresh_fittable_estimator() -> None:
    factory = MODEL_REGISTRY.get("random_forest")
    est = factory(n_estimators=10)
    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.array([0, 1] * 10)
    est.fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (20, 2)


# --- synthetic ---
def test_synthetic_dataset_shape_and_balance() -> None:
    peptides, y = make_synthetic_cpp_dataset(n_per_class=50, seed=0)
    assert len(peptides) == 100
    assert len(y) == 100
    # roughly balanced (label noise perturbs it slightly)
    assert 30 < int(y.sum()) < 70


def test_synthetic_is_canonical_and_binary() -> None:
    peptides, y = make_synthetic_cpp_dataset(n_per_class=30, seed=0)
    assert all(p.is_canonical for p in peptides)
    assert set(np.unique(y).tolist()) <= {0, 1}


def test_synthetic_reproducible() -> None:
    a_pep, a_y = make_synthetic_cpp_dataset(n_per_class=20, seed=7)
    b_pep, b_y = make_synthetic_cpp_dataset(n_per_class=20, seed=7)
    assert [p.sequence for p in a_pep] == [p.sequence for p in b_pep]
    assert np.array_equal(a_y, b_y)


def test_synthetic_label_noise_validation() -> None:
    with pytest.raises(ValueError):
        make_synthetic_cpp_dataset(n_per_class=10, label_noise=0.6)


def test_synthetic_records_true_label_metadata() -> None:
    peptides, _ = make_synthetic_cpp_dataset(n_per_class=10, seed=1)
    assert "synthetic_true_label" in peptides[0].metadata
