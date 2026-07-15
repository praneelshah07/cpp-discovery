"""Tests for the end-to-end CPPClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.schema import Peptide
from cpp_ai.prediction import CPPClassifier, FeatureBuilder, make_synthetic_cpp_dataset

_BLOCKS = ["charge", "composition", "zscales"]


def _fitted(model: str = "random_forest", n: int = 80) -> tuple[CPPClassifier, list, np.ndarray]:
    peptides, y = make_synthetic_cpp_dataset(n_per_class=n, seed=3)
    fb = FeatureBuilder(descriptor_blocks=_BLOCKS)
    clf = CPPClassifier(model, feature_builder=fb).fit(peptides, y)
    return clf, peptides, y


def test_predict_returns_results_in_unit_interval() -> None:
    clf, peptides, _ = _fitted()
    results = clf.predict(peptides[:10])
    assert len(results) == 10
    for r in results:
        assert 0.0 <= r.probability <= 1.0
        assert r.predicted_label in (0, 1)
        assert 0.0 <= r.predictive_entropy <= 1.0
        assert r.model_name == "random_forest"


def test_random_forest_reports_epistemic_std() -> None:
    clf, peptides, _ = _fitted("random_forest")
    r = clf.predict(peptides[:5])[0]
    assert r.epistemic_std is not None
    assert r.epistemic_std >= 0.0


def test_non_ensemble_has_no_epistemic_std() -> None:
    clf, peptides, _ = _fitted("logistic_regression")
    r = clf.predict(peptides[:5])[0]
    assert r.epistemic_std is None


def test_predict_before_fit_raises() -> None:
    clf = CPPClassifier("random_forest", feature_builder=FeatureBuilder(descriptor_blocks=_BLOCKS))
    with pytest.raises(ValidationError):
        clf.predict([Peptide.from_sequence("KWKL", dataset="t")])


def test_label_length_mismatch_raises() -> None:
    peptides, y = make_synthetic_cpp_dataset(n_per_class=10, seed=0)
    clf = CPPClassifier("random_forest", feature_builder=FeatureBuilder(descriptor_blocks=_BLOCKS))
    with pytest.raises(ValidationError):
        clf.fit(peptides, y[:-1])


def test_single_class_training_raises() -> None:
    peptides, _ = make_synthetic_cpp_dataset(n_per_class=10, seed=0)
    clf = CPPClassifier("random_forest", feature_builder=FeatureBuilder(descriptor_blocks=_BLOCKS))
    with pytest.raises(ValidationError):
        clf.fit(peptides, [1] * len(peptides))


def test_uncalibrated_path_works() -> None:
    peptides, y = make_synthetic_cpp_dataset(n_per_class=40, seed=0)
    fb = FeatureBuilder(descriptor_blocks=_BLOCKS)
    clf = CPPClassifier("random_forest", feature_builder=fb, calibrate=False).fit(peptides, y)
    probs = clf.predict_proba(peptides[:5])
    assert probs.shape == (5,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_predict_empty_returns_empty() -> None:
    clf, _, _ = _fitted()
    assert clf.predict([]) == []


def test_classifier_learns_signal() -> None:
    clf, peptides, y = _fitted("random_forest", n=100)
    # in-sample sanity: calibrated probabilities correlate with labels
    probs = clf.predict_proba(peptides)
    auc_like = np.corrcoef(probs, y)[0, 1]
    assert auc_like > 0.3
