"""Phase 5: machine learning for CPP likelihood.

A uniform model zoo (Random Forest, SVM, MLP, Logistic Regression, and — when
installed — XGBoost / LightGBM) behind one registry, with:

* stratified cross-validation and multi-metric model comparison,
* probability calibration (isotonic / Platt),
* uncertainty on every prediction (predictive entropy + ensemble disagreement),
* descriptors and/or embeddings as inputs (embeddings optional).

Every prediction is a :class:`PredictionResult` — a calibrated probability with
confidence and uncertainty, never a bare score.
"""

from __future__ import annotations

# Import models for their registration side effects.
from . import models  # noqa: F401
from .base import MODEL_REGISTRY, PredictionResult, binary_entropy
from .classifier import CPPClassifier
from .evaluation import (
    CalibrationCurve,
    CVReport,
    calibration_curve_data,
    compare_models,
    cross_validate,
)
from .features import FeatureBuilder
from .models import ENSEMBLE_MODELS
from .synthetic import make_synthetic_cpp_dataset
from .validation import cluster_sequences, family_split_cross_validate

__all__ = [
    "MODEL_REGISTRY",
    "ENSEMBLE_MODELS",
    "PredictionResult",
    "binary_entropy",
    "FeatureBuilder",
    "CPPClassifier",
    "cross_validate",
    "compare_models",
    "calibration_curve_data",
    "CVReport",
    "CalibrationCurve",
    "make_synthetic_cpp_dataset",
    "cluster_sequences",
    "family_split_cross_validate",
]
