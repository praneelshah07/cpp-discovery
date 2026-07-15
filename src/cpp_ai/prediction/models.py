"""The model zoo: sklearn-compatible classifier factories.

Each factory returns a fresh, unfitted estimator with sensible CPP-oriented
defaults and a fixed ``random_state`` for reproducibility. All are registered in
``MODEL_REGISTRY`` so evaluation/comparison code treats them uniformly.

Availability
------------
Random Forest, SVM, MLP (the "simple neural network"), and Logistic Regression
come from scikit-learn and are always available. XGBoost and LightGBM are
optional (the ``ml`` extra); each is registered only if it imports successfully,
so a missing/broken install (e.g. an XGBoost libomp conflict on macOS) simply
removes that model from the zoo rather than breaking the platform.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .base import MODEL_REGISTRY

logger = logging.getLogger(__name__)

_SEED = 42

# Models whose base estimator exposes per-member predictions (``estimators_``),
# enabling ensemble-disagreement (epistemic) uncertainty.
ENSEMBLE_MODELS: frozenset[str] = frozenset({"random_forest", "xgboost", "lightgbm"})


def random_forest(**params: Any) -> RandomForestClassifier:
    defaults = dict(n_estimators=300, max_depth=None, random_state=_SEED, n_jobs=-1)
    return RandomForestClassifier(**{**defaults, **params})


def svm(**params: Any) -> Pipeline:
    # SVM needs scaled inputs and probability=True for calibratable outputs.
    defaults = dict(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=_SEED)
    return Pipeline(
        [("scale", StandardScaler()), ("svc", SVC(**{**defaults, **params}))]
    )


def mlp(**params: Any) -> Pipeline:
    """A simple feed-forward neural network (scikit-learn, no torch)."""
    defaults = dict(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-3,
        max_iter=500,
        random_state=_SEED,
    )
    return Pipeline(
        [("scale", StandardScaler()), ("mlp", MLPClassifier(**{**defaults, **params}))]
    )


def logistic_regression(**params: Any) -> Pipeline:
    defaults = dict(C=1.0, max_iter=1000, random_state=_SEED)
    return Pipeline(
        [("scale", StandardScaler()), ("lr", LogisticRegression(**{**defaults, **params}))]
    )


MODEL_REGISTRY.register("random_forest", random_forest)
MODEL_REGISTRY.register("svm", svm)
MODEL_REGISTRY.register("mlp", mlp)
MODEL_REGISTRY.register("logistic_regression", logistic_regression)


def _register_optional() -> None:
    """Register XGBoost / LightGBM only if they import and load cleanly."""
    try:
        from xgboost import XGBClassifier

        def xgboost(**params: Any) -> Any:
            defaults = dict(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                eval_metric="logloss",
                random_state=_SEED,
            )
            return XGBClassifier(**{**defaults, **params})

        MODEL_REGISTRY.register("xgboost", xgboost, overwrite=True)
    except Exception as exc:  # noqa: BLE001 - import OR native-load (libomp) failure
        logger.warning("XGBoost unavailable, skipping: %s", exc)

    try:
        from lightgbm import LGBMClassifier

        def lightgbm(**params: Any) -> Any:
            defaults = dict(
                n_estimators=400,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.9,
                random_state=_SEED,
                n_jobs=-1,
                verbose=-1,
            )
            return LGBMClassifier(**{**defaults, **params})

        MODEL_REGISTRY.register("lightgbm", lightgbm, overwrite=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LightGBM unavailable, skipping: %s", exc)


_register_optional()
