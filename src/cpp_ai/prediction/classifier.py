"""CPPClassifier: the trained, calibrated, uncertainty-aware predictor.

Wraps a model factory + :class:`FeatureBuilder` into an end-to-end classifier
that turns peptides into :class:`PredictionResult`s. Two commitments:

* **Calibration.** Raw classifier scores are not probabilities. We wrap the
  model in ``CalibratedClassifierCV`` (isotonic or sigmoid/Platt) so a reported
  0.7 really means ~70% of such peptides are CPPs — essential for honest,
  comparable rankings.
* **Uncertainty.** Every prediction carries predictive entropy, and — for
  ensemble models — the disagreement (std) among ensemble members, flagging
  peptides where the model is extrapolating.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV

from ..core.exceptions import ValidationError
from ..core.schema import Peptide
from .base import MODEL_REGISTRY, EstimatorFactory, PredictionResult
from .features import FeatureBuilder
from .models import ENSEMBLE_MODELS

logger = logging.getLogger(__name__)


class CPPClassifier:
    """Train a calibrated CPP-likelihood model and predict with uncertainty."""

    def __init__(
        self,
        model: str | EstimatorFactory = "random_forest",
        *,
        feature_builder: FeatureBuilder | None = None,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        calibration_cv: int = 5,
        threshold: float = 0.5,
    ) -> None:
        if isinstance(model, str):
            self.model_name = model
            self._factory: EstimatorFactory = MODEL_REGISTRY.get(model)
        else:
            self.model_name = getattr(model, "__name__", "custom")
            self._factory = model
        self.feature_builder = feature_builder or FeatureBuilder()
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.threshold = threshold
        self._base: Any = None
        self._calibrated: Any = None
        self._fitted = False

    def fit(self, peptides: Sequence[Peptide], y: Sequence[int]) -> "CPPClassifier":
        """Fit the feature builder, base model, and probability calibrator."""
        y_arr = np.asarray(y).astype(int)
        if len(peptides) != len(y_arr):
            raise ValidationError("peptides and labels must have equal length.")
        if set(np.unique(y_arr).tolist()) != {0, 1}:
            raise ValidationError("Training labels must contain both classes 0 and 1.")

        X = self.feature_builder.fit_transform(peptides)

        # Uncalibrated base model — used for ensemble (epistemic) uncertainty.
        self._base = self._factory()
        self._base.fit(X, y_arr)

        if self.calibrate:
            n_splits = min(self.calibration_cv, int(np.bincount(y_arr).min()))
            if n_splits < 2:
                raise ValidationError("Too few minority-class samples to calibrate.")
            self._calibrated = CalibratedClassifierCV(
                self._factory(), method=self.calibration_method, cv=n_splits
            )
            self._calibrated.fit(X, y_arr)
        else:
            self._calibrated = self._base
        self._fitted = True
        return self

    def predict(self, peptides: Sequence[Peptide]) -> list[PredictionResult]:
        """Predict calibrated probability + uncertainty for each peptide."""
        self._require_fitted()
        if not peptides:
            return []
        X = self.feature_builder.transform(peptides)
        probs = self._calibrated.predict_proba(X)[:, 1]
        epi = self._epistemic_std(X)
        return [
            PredictionResult.from_probability(
                sequence=pep.sequence,
                probability=float(probs[i]),
                model_name=self.model_name,
                peptide_id=pep.peptide_id,
                threshold=self.threshold,
                epistemic_std=None if epi is None else float(epi[i]),
            )
            for i, pep in enumerate(peptides)
        ]

    def predict_proba(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        """Calibrated P(CPP) for each peptide."""
        self._require_fitted()
        X = self.feature_builder.transform(peptides)
        return np.asarray(self._calibrated.predict_proba(X)[:, 1], dtype=np.float64)

    def _epistemic_std(
        self, X: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64] | None:
        """Std of positive-class probability across ensemble members, if available."""
        if self.model_name not in ENSEMBLE_MODELS:
            return None
        estimators = getattr(self._base, "estimators_", None)
        if estimators is None:
            return None
        try:
            per_member = np.stack(
                [est.predict_proba(X)[:, 1] for est in np.ravel(estimators)]
            )
        except Exception:  # noqa: BLE001 - member without predict_proba
            return None
        return np.asarray(per_member.std(axis=0), dtype=np.float64)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValidationError("CPPClassifier is not fitted; call fit() first.")
