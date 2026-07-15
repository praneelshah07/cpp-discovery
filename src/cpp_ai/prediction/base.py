"""Prediction primitives: the model registry and the explainable result type.

The ML layer answers "how CPP-like is this peptide?" as a **calibrated
probability with an uncertainty estimate**, never a bare score. Everything a
downstream phase (ranking, optimization) needs to reason honestly about a
prediction lives on :class:`PredictionResult`.

Models are sklearn-compatible estimator *factories* registered by name, so the
cross-validation, calibration, and comparison machinery can treat every model
identically and swap them from configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from ..core.registry import Registry

#: A factory returning a fresh, unfitted sklearn-compatible classifier.
EstimatorFactory = Callable[..., Any]

#: Registry of model factories, selectable by name.
MODEL_REGISTRY: Registry[EstimatorFactory] = Registry("model")


def binary_entropy(p: float) -> float:
    """Predictive (Shannon) entropy of a Bernoulli(p), normalized to [0, 1].

    0 = fully confident (p=0 or 1), 1 = maximally uncertain (p=0.5). This is the
    model-agnostic uncertainty attached to every prediction.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


@dataclass(frozen=True)
class PredictionResult:
    """A single calibrated, uncertainty-aware prediction.

    Attributes
    ----------
    probability:
        Calibrated P(positive class = "CPP-like").
    predicted_label:
        1 if ``probability >= threshold`` else 0.
    confidence:
        ``max(probability, 1 - probability)`` — how decisive the call is.
    predictive_entropy:
        Total predictive uncertainty in [0, 1] (see :func:`binary_entropy`).
    epistemic_std:
        Disagreement among ensemble members (std of positive-class probability),
        when the model is an ensemble; ``None`` otherwise. High values flag
        peptides in sparsely-sampled regions where the model is guessing.
    """

    sequence: str
    probability: float
    predicted_label: int
    confidence: float
    predictive_entropy: float
    model_name: str
    peptide_id: str | None = None
    epistemic_std: float | None = None

    @classmethod
    def from_probability(
        cls,
        *,
        sequence: str,
        probability: float,
        model_name: str,
        peptide_id: str | None = None,
        threshold: float = 0.5,
        epistemic_std: float | None = None,
    ) -> "PredictionResult":
        """Build a result from a calibrated probability, deriving the rest."""
        p = float(probability)
        return cls(
            sequence=sequence,
            probability=p,
            predicted_label=int(p >= threshold),
            confidence=max(p, 1.0 - p),
            predictive_entropy=binary_entropy(p),
            model_name=model_name,
            peptide_id=peptide_id,
            epistemic_std=None if epistemic_std is None else float(epistemic_std),
        )
