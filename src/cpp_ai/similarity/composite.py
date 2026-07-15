"""Composite similarity: a user-weighted blend of individual metrics.

Maximum adjustability is the design goal:

* **Any subset** of registered metrics can be included.
* **Arbitrary non-negative weights** — they need not sum to 1; they are
  normalized internally, so ``{"embedding_cosine": 3, "sequence_identity": 1}``
  is a valid "3:1" blend.
* **Custom-parameterized metric instances** can be injected (e.g. Smith-Waterman
  with different gap penalties) via ``overrides``.
* Every result carries a **full per-metric breakdown**, so a composite score is
  never returned without an explanation of how it was reached.

The default weighting favors embedding + descriptor similarity (0.30 each, 0.60
combined) over the three sequence-alignment metrics (0.40 combined), because
learned and physicochemical closeness track *functional* CPP similarity better
than raw sequence identity. All of this is overridable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..core.exceptions import ValidationError
from .features import PeptideFeatures
from .metrics import SIMILARITY_REGISTRY, SimilarityMetric

#: Default weights (un-normalized; normalized at construction).
DEFAULT_WEIGHTS: dict[str, float] = {
    "embedding_cosine": 0.30,
    "descriptor_similarity": 0.30,
    "sequence_identity": 0.40 / 3,
    "smith_waterman": 0.40 / 3,
    "needleman_wunsch": 0.40 / 3,
}


@dataclass(frozen=True)
class MetricScore:
    """One metric's contribution to a composite score."""

    metric: str
    score: float           # raw metric similarity in [0, 1]
    weight: float          # normalized weight in [0, 1]
    contribution: float    # score * weight


@dataclass(frozen=True)
class SimilarityBreakdown:
    """A composite similarity plus its full per-metric explanation."""

    composite: float
    components: tuple[MetricScore, ...]

    def top_contributors(self, n: int = 3) -> tuple[MetricScore, ...]:
        """The ``n`` metrics contributing most to the composite score."""
        return tuple(sorted(self.components, key=lambda m: m.contribution, reverse=True)[:n])


class CompositeSimilarity:
    """A normalized, weighted blend of similarity metrics."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        overrides: Mapping[str, SimilarityMetric] | None = None,
    ) -> None:
        raw = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if not raw:
            raise ValidationError("At least one metric weight must be provided.")
        if any(w < 0 for w in raw.values()):
            raise ValidationError("Metric weights must be non-negative.")
        total = sum(raw.values())
        if total <= 0:
            raise ValidationError("Sum of metric weights must be positive.")

        overrides = dict(overrides or {})
        self._metrics: dict[str, SimilarityMetric] = {}
        self._weights: dict[str, float] = {}
        for name, weight in raw.items():
            if weight == 0:
                continue
            metric = overrides.get(name) or SIMILARITY_REGISTRY.get(name)
            self._metrics[name] = metric
            self._weights[name] = weight / total

    @property
    def weights(self) -> Mapping[str, float]:
        """The normalized weights actually in use (summing to 1)."""
        return dict(self._weights)

    @property
    def required_features(self) -> frozenset[str]:
        """Union of features every active metric needs (for the engine)."""
        needed: set[str] = set()
        for metric in self._metrics.values():
            needed |= metric.requires
        return frozenset(needed)

    def score(self, query: PeptideFeatures, target: PeptideFeatures) -> SimilarityBreakdown:
        """Compute the composite similarity and its per-metric breakdown."""
        components: list[MetricScore] = []
        composite = 0.0
        for name, metric in self._metrics.items():
            weight = self._weights[name]
            raw = metric(query, target)
            if not 0.0 <= raw <= 1.0:
                raise ValidationError(
                    f"Metric {name!r} returned {raw!r}, outside [0, 1]."
                )
            contribution = raw * weight
            composite += contribution
            components.append(MetricScore(name, raw, weight, contribution))
        return SimilarityBreakdown(composite=composite, components=tuple(components))
