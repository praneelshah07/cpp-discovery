"""Phase 4: the similarity engine.

Rank peptides by similarity to a reference (e.g. pVEC, ClWOX) using a
user-weighted blend of complementary metrics:

* sequence: identity, Smith-Waterman (local), Needleman-Wunsch (global)
* embedding: pLM cosine similarity
* descriptor: distance in standardized physicochemical space

Every metric is registered and swappable (``SIMILARITY_REGISTRY``); weights are
fully user-adjustable (``CompositeSimilarity``); every score comes with a
per-metric explanation.
"""

from __future__ import annotations

from .composite import (
    DEFAULT_WEIGHTS,
    CompositeSimilarity,
    MetricScore,
    SimilarityBreakdown,
)
from .engine import SimilarityEngine, SimilarityHit
from .features import PeptideFeatures
from .metrics import (
    SIMILARITY_REGISTRY,
    DescriptorSimilarity,
    EmbeddingCosine,
    NeedlemanWunsch,
    SequenceIdentity,
    SimilarityMetric,
    SmithWaterman,
)

__all__ = [
    "SIMILARITY_REGISTRY",
    "SimilarityMetric",
    "SequenceIdentity",
    "SmithWaterman",
    "NeedlemanWunsch",
    "EmbeddingCosine",
    "DescriptorSimilarity",
    "CompositeSimilarity",
    "DEFAULT_WEIGHTS",
    "MetricScore",
    "SimilarityBreakdown",
    "PeptideFeatures",
    "SimilarityEngine",
    "SimilarityHit",
]
