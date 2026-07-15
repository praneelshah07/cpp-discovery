"""Individual similarity metrics, each registered and swappable.

Every metric maps a (query, target) pair to a similarity in **[0, 1]** (1 =
identical) so heterogeneous metrics can be combined into a composite score. Each
declares what feature it ``requires`` so the engine knows what to precompute.

Metric families
---------------
* **Sequence** — ``sequence_identity`` (alignment identity fraction),
  ``smith_waterman`` (local alignment), ``needleman_wunsch`` (global alignment).
  Alignment uses BLOSUM62; raw scores are normalized by self-alignment score so
  identical sequences give 1.0.
* **Embedding** — ``embedding_cosine``: cosine similarity of pLM vectors,
  mapped from [-1, 1] to [0, 1]. Captures learned functional/structural context.
* **Descriptor** — ``descriptor_similarity``: closeness in standardized
  physicochemical descriptor space, as ``1 / (1 + Euclidean distance)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

from ..core.exceptions import ValidationError
from ..core.registry import Registry
from .features import PeptideFeatures

#: Registry of similarity-metric instances, selectable by name.
SIMILARITY_REGISTRY: Registry["SimilarityMetric"] = Registry("similarity metric")


class SimilarityMetric(ABC):
    """Base class: a named, bounded [0, 1] similarity between two peptides."""

    #: Stable metric name (registry key, weight key, breakdown label).
    name: str = "unknown"
    #: Feature attributes this metric reads: subset of {"sequence","embedding","descriptors"}.
    requires: frozenset[str] = frozenset({"sequence"})

    @abstractmethod
    def __call__(self, query: PeptideFeatures, target: PeptideFeatures) -> float:
        """Return similarity in [0, 1]."""


# --------------------------------------------------------------------------- #
# Sequence-alignment metrics
# --------------------------------------------------------------------------- #
def _make_aligner(mode: str, open_gap: float, extend_gap: float) -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap
    aligner.mode = mode
    return aligner


def _normalized_alignment_similarity(aligner: PairwiseAligner, a: str, b: str) -> float:
    """Self-normalized alignment score in [0, 1].

    ``score(a,b) / max(score(a,a), score(b,b))`` is 1.0 for identical sequences
    and is clamped to [0, 1] (global alignment scores can go negative for very
    dissimilar sequences).
    """
    s_ab = aligner.score(a, b)
    denom = max(aligner.score(a, a), aligner.score(b, b))
    if denom <= 0:
        return 0.0
    return float(min(1.0, max(0.0, s_ab / denom)))


class SmithWaterman(SimilarityMetric):
    """Local-alignment similarity (BLOSUM62), self-normalized to [0, 1]."""

    name = "smith_waterman"
    requires = frozenset({"sequence"})

    def __init__(self, open_gap: float = -11.0, extend_gap: float = -1.0) -> None:
        self._aligner = _make_aligner("local", open_gap, extend_gap)

    def __call__(self, query: PeptideFeatures, target: PeptideFeatures) -> float:
        return _normalized_alignment_similarity(self._aligner, query.sequence, target.sequence)


class NeedlemanWunsch(SimilarityMetric):
    """Global-alignment similarity (BLOSUM62), self-normalized to [0, 1]."""

    name = "needleman_wunsch"
    requires = frozenset({"sequence"})

    def __init__(self, open_gap: float = -11.0, extend_gap: float = -1.0) -> None:
        self._aligner = _make_aligner("global", open_gap, extend_gap)

    def __call__(self, query: PeptideFeatures, target: PeptideFeatures) -> float:
        return _normalized_alignment_similarity(self._aligner, query.sequence, target.sequence)


class SequenceIdentity(SimilarityMetric):
    """Fraction of identical positions in the optimal global alignment, in [0, 1]."""

    name = "sequence_identity"
    requires = frozenset({"sequence"})

    def __init__(self, open_gap: float = -11.0, extend_gap: float = -1.0) -> None:
        self._aligner = _make_aligner("global", open_gap, extend_gap)

    def __call__(self, query: PeptideFeatures, target: PeptideFeatures) -> float:
        a, b = query.sequence, target.sequence
        if a == b:
            return 1.0
        alignment = self._aligner.align(a, b)[0]
        identities = int(alignment.counts().identities)
        columns = int(alignment.shape[1])  # aligned length incl. gaps
        return identities / columns if columns else 0.0


# --------------------------------------------------------------------------- #
# Embedding metric
# --------------------------------------------------------------------------- #
class EmbeddingCosine(SimilarityMetric):
    """Cosine similarity of pLM embeddings, mapped from [-1, 1] to [0, 1]."""

    name = "embedding_cosine"
    requires = frozenset({"embedding"})

    def __call__(self, query: PeptideFeatures, target: PeptideFeatures) -> float:
        if query.embedding is None or target.embedding is None:
            raise ValidationError(
                "embedding_cosine requires embeddings on both peptides; "
                "none were provided (is an embedder configured?)."
            )
        a = query.embedding.astype(np.float64)
        b = target.embedding.astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        cos = float(np.dot(a, b) / (na * nb))
        return max(0.0, min(1.0, (cos + 1.0) / 2.0))


# --------------------------------------------------------------------------- #
# Descriptor metric
# --------------------------------------------------------------------------- #
class DescriptorSimilarity(SimilarityMetric):
    """Closeness in standardized descriptor space: ``1 / (1 + ||a - b||)``.

    Expects descriptor vectors already standardized by the engine (z-scored per
    feature across the candidate population) so that features on different scales
    contribute comparably. Identical vectors give 1.0; similarity decreases
    monotonically with Euclidean distance.
    """

    name = "descriptor_similarity"
    requires = frozenset({"descriptors"})

    def __call__(self, query: PeptideFeatures, target: PeptideFeatures) -> float:
        if query.descriptors is None or target.descriptors is None:
            raise ValidationError(
                "descriptor_similarity requires descriptor vectors on both peptides."
            )
        distance = float(np.linalg.norm(query.descriptors - target.descriptors))
        return 1.0 / (1.0 + distance)


def _register_defaults() -> None:
    for metric in (
        SequenceIdentity(),
        SmithWaterman(),
        NeedlemanWunsch(),
        EmbeddingCosine(),
        DescriptorSimilarity(),
    ):
        SIMILARITY_REGISTRY.register(metric.name, metric, overwrite=True)


_register_defaults()
