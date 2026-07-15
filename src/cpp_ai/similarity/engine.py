"""The similarity engine: rank candidate peptides against a reference.

Given a reference peptide (e.g. pVEC) and candidates (e.g. all known CPPs), the
engine precomputes exactly the features the active metrics need — pLM embeddings
via the cached :class:`EmbeddingService`, and descriptor vectors standardized
across the population — then scores every candidate with the composite metric
and returns an explainable, ranked list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from ..core.schema import Peptide
from ..descriptors import compute_descriptors, descriptor_matrix
from ..descriptors.base import DescriptorSet
from ..embeddings.service import EmbeddingService
from .composite import CompositeSimilarity, SimilarityBreakdown
from .features import PeptideFeatures

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarityHit:
    """One ranked candidate with its explainable similarity breakdown."""

    reference_id: str | None
    target_id: str | None
    target_sequence: str
    breakdown: SimilarityBreakdown

    @property
    def composite(self) -> float:
        return self.breakdown.composite


class SimilarityEngine:
    """Rank candidates against a reference using a composite similarity."""

    def __init__(
        self,
        composite: CompositeSimilarity | None = None,
        *,
        embedding_service: EmbeddingService | None = None,
        descriptor_blocks: Sequence[str] | None = None,
    ) -> None:
        self.composite = composite or CompositeSimilarity()
        self.embedding_service = embedding_service
        self.descriptor_blocks = list(descriptor_blocks) if descriptor_blocks else None

        needs = self.composite.required_features
        if "embedding" in needs and embedding_service is None:
            raise ValidationError(
                "The composite includes an embedding metric but no embedding_service "
                "was provided. Pass one, or set its weight to 0."
            )

    def rank(
        self,
        reference: Peptide,
        candidates: Sequence[Peptide],
        *,
        top_k: int | None = None,
    ) -> list[SimilarityHit]:
        """Return candidates ranked by composite similarity to ``reference``."""
        if not candidates:
            return []

        peptides = [reference, *candidates]
        features = self._build_features(peptides)
        ref_features = features[0]

        hits: list[SimilarityHit] = []
        for candidate, feat in zip(candidates, features[1:]):
            breakdown = self.composite.score(ref_features, feat)
            hits.append(
                SimilarityHit(
                    reference_id=reference.peptide_id,
                    target_id=candidate.peptide_id,
                    target_sequence=candidate.sequence,
                    breakdown=breakdown,
                )
            )

        hits.sort(key=lambda h: h.composite, reverse=True)
        return hits[:top_k] if top_k is not None else hits

    # ------------------------------------------------------------------ #
    def _build_features(self, peptides: Sequence[Peptide]) -> list[PeptideFeatures]:
        needs = self.composite.required_features
        embeddings = self._compute_embeddings(peptides) if "embedding" in needs else None
        descriptors = self._compute_descriptors(peptides) if "descriptors" in needs else None

        features: list[PeptideFeatures] = []
        for i, pep in enumerate(peptides):
            features.append(
                PeptideFeatures(
                    sequence=pep.sequence,
                    peptide_id=pep.peptide_id,
                    embedding=None if embeddings is None else embeddings[i],
                    descriptors=None if descriptors is None else descriptors[i],
                )
            )
        return features

    def _compute_embeddings(
        self, peptides: Sequence[Peptide]
    ) -> list[npt.NDArray[np.float32]]:
        assert self.embedding_service is not None
        records = self.embedding_service.embed_peptides(peptides)
        return [r.vector for r in records]

    def _compute_descriptors(
        self, peptides: Sequence[Peptide]
    ) -> list[npt.NDArray[np.float64]]:
        non_canonical = [p.peptide_id for p in peptides if not p.is_canonical]
        if non_canonical:
            raise ValidationError(
                "Descriptor-based similarity requires canonical sequences; these "
                f"peptides are non-canonical: {non_canonical}. Filter them first."
            )
        sets: list[DescriptorSet] = [
            compute_descriptors(
                p.sequence, blocks=self.descriptor_blocks, peptide_id=p.peptide_id
            )
            for p in peptides
        ]
        dm = descriptor_matrix(sets)
        # Standardize per feature across the population so different-scale
        # descriptors contribute comparably to the distance.
        mean = dm.matrix.mean(axis=0)
        std = dm.matrix.std(axis=0)
        std[std == 0] = 1.0
        standardized = (dm.matrix - mean) / std
        return [standardized[i] for i in range(standardized.shape[0])]
