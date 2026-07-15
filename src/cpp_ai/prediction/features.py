"""Assemble model feature matrices from peptides.

Combines Phase 2 descriptors and/or Phase 3 embeddings into one matrix. Feature
*scaling* is intentionally left to the model pipelines (so it is re-fit per CV
fold and never leaks across folds); this builder only assembles raw features in
a stable column order.

Embeddings are an **optional** input, matching the project spec — set
``embedding_service`` to include them, otherwise descriptors alone are used.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from ..core.schema import Peptide
from ..descriptors import compute_descriptors, descriptor_matrix
from ..embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build a ``(n_peptides, n_features)`` matrix from descriptors + embeddings."""

    def __init__(
        self,
        *,
        use_descriptors: bool = True,
        descriptor_blocks: Sequence[str] | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        if not use_descriptors and embedding_service is None:
            raise ValidationError(
                "FeatureBuilder needs at least one feature source "
                "(use_descriptors=True and/or an embedding_service)."
            )
        self.use_descriptors = use_descriptors
        self.descriptor_blocks = list(descriptor_blocks) if descriptor_blocks else None
        self.embedding_service = embedding_service
        self._descriptor_features: tuple[str, ...] | None = None
        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        if not self._fitted:
            raise ValidationError("FeatureBuilder is not fitted.")
        names: list[str] = list(self._descriptor_features or ())
        if self.embedding_service is not None:
            names += [f"emb_{i}" for i in range(self.embedding_service.embedder.dim)]
        return names

    def fit(self, peptides: Sequence[Peptide]) -> "FeatureBuilder":
        """Establish the descriptor feature set (stable column order)."""
        if self.use_descriptors:
            self._check_canonical(peptides)
            sets = [
                compute_descriptors(p.sequence, blocks=self.descriptor_blocks)
                for p in peptides
            ]
            self._descriptor_features = descriptor_matrix(sets).feature_names
        else:
            self._descriptor_features = ()
        self._fitted = True
        return self

    def transform(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        """Assemble the feature matrix for ``peptides``."""
        if not self._fitted:
            raise ValidationError("Call fit() (or fit_transform()) before transform().")
        blocks: list[npt.NDArray[np.float64]] = []

        if self.use_descriptors:
            assert self._descriptor_features is not None
            self._check_canonical(peptides)
            rows = [
                compute_descriptors(p.sequence, blocks=self.descriptor_blocks).vector(
                    self._descriptor_features
                )
                for p in peptides
            ]
            blocks.append(np.vstack(rows) if rows else np.empty((0, len(self._descriptor_features))))

        if self.embedding_service is not None:
            records = self.embedding_service.embed_peptides(peptides)
            emb = np.vstack([r.vector for r in records]).astype(np.float64)
            blocks.append(emb)

        return np.hstack(blocks) if len(blocks) > 1 else blocks[0]

    def fit_transform(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        return self.fit(peptides).transform(peptides)

    @staticmethod
    def _check_canonical(peptides: Sequence[Peptide]) -> None:
        bad = [p.peptide_id for p in peptides if not p.is_canonical]
        if bad:
            raise ValidationError(
                f"Descriptor features require canonical sequences; non-canonical: {bad}."
            )
