"""A precomputed index for fast anchor-similarity queries over the library.

Building the index computes descriptor vectors (standardized) and, optionally,
pLM embeddings for every library peptide once. Queries then rank the whole
library against an anchor by weighted cosine similarity in milliseconds — the
per-query descriptor/embedding recomputation that would make a UI sluggish is
paid a single time up front.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..descriptors import compute_descriptors, descriptor_matrix
from ..embeddings.service import EmbeddingService
from .candidate import ScreenCandidate

_DEFAULT_BLOCKS = ["charge", "composition", "hydrophobic_moment", "biopython_props", "zscales", "aggregation"]


def _l2_normalize(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(matrix / norms, dtype=np.float64)


@dataclass
class ScreeningIndex:
    """Precomputed feature matrices for fast similarity ranking."""

    candidates: list[ScreenCandidate]
    feature_names: tuple[str, ...]
    descriptor_unit: npt.NDArray[np.float64]  # standardized + L2-normalized (n, d)
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    blocks: list[str]
    embedding_unit: npt.NDArray[np.float64] | None = None  # L2-normalized (n, e)
    embedding_service: EmbeddingService | None = None

    @classmethod
    def build(
        cls,
        candidates: Sequence[ScreenCandidate],
        *,
        descriptor_blocks: Sequence[str] | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> "ScreeningIndex":
        blocks = list(descriptor_blocks) if descriptor_blocks else list(_DEFAULT_BLOCKS)
        sets = [compute_descriptors(c.sequence, blocks=blocks) for c in candidates]
        dm = descriptor_matrix(sets)
        mean = dm.matrix.mean(axis=0)
        std = dm.matrix.std(axis=0)
        std[std == 0] = 1.0
        descriptor_unit = _l2_normalize((dm.matrix - mean) / std)

        embedding_unit = None
        if embedding_service is not None:
            recs = embedding_service.embed_sequences([c.sequence for c in candidates])
            embedding_unit = _l2_normalize(
                embedding_service.embedding_matrix(recs).astype(np.float64)
            )

        return cls(
            candidates=list(candidates),
            feature_names=dm.feature_names,
            descriptor_unit=descriptor_unit,
            mean=mean,
            std=std,
            blocks=blocks,
            embedding_unit=embedding_unit,
            embedding_service=embedding_service,
        )

    @property
    def has_embeddings(self) -> bool:
        return self.embedding_unit is not None

    def rank(
        self,
        anchor_sequence: str,
        *,
        embed_weight: float = 0.5,
        top_k: int | None = None,
    ) -> list[ScreenCandidate]:
        """Rank the library against ``anchor_sequence`` by weighted cosine similarity."""
        # Descriptor-profile similarity (always available).
        a_vec = compute_descriptors(anchor_sequence, blocks=self.blocks).vector(self.feature_names)
        a_unit = (a_vec - self.mean) / self.std
        norm = np.linalg.norm(a_unit)
        a_unit = a_unit / norm if norm else a_unit
        desc_sim = (self.descriptor_unit @ a_unit + 1.0) / 2.0  # [-1,1] -> [0,1]

        if self.has_embeddings and self.embedding_service is not None and embed_weight > 0:
            assert self.embedding_unit is not None
            ae = self.embedding_service.embedder.embed(anchor_sequence).astype(np.float64)
            n = np.linalg.norm(ae)
            ae = ae / n if n else ae
            emb_sim = (self.embedding_unit @ ae + 1.0) / 2.0
            w = min(max(embed_weight, 0.0), 1.0)
            sim = w * emb_sim + (1.0 - w) * desc_sim
        else:
            sim = desc_sim

        order = np.argsort(-sim)
        ranked = [self.candidates[i].with_similarity(float(sim[i])) for i in order]
        return ranked[:top_k] if top_k is not None else ranked
