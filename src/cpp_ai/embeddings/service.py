"""Cache-backed embedding service.

Ties an :class:`Embedder` to an :class:`EmbeddingCache` so callers get
embeddings without worrying about recomputation: cache hits are returned
directly, and only cache *misses* are sent to the (expensive) model, in a single
batch, then persisted.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from ..core.schema import Peptide
from .base import Embedder, EmbeddingRecord
from .cache import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Compute-or-fetch embeddings through a disk cache."""

    def __init__(self, embedder: Embedder, cache: EmbeddingCache) -> None:
        self.embedder = embedder
        self.cache = cache

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def embed_sequences(
        self,
        sequences: Sequence[str],
        *,
        peptide_ids: Sequence[str | None] | None = None,
    ) -> list[EmbeddingRecord]:
        """Return one :class:`EmbeddingRecord` per input sequence, cache-first.

        Order is preserved. Duplicate sequences within the batch are computed
        once. Only sequences absent from the cache hit the model.
        """
        if peptide_ids is not None and len(peptide_ids) != len(sequences):
            raise ValueError("peptide_ids must align 1:1 with sequences.")
        ids = list(peptide_ids) if peptide_ids is not None else [None] * len(sequences)

        cleaned = [s.strip().upper() for s in sequences]
        model = self.embedder.model_name

        # Gather the unique sequences that are not already cached.
        to_compute: list[str] = []
        seen: set[str] = set()
        for s in cleaned:
            if s not in seen and not self.cache.has(model, s):
                to_compute.append(s)
                seen.add(s)

        if to_compute:
            logger.info(
                "Embedding %d new sequence(s) with %s (%d cached)",
                len(to_compute),
                model,
                len(cleaned) - len(to_compute),
            )
            vectors = self.embedder.embed_many(to_compute)
            for seq, vec in zip(to_compute, vectors):
                self.cache.put(model, seq, vec)

        records: list[EmbeddingRecord] = []
        for seq, pid in zip(cleaned, ids):
            vector = self.cache.get(model, seq)
            assert vector is not None  # just computed or already present
            records.append(
                EmbeddingRecord(
                    model_name=model, sequence=seq, vector=vector, peptide_id=pid
                )
            )
        return records

    def embed_peptides(self, peptides: Iterable[Peptide]) -> list[EmbeddingRecord]:
        """Embed :class:`Peptide` objects, carrying their ``peptide_id`` through."""
        peptides = list(peptides)
        return self.embed_sequences(
            [p.sequence for p in peptides],
            peptide_ids=[p.peptide_id for p in peptides],
        )

    def embedding_matrix(
        self, records: Sequence[EmbeddingRecord]
    ) -> npt.NDArray[np.float32]:
        """Stack records into an ``(N, dim)`` matrix for downstream use."""
        if not records:
            return np.empty((0, self.embedder.dim), dtype=np.float32)
        return np.vstack([r.vector for r in records]).astype(np.float32)
