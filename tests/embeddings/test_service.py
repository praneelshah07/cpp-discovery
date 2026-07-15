"""Tests for the cache-backed EmbeddingService."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from cpp_ai.core.schema import Peptide
from cpp_ai.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder


class CountingEmbedder(MockEmbedder):
    """MockEmbedder that records how many sequences it actually computed."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.computed: list[str] = []

    def _embed_sequences(self, sequences: Sequence[str]) -> npt.NDArray[np.float32]:
        self.computed.extend(sequences)
        return super()._embed_sequences(sequences)


def _service(tmp_path: Path) -> tuple[EmbeddingService, CountingEmbedder]:
    embedder = CountingEmbedder()
    service = EmbeddingService(embedder, EmbeddingCache(tmp_path))
    return service, embedder


def test_first_call_computes_second_call_hits_cache(tmp_path: Path) -> None:
    service, embedder = _service(tmp_path)
    service.embed_sequences(["KWKLFKKI", "LLIILRRK"])
    assert sorted(embedder.computed) == ["KWKLFKKI", "LLIILRRK"]

    # Second call: nothing new should be computed.
    embedder.computed.clear()
    recs = service.embed_sequences(["KWKLFKKI", "LLIILRRK"])
    assert embedder.computed == []
    assert len(recs) == 2


def test_duplicate_sequences_computed_once(tmp_path: Path) -> None:
    service, embedder = _service(tmp_path)
    recs = service.embed_sequences(["KWKL", "KWKL", "KWKL"])
    assert embedder.computed == ["KWKL"]
    assert len(recs) == 3
    assert np.array_equal(recs[0].vector, recs[2].vector)


def test_order_preserved(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    seqs = ["AAAA", "CCCC", "DDDD"]
    recs = service.embed_sequences(seqs)
    assert [r.sequence for r in recs] == seqs


def test_embed_peptides_carries_ids(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    peps = [
        Peptide.from_sequence("KWKLFKKI", dataset="t"),
        Peptide.from_sequence("LLIILRRK", dataset="t"),
    ]
    recs = service.embed_peptides(peps)
    assert [r.peptide_id for r in recs] == [p.peptide_id for p in peps]


def test_partial_cache_only_computes_misses(tmp_path: Path) -> None:
    service, embedder = _service(tmp_path)
    service.embed_sequences(["KWKL"])
    embedder.computed.clear()
    service.embed_sequences(["KWKL", "NEWSEQ"])
    assert embedder.computed == ["NEWSEQ"]


def test_embedding_matrix_shape(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    recs = service.embed_sequences(["AAAA", "CCCC"])
    mat = service.embedding_matrix(recs)
    assert mat.shape == (2, 32)
