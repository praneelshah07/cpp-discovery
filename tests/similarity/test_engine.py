"""Tests for the SimilarityEngine."""

from __future__ import annotations

import tempfile

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.schema import Peptide
from cpp_ai.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder
from cpp_ai.similarity import CompositeSimilarity, SimilarityEngine


def _P(seq: str) -> Peptide:
    return Peptide.from_sequence(seq, dataset="test")


def _mock_service() -> EmbeddingService:
    return EmbeddingService(MockEmbedder(dim=64), EmbeddingCache(tempfile.mkdtemp()))


def test_ranks_closer_peptides_higher_sequence_only() -> None:
    engine = SimilarityEngine(CompositeSimilarity({"sequence_identity": 1}))
    ref = _P("LLIILRRRIRKQAHAHSK")
    candidates = [_P("DDDDEEEEDDDD"), _P("LLIILRRRIRKQAHAHSA")]
    hits = engine.rank(ref, candidates)
    assert hits[0].target_sequence == "LLIILRRRIRKQAHAHSA"
    assert hits[0].composite > hits[1].composite


def test_empty_candidates_returns_empty() -> None:
    engine = SimilarityEngine(CompositeSimilarity({"sequence_identity": 1}))
    assert engine.rank(_P("LLIILRRK"), []) == []


def test_top_k_limits_results() -> None:
    engine = SimilarityEngine(CompositeSimilarity({"sequence_identity": 1}))
    cands = [_P("LLIILRRK"), _P("LLIILRRR"), _P("DDDDEEEE")]
    assert len(engine.rank(_P("LLIILRRK"), cands, top_k=2)) == 2


def test_embedding_metric_without_service_raises() -> None:
    with pytest.raises(ValidationError):
        SimilarityEngine(CompositeSimilarity({"embedding_cosine": 1}))


def test_descriptor_similarity_end_to_end() -> None:
    engine = SimilarityEngine(CompositeSimilarity({"descriptor_similarity": 1}))
    ref = _P("LLIILRRRIRKQAHAHSK")
    cands = [_P("DDDDEEEEDDDDEEEE"), _P("LLIILRRRIRKQAHAHSA")]
    hits = engine.rank(ref, cands)
    # physicochemically closest variant should win
    assert hits[0].target_sequence == "LLIILRRRIRKQAHAHSA"


def test_non_canonical_candidate_raises_for_descriptor_metric() -> None:
    engine = SimilarityEngine(CompositeSimilarity({"descriptor_similarity": 1}))
    with pytest.raises(ValidationError):
        engine.rank(_P("LLIILRRK"), [_P("LLXILRRK")])


def test_full_composite_with_mock_embeddings() -> None:
    engine = SimilarityEngine(CompositeSimilarity(), embedding_service=_mock_service())
    ref = _P("LLIILRRRIRKQAHAHSK")
    cands = [_P("LLIILRRRIRKQAHAHSA"), _P("DDDDEEEEDDDDEEEE")]
    hits = engine.rank(ref, cands)
    assert hits[0].target_sequence == "LLIILRRRIRKQAHAHSA"
    # every active metric appears in the explanation
    metrics = {c.metric for c in hits[0].breakdown.components}
    assert metrics == set(engine.composite.weights)


def test_hit_carries_reference_and_target_ids() -> None:
    engine = SimilarityEngine(CompositeSimilarity({"sequence_identity": 1}))
    ref = _P("LLIILRRK")
    cand = _P("LLIILRRR")
    hit = engine.rank(ref, [cand])[0]
    assert hit.reference_id == ref.peptide_id
    assert hit.target_id == cand.peptide_id
