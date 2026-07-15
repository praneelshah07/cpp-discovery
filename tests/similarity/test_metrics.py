"""Tests for individual similarity metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.similarity import (
    SIMILARITY_REGISTRY,
    DescriptorSimilarity,
    EmbeddingCosine,
    NeedlemanWunsch,
    SequenceIdentity,
    SmithWaterman,
)
from cpp_ai.similarity.features import PeptideFeatures


def _f(seq: str = "", *, emb=None, desc=None) -> PeptideFeatures:
    return PeptideFeatures(sequence=seq, embedding=emb, descriptors=desc)


# --- sequence metrics ---
@pytest.mark.parametrize("metric", [SequenceIdentity(), SmithWaterman(), NeedlemanWunsch()])
def test_identical_sequences_score_one(metric) -> None:
    assert metric(_f("LLIILRRK"), _f("LLIILRRK")) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [SequenceIdentity(), SmithWaterman(), NeedlemanWunsch()])
def test_scores_are_in_unit_interval(metric) -> None:
    s = metric(_f("LLIILRRK"), _f("DDDDEEEE"))
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("metric", [SmithWaterman(), NeedlemanWunsch()])
def test_alignment_is_symmetric(metric) -> None:
    a, b = _f("LLIILRRK"), _f("LLIILRRR")
    assert metric(a, b) == pytest.approx(metric(b, a))


def test_sequence_identity_single_mismatch() -> None:
    # 7/8 identical, no gaps -> 0.875
    assert SequenceIdentity()(_f("LLIILRRK"), _f("LLIILRRR")) == pytest.approx(7 / 8)


def test_similar_scores_higher_than_dissimilar() -> None:
    sw = SmithWaterman()
    close = sw(_f("LLIILRRK"), _f("LLIILRRR"))
    far = sw(_f("LLIILRRK"), _f("DDDDEEEE"))
    assert close > far


# --- embedding metric ---
def test_embedding_cosine_identical_is_one() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert EmbeddingCosine()(_f(emb=v), _f(emb=v)) == pytest.approx(1.0)


def test_embedding_cosine_opposite_is_zero() -> None:
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert EmbeddingCosine()(_f(emb=v), _f(emb=-v)) == pytest.approx(0.0)


def test_embedding_cosine_orthogonal_is_half() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert EmbeddingCosine()(_f(emb=a), _f(emb=b)) == pytest.approx(0.5)


def test_embedding_cosine_missing_raises() -> None:
    with pytest.raises(ValidationError):
        EmbeddingCosine()(_f("AAAA"), _f("AAAA"))


# --- descriptor metric ---
def test_descriptor_identical_is_one() -> None:
    v = np.array([0.5, -1.0, 2.0])
    assert DescriptorSimilarity()(_f(desc=v), _f(desc=v)) == pytest.approx(1.0)


def test_descriptor_decreases_with_distance() -> None:
    base = np.zeros(3)
    near = DescriptorSimilarity()(_f(desc=base), _f(desc=np.array([1.0, 0, 0])))
    far = DescriptorSimilarity()(_f(desc=base), _f(desc=np.array([5.0, 0, 0])))
    assert 0 < far < near < 1


def test_descriptor_missing_raises() -> None:
    with pytest.raises(ValidationError):
        DescriptorSimilarity()(_f("AAAA"), _f("AAAA"))


def test_all_metrics_registered() -> None:
    for name in (
        "sequence_identity",
        "smith_waterman",
        "needleman_wunsch",
        "embedding_cosine",
        "descriptor_similarity",
    ):
        assert name in SIMILARITY_REGISTRY
