"""Tests for the Embedder interface, pooling, records, and MockEmbedder."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.embeddings import EMBEDDER_REGISTRY, EmbeddingRecord, MockEmbedder


def test_record_dim() -> None:
    rec = EmbeddingRecord("m", "KWKL", np.zeros(320, dtype=np.float32))
    assert rec.dim == 320


def test_invalid_pooling_rejected() -> None:
    with pytest.raises(ValidationError):
        MockEmbedder(pooling="median")


def test_pool_mean_and_max() -> None:
    e_mean = MockEmbedder(pooling="mean")
    e_max = MockEmbedder(pooling="max")
    reps = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(e_mean._pool(reps), np.array([2.0, 3.0], dtype=np.float32))
    assert np.array_equal(e_max._pool(reps), np.array([3.0, 4.0], dtype=np.float32))


def test_embed_single_shape() -> None:
    vec = MockEmbedder(dim=64).embed("KWKLFKKI")
    assert vec.shape == (64,)
    assert vec.dtype == np.float32


def test_embed_many_shape() -> None:
    mat = MockEmbedder(dim=64).embed_many(["KWKL", "LLII", "RRRR"])
    assert mat.shape == (3, 64)


def test_embed_empty_batch() -> None:
    mat = MockEmbedder(dim=64).embed_many([])
    assert mat.shape == (0, 64)


def test_embed_empty_sequence_rejected() -> None:
    with pytest.raises(ValidationError):
        MockEmbedder().embed("   ")


def test_mock_is_deterministic() -> None:
    a = MockEmbedder(dim=128).embed("KWKLFKKI")
    b = MockEmbedder(dim=128).embed("KWKLFKKI")
    assert np.array_equal(a, b)


def test_mock_differs_by_sequence() -> None:
    a = MockEmbedder(dim=128).embed("KWKLFKKI")
    b = MockEmbedder(dim=128).embed("LLIILRRK")
    assert not np.array_equal(a, b)


def test_mock_is_unit_norm() -> None:
    vec = MockEmbedder(dim=128).embed("KWKLFKKI")
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_embedders_registered() -> None:
    for name in ("mock", "prott5", "esm2_t6_8M", "esm2_t33_650M"):
        assert name in EMBEDDER_REGISTRY
