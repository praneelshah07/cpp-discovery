"""Tests for the content-addressed embedding cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cpp_ai.embeddings.cache import EmbeddingCache, embedding_key


def test_key_deterministic_and_case_insensitive() -> None:
    assert embedding_key("m", "KLA") == embedding_key("m", "kla")
    assert embedding_key("m", "KLA") == embedding_key("m", " kla ")


def test_key_differs_by_model() -> None:
    assert embedding_key("esm2", "KLA") != embedding_key("prott5", "KLA")


def test_key_differs_by_sequence() -> None:
    assert embedding_key("m", "KLA") != embedding_key("m", "KLK")


def test_put_get_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    vec = np.arange(320, dtype=np.float32)
    cache.put("esm2_t6_8M", "KWKLFKKI", vec)
    got = cache.get("esm2_t6_8M", "KWKLFKKI")
    assert got is not None
    assert np.array_equal(got, vec)
    assert got.dtype == np.float32


def test_miss_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    assert cache.get("esm2", "KWKL") is None
    assert not cache.has("esm2", "KWKL")


def test_model_names_are_namespaced(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.put("model_a", "KWKL", np.ones(4, dtype=np.float32))
    cache.put("model_b", "KWKL", np.zeros(4, dtype=np.float32))
    assert np.array_equal(cache.get("model_a", "KWKL"), np.ones(4, dtype=np.float32))
    assert np.array_equal(cache.get("model_b", "KWKL"), np.zeros(4, dtype=np.float32))


def test_unsafe_model_name_is_sanitized(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.put("Rostlab/prot_t5", "KWKL", np.ones(4, dtype=np.float32))
    assert cache.has("Rostlab/prot_t5", "KWKL")


def test_len_counts_vectors(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.put("m", "AAAA", np.ones(2, dtype=np.float32))
    cache.put("m", "CCCC", np.ones(2, dtype=np.float32))
    assert len(cache) == 2


def test_overwrite_is_clean(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.put("m", "AAAA", np.ones(2, dtype=np.float32))
    cache.put("m", "AAAA", np.full(2, 9.0, dtype=np.float32))
    assert np.array_equal(cache.get("m", "AAAA"), np.full(2, 9.0, dtype=np.float32))
    assert len(cache) == 1
