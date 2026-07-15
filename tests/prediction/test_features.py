"""Tests for the FeatureBuilder."""

from __future__ import annotations

import tempfile

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.schema import Peptide
from cpp_ai.embeddings import EmbeddingCache, EmbeddingService, MockEmbedder
from cpp_ai.prediction import FeatureBuilder

_BLOCKS = ["charge", "composition"]


def _peps() -> list[Peptide]:
    return [
        Peptide.from_sequence("KWKLFKKI", dataset="t"),
        Peptide.from_sequence("LLIILRRK", dataset="t"),
        Peptide.from_sequence("DDDDEEEE", dataset="t"),
    ]


def test_requires_a_feature_source() -> None:
    with pytest.raises(ValidationError):
        FeatureBuilder(use_descriptors=False, embedding_service=None)


def test_descriptor_only_shape_and_names() -> None:
    fb = FeatureBuilder(descriptor_blocks=_BLOCKS)
    X = fb.fit_transform(_peps())
    assert X.shape[0] == 3
    assert X.shape[1] == len(fb.feature_names)
    assert "frac_K" in fb.feature_names


def test_transform_before_fit_raises() -> None:
    fb = FeatureBuilder(descriptor_blocks=_BLOCKS)
    with pytest.raises(ValidationError):
        fb.transform(_peps())


def test_feature_names_before_fit_raises() -> None:
    fb = FeatureBuilder(descriptor_blocks=_BLOCKS)
    with pytest.raises(ValidationError):
        _ = fb.feature_names


def test_non_canonical_raises() -> None:
    fb = FeatureBuilder(descriptor_blocks=_BLOCKS)
    with pytest.raises(ValidationError):
        fb.fit_transform([Peptide.from_sequence("KWXL", dataset="t")])


def test_transform_column_order_is_stable() -> None:
    fb = FeatureBuilder(descriptor_blocks=_BLOCKS)
    fb.fit(_peps())
    x_all = fb.transform(_peps())
    x_one = fb.transform([_peps()[1]])
    # row for peptide index 1 must match whether transformed alone or in a batch
    assert x_one.shape == (1, x_all.shape[1])
    assert (x_one[0] == x_all[1]).all()


def test_embeddings_add_dimensions() -> None:
    svc = EmbeddingService(MockEmbedder(dim=16), EmbeddingCache(tempfile.mkdtemp()))
    fb_desc = FeatureBuilder(descriptor_blocks=_BLOCKS)
    fb_both = FeatureBuilder(descriptor_blocks=_BLOCKS, embedding_service=svc)
    n_desc = fb_desc.fit_transform(_peps()).shape[1]
    x_both = fb_both.fit_transform(_peps())
    assert x_both.shape[1] == n_desc + 16


def test_embeddings_only() -> None:
    svc = EmbeddingService(MockEmbedder(dim=16), EmbeddingCache(tempfile.mkdtemp()))
    fb = FeatureBuilder(use_descriptors=False, embedding_service=svc)
    X = fb.fit_transform(_peps())
    assert X.shape == (3, 16)
