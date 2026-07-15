"""Tests for dimensionality reduction."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.visualization import reduce_dimensions

try:
    import umap  # noqa: F401

    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def _X(n: int = 20, d: int = 15) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, d))


def test_pca_shape() -> None:
    coords = reduce_dimensions(_X(), method="pca")
    assert coords.shape == (20, 2)


def test_tsne_shape() -> None:
    coords = reduce_dimensions(_X(), method="tsne")
    assert coords.shape == (20, 2)


def test_pca_is_deterministic() -> None:
    a = reduce_dimensions(_X(), method="pca")
    b = reduce_dimensions(_X(), method="pca")
    assert np.allclose(a, b)


def test_invalid_method() -> None:
    with pytest.raises(ValidationError):
        reduce_dimensions(_X(), method="magic")


def test_too_few_samples() -> None:
    with pytest.raises(ValidationError):
        reduce_dimensions(np.zeros((1, 5)), method="pca")


def test_non_2d_input() -> None:
    with pytest.raises(ValidationError):
        reduce_dimensions(np.zeros(10), method="pca")


@pytest.mark.skipif(_HAS_UMAP, reason="umap-learn is installed")
def test_umap_absent_raises_clear_error() -> None:
    with pytest.raises(ImportError):
        reduce_dimensions(_X(), method="umap")
