"""Dimensionality reduction for embedding / descriptor visualization.

Projects high-dimensional peptide representations (pLM embeddings or descriptor
vectors) to 2-D for plotting. PCA and t-SNE come from scikit-learn and are always
available; UMAP is optional (``umap-learn``) and guarded with a clear error so a
missing/unbuildable install degrades gracefully rather than breaking imports.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ..core.exceptions import ValidationError

logger = logging.getLogger(__name__)

_METHODS = ("pca", "tsne", "umap")


def reduce_dimensions(
    X: npt.NDArray[np.float64],
    *,
    method: str = "pca",
    n_components: int = 2,
    standardize: bool = True,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Project ``X`` (n_samples, n_features) to ``n_components`` dimensions."""
    if method not in _METHODS:
        raise ValidationError(f"method must be one of {_METHODS}, got {method!r}.")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValidationError("X must be a 2-D (n_samples, n_features) array.")
    n = X.shape[0]
    if n < 2:
        raise ValidationError("Need at least 2 samples to reduce dimensions.")

    data = StandardScaler().fit_transform(X) if standardize else X

    if method == "pca":
        reducer: Any = PCA(n_components=min(n_components, X.shape[1]), random_state=seed)
        return np.asarray(reducer.fit_transform(data), dtype=np.float64)

    if method == "tsne":
        # perplexity must be < n_samples; scale it down for tiny sets.
        perplexity = min(30.0, max(1.0, (n - 1) / 3.0))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed, init="pca")
        return np.asarray(reducer.fit_transform(data), dtype=np.float64)

    # umap
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "UMAP requires umap-learn (optional). Install it, or use "
            "method='pca' / method='tsne'."
        ) from exc
    reducer = umap.UMAP(n_components=n_components, random_state=seed)
    return np.asarray(reducer.fit_transform(data), dtype=np.float64)
