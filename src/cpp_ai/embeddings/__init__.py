"""Phase 3: protein language-model embeddings.

Produce fixed-length vectors for peptides using ESM-2 or ProtT5, cached on disk
so they are never recomputed. A deterministic ``MockEmbedder`` lets the whole
downstream pipeline run offline without torch.

Registered embedders (``EMBEDDER_REGISTRY``): ``mock``, ``prott5``, and the
ESM-2 checkpoints ``esm2_t6_8M`` / ``esm2_t12_35M`` / ``esm2_t30_150M`` /
``esm2_t33_650M`` (the default).

Heavy dependencies (torch, fair-esm, transformers) are imported lazily and live
behind the ``embeddings`` install extra; importing this package is cheap.
"""

from __future__ import annotations

# Import submodules for their registration side effects (no torch at import time).
from . import esm2, mock, prott5  # noqa: F401
from .base import EMBEDDER_REGISTRY, Embedder, EmbeddingRecord
from .cache import EmbeddingCache, embedding_key
from .esm2 import ESM2_MODELS, ESM2Embedder
from .mock import MockEmbedder
from .prott5 import ProtT5Embedder
from .service import EmbeddingService

__all__ = [
    "Embedder",
    "EmbeddingRecord",
    "EMBEDDER_REGISTRY",
    "EmbeddingCache",
    "embedding_key",
    "EmbeddingService",
    "MockEmbedder",
    "ESM2Embedder",
    "ESM2_MODELS",
    "ProtT5Embedder",
]
