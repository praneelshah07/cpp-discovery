"""A deterministic, dependency-free mock embedder.

This is not only for tests. It lets the *entire* downstream platform (similarity,
ML, ranking) be wired up and exercised offline, without torch or multi-gigabyte
model downloads. Vectors are a deterministic function of the sequence, so the
same peptide always maps to the same point — useful for reproducible pipeline
development. It carries **no biological meaning** and must never be used for
real analysis.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import numpy.typing as npt

from .base import EMBEDDER_REGISTRY, Embedder


class MockEmbedder(Embedder):
    """Deterministic hash-based embedder. For plumbing and tests only."""

    model_name = "mock"

    def __init__(self, *, dim: int = 320, pooling: str = "mean") -> None:
        super().__init__(pooling=pooling)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_sequences(self, sequences: Sequence[str]) -> npt.NDArray[np.float32]:
        out = np.empty((len(sequences), self._dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            # Seed a per-sequence RNG from a stable hash for reproducibility.
            seed = int.from_bytes(
                hashlib.sha256(seq.encode("ascii")).digest()[:8], "big"
            )
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            out[i] = vec / norm if norm > 0 else vec
        return out


EMBEDDER_REGISTRY.register("mock", MockEmbedder)
