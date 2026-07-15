"""Embedding interfaces and records.

Protein language models (pLMs) like ESM-2 and ProtT5 map a sequence to a
fixed-length vector that captures evolutionary and structural context far beyond
hand-crafted descriptors. Those vectors power the similarity engine (Phase 4)
and can be optional inputs to the ML models (Phase 5).

Design
------
* **`Embedder`** is a thin, model-agnostic interface: subclasses only implement
  batched `_embed_sequences`. Heavy libraries (torch, fair-esm, transformers)
  are imported lazily *inside* subclasses so the rest of the platform never
  pays for them.
* **`EmbeddingRecord`** bundles a vector with the model it came from and the
  peptide it describes, so a cached vector is never mistaken for one from a
  different model.
* Embeddings are **deterministic per (model, sequence)** and expensive, so they
  are always routed through the disk cache (see ``cache.py``) — never recomputed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from ..core.registry import Registry

logger = logging.getLogger(__name__)

#: Registry of embedder classes, selectable by model name from config.
EMBEDDER_REGISTRY: Registry[type["Embedder"]] = Registry("embedder")

# Pooling strategies for reducing per-residue representations to one vector.
_POOLINGS = ("mean", "max")


@dataclass(frozen=True, eq=False)
class EmbeddingRecord:
    """One peptide's embedding under one model.

    ``eq=False`` because numpy arrays don't support scalar equality; compare
    records via ``model_name``/``peptide_id`` and ``np.allclose`` on ``vector``.
    """

    model_name: str
    sequence: str
    vector: npt.NDArray[np.float32]
    peptide_id: str | None = None

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])


class Embedder(ABC):
    """Model-agnostic interface for producing per-sequence embeddings."""

    #: Stable identifier for the model; used as the cache namespace.
    model_name: str = "unknown"

    def __init__(self, *, pooling: str = "mean") -> None:
        if pooling not in _POOLINGS:
            raise ValidationError(f"pooling must be one of {_POOLINGS}, got {pooling!r}.")
        self.pooling = pooling

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality (D)."""

    @abstractmethod
    def _embed_sequences(self, sequences: Sequence[str]) -> npt.NDArray[np.float32]:
        """Return a ``(len(sequences), dim)`` float32 array. Implemented by subclasses."""

    def embed(self, sequence: str) -> npt.NDArray[np.float32]:
        """Embed a single sequence into a ``(dim,)`` vector."""
        return np.asarray(self.embed_many([sequence])[0], dtype=np.float32)

    def embed_many(self, sequences: Sequence[str]) -> npt.NDArray[np.float32]:
        """Embed many sequences into a ``(N, dim)`` array, with validation."""
        if not sequences:
            return np.empty((0, self.dim), dtype=np.float32)
        cleaned = [s.strip().upper() for s in sequences]
        for s in cleaned:
            if not s:
                raise ValidationError("Cannot embed an empty sequence.")
        out = np.asarray(self._embed_sequences(cleaned), dtype=np.float32)
        if out.shape != (len(cleaned), self.dim):
            raise ValidationError(
                f"{type(self).__name__} returned shape {out.shape}, "
                f"expected {(len(cleaned), self.dim)}."
            )
        return out

    def _pool(self, residue_reps: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Reduce ``(L, D)`` per-residue representations to ``(D,)``."""
        pooled = (
            residue_reps.mean(axis=0)
            if self.pooling == "mean"
            else residue_reps.max(axis=0)
        )
        return np.asarray(pooled, dtype=np.float32)
