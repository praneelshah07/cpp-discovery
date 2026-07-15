"""Content-addressed on-disk cache for embeddings.

Embeddings are deterministic per (model, sequence) but expensive to compute
(CPU inference on a 650M-parameter model is seconds per peptide). This cache
guarantees each vector is computed **once** and reused forever, keyed by a hash
of the model name and sequence so vectors from different models never collide.

Layout: ``<root>/<safe_model_name>/<key>.npy`` (float32). Writes are atomic
(temp file + rename) so an interrupted run cannot leave a corrupt vector.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(model_name: str) -> str:
    return _SAFE_NAME.sub("_", model_name)


def embedding_key(model_name: str, sequence: str) -> str:
    """Content-addressed key for a (model, sequence) pair."""
    normalized = sequence.strip().upper()
    payload = f"{model_name}\x00{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EmbeddingCache:
    """A directory-backed cache mapping (model, sequence) -> float32 vector."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str, sequence: str) -> Path:
        return self.root / _safe(model_name) / f"{embedding_key(model_name, sequence)}.npy"

    def has(self, model_name: str, sequence: str) -> bool:
        return self._path(model_name, sequence).exists()

    def get(self, model_name: str, sequence: str) -> npt.NDArray[np.float32] | None:
        """Return the cached vector, or ``None`` on a miss."""
        path = self._path(model_name, sequence)
        if not path.exists():
            return None
        return np.asarray(np.load(path), dtype=np.float32)

    def put(
        self, model_name: str, sequence: str, vector: npt.NDArray[np.float32]
    ) -> None:
        """Store a vector atomically."""
        path = self._path(model_name, sequence)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Writing through a file handle stops np.save from appending ".npy",
        # keeping the temp name exact for an atomic rename.
        tmp = path.parent / f"{path.name}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            np.save(handle, np.asarray(vector, dtype=np.float32))
        os.replace(tmp, path)

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.npy"))
