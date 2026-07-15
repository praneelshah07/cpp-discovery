"""ESM-2 protein language-model embedder (Meta AI / fair-esm).

ESM-2 is a transformer trained on ~65M protein sequences; its residue
representations encode structural and evolutionary context. We default to
``esm2_t33_650M_UR50D`` — a strong quality/size trade-off that runs on CPU
(no GPU required, just slower) — and expose the tiny ``esm2_t6_8M_UR50D`` for
fast tests and CI.

torch and fair-esm are imported lazily so importing this module (e.g. to read
``ESM2_MODELS``) costs nothing until a model is actually loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from .base import EMBEDDER_REGISTRY, Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ESM2ModelSpec:
    """Metadata for one ESM-2 checkpoint."""

    loader: str      # fair-esm factory function name
    n_layers: int    # final representation layer to read
    dim: int         # embedding dimensionality


# Common ESM-2 checkpoints. `loader` names a function on `esm.pretrained`.
ESM2_MODELS: dict[str, ESM2ModelSpec] = {
    "esm2_t6_8M": ESM2ModelSpec("esm2_t6_8M_UR50D", 6, 320),
    "esm2_t12_35M": ESM2ModelSpec("esm2_t12_35M_UR50D", 12, 480),
    "esm2_t30_150M": ESM2ModelSpec("esm2_t30_150M_UR50D", 30, 640),
    "esm2_t33_650M": ESM2ModelSpec("esm2_t33_650M_UR50D", 33, 1280),
}

_DEFAULT_MODEL = "esm2_t33_650M"


class ESM2Embedder(Embedder):
    """Embed peptides with ESM-2 via mean/max pooling over residue tokens."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        *,
        pooling: str = "mean",
        batch_size: int = 8,
    ) -> None:
        super().__init__(pooling=pooling)
        if model_name not in ESM2_MODELS:
            raise ValidationError(
                f"Unknown ESM-2 model {model_name!r}. "
                f"Available: {sorted(ESM2_MODELS)}"
            )
        self.model_name = model_name
        self._spec = ESM2_MODELS[model_name]
        self.batch_size = batch_size
        self._model: Any = None
        self._batch_converter: Any = None

    @property
    def dim(self) -> int:
        return self._spec.dim

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import esm
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "ESM2Embedder requires torch and fair-esm. Install with: "
                'pip install "cpp-ai[embeddings]"'
            ) from exc

        logger.info("Loading ESM-2 checkpoint %s (first use downloads weights)", self.model_name)
        model, alphabet = getattr(esm.pretrained, self._spec.loader)()
        model.eval()
        self._model = model
        self._batch_converter = alphabet.get_batch_converter()

    def _embed_sequences(self, sequences: Sequence[str]) -> npt.NDArray[np.float32]:
        import torch

        self._ensure_loaded()
        layer = self._spec.n_layers
        out = np.empty((len(sequences), self.dim), dtype=np.float32)

        for start in range(0, len(sequences), self.batch_size):
            chunk = list(sequences[start : start + self.batch_size])
            data = [(str(i), seq) for i, seq in enumerate(chunk)]
            _, _, tokens = self._batch_converter(data)
            with torch.no_grad():
                result = self._model(tokens, repr_layers=[layer], return_contacts=False)
            reps = result["representations"][layer]  # (B, L, D), includes BOS/EOS
            for j, seq in enumerate(chunk):
                # Drop BOS (index 0) and EOS/pad; keep the len(seq) residue tokens.
                residue_reps = reps[j, 1 : len(seq) + 1].cpu().numpy().astype(np.float32)
                out[start + j] = self._pool(residue_reps)
        return out


def _register_esm2_models() -> None:
    """Register a named embedder factory for each ESM-2 checkpoint."""
    for name in ESM2_MODELS:
        EMBEDDER_REGISTRY.register(
            name, _make_factory(name), overwrite=True
        )


def _make_factory(model_name: str) -> type[Embedder]:
    # Registry stores classes; wrap the fixed model_name in a subclass so the
    # registry entry is zero-arg constructible.
    class _Bound(ESM2Embedder):
        def __init__(self, *, pooling: str = "mean", batch_size: int = 8) -> None:
            super().__init__(model_name, pooling=pooling, batch_size=batch_size)

    _Bound.__name__ = f"ESM2Embedder_{model_name}"
    return _Bound


_register_esm2_models()
