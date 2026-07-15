"""ProtT5 protein language-model embedder (Rostlab, via HuggingFace transformers).

ProtT5-XL-UniRef50 is an encoder-decoder pLM; we use the encoder half to produce
1024-D per-residue representations, mean/max pooled to a per-sequence vector. It
complements ESM-2 with a different training objective and architecture, which is
useful when comparing embedding spaces in the similarity engine.

transformers and torch are imported lazily. The model is large (~2-3 GB); the
half-precision encoder-only checkpoint is used to keep CPU memory reasonable.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from .base import EMBEDDER_REGISTRY, Embedder

logger = logging.getLogger(__name__)

_MODEL_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
_NON_CANONICAL = re.compile(r"[UZOB]")


class ProtT5Embedder(Embedder):
    """Embed peptides with the ProtT5 encoder."""

    model_name = "prott5_xl_uniref50"

    def __init__(self, *, pooling: str = "mean", batch_size: int = 4) -> None:
        super().__init__(pooling=pooling)
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def dim(self) -> int:
        return 1024

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "ProtT5Embedder requires torch and transformers. Install with: "
                'pip install "cpp-ai[embeddings]"'
            ) from exc

        logger.info("Loading ProtT5 checkpoint %s (first use downloads weights)", _MODEL_ID)
        self._tokenizer = T5Tokenizer.from_pretrained(_MODEL_ID, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(_MODEL_ID)
        model.eval()
        self._model = model

    def _embed_sequences(self, sequences: Sequence[str]) -> npt.NDArray[np.float32]:
        import torch

        self._ensure_loaded()
        out = np.empty((len(sequences), self.dim), dtype=np.float32)

        for start in range(0, len(sequences), self.batch_size):
            chunk = list(sequences[start : start + self.batch_size])
            # ProtT5 expects space-separated residues; rare residues map to X.
            prepared = [" ".join(_NON_CANONICAL.sub("X", s)) for s in chunk]
            encoded = self._tokenizer.batch_encode_plus(
                prepared, add_special_tokens=True, padding="longest", return_tensors="pt"
            )
            with torch.no_grad():
                embedding = self._model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                ).last_hidden_state  # (B, L, 1024)
            for j, seq in enumerate(chunk):
                residue_reps = embedding[j, : len(seq)].cpu().numpy().astype(np.float32)
                out[start + j] = self._pool(residue_reps)
        return out


EMBEDDER_REGISTRY.register("prott5", ProtT5Embedder)
