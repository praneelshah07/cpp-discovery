"""Block-wise, distance-aware, calibrated physicochemical similarity.

Fixes three issues with the previous whole-vector cosine:

1. **Block imbalance / redundancy** — each biological block (charge,
   hydrophobicity, amphipathicity, structure, composition, aggregation) yields
   exactly one similarity from a curated, non-redundant feature set, so no
   concept is counted ten times.
2. **Cosine ignores magnitude** — per-block similarity is a Gaussian (RBF) on
   the *standardized Euclidean distance*, so a peptide that is far more extreme
   than the anchor scores low even if its property *pattern* points the same way.
3. **Uninterpretable scale** — the composite is reported alongside a
   library-calibrated **percentile** ("top X% physicochemical match"), and the
   per-block scores are exposed so the user sees *why*.

similarity_block = exp(-d² / 2σ²),  d = RMS standardized distance over the block
composite         = Σ wᵦ·similarity_block / Σ wᵦ
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from ..descriptors import compute_descriptors
from .blocks import BIOLOGICAL_BLOCKS, REQUIRED_DESCRIPTOR_BLOCKS, BiologicalBlock, all_block_features


@dataclass(frozen=True)
class BlockScore:
    """One biological block's similarity and its standardized distance."""

    name: str
    similarity: float
    distance: float


@dataclass(frozen=True)
class PhyschemProfile:
    """Interpretable per-block physicochemical similarity to an anchor."""

    sequence: str
    composite: float
    percentile: float  # fraction of the library scoring <= this (0..1); higher = closer
    blocks: tuple[BlockScore, ...]

    def block(self, name: str) -> float:
        for b in self.blocks:
            if b.name == name:
                return b.similarity
        raise KeyError(name)


class BlockSimilarityIndex:
    """Precomputed block matrices for fast, interpretable anchor scoring."""

    def __init__(
        self,
        sequences: Sequence[str],
        blocks: Sequence[BiologicalBlock] = BIOLOGICAL_BLOCKS,
        *,
        sigma: float = 1.0,
    ) -> None:
        if not sequences:
            raise ValidationError("Need at least one sequence to build the index.")
        self.sequences = list(sequences)
        self.blocks = tuple(blocks)
        self.sigma = sigma
        self._features = all_block_features()
        self._feat_index = {f: i for i, f in enumerate(self._features)}

        rows = []
        for s in self.sequences:
            ds = compute_descriptors(s, blocks=list(REQUIRED_DESCRIPTOR_BLOCKS))
            rows.append([ds[f] for f in self._features])
        matrix = np.asarray(rows, dtype=np.float64)
        self._mean = matrix.mean(axis=0)
        self._std = matrix.std(axis=0)
        self._std[self._std == 0] = 1.0
        self._std_matrix = (matrix - self._mean) / self._std  # (n, F)

        # per-block column indices
        self._cols = {b.name: [self._feat_index[f] for f in b.features] for b in self.blocks}

    def _standardize(self, sequence: str) -> npt.NDArray[np.float64]:
        ds = compute_descriptors(sequence, blocks=list(REQUIRED_DESCRIPTOR_BLOCKS))
        vals = np.asarray([ds[f] for f in self._features], dtype=np.float64)
        return np.asarray((vals - self._mean) / self._std, dtype=np.float64)

    def rank(
        self,
        anchor_sequence: str,
        *,
        weights: Mapping[str, float] | None = None,
        top_k: int | None = None,
    ) -> list[PhyschemProfile]:
        """Score every library peptide against the anchor, per block + composite."""
        a_std = self._standardize(anchor_sequence)
        # weights=None -> block defaults; a dict fully specifies weights
        # (unlisted blocks get 0, so passing {"charge": 1} means charge-only).
        if weights is None:
            w = {b.name: b.default_weight for b in self.blocks}
        else:
            w = {b.name: float(weights.get(b.name, 0.0)) for b in self.blocks}
        total_w = sum(w.values())
        if total_w <= 0:
            raise ValidationError("Sum of block weights must be positive.")

        n = self._std_matrix.shape[0]
        composite = np.zeros(n)
        block_sims: dict[str, npt.NDArray[np.float64]] = {}
        block_dists: dict[str, npt.NDArray[np.float64]] = {}
        for b in self.blocks:
            cols = self._cols[b.name]
            diff = self._std_matrix[:, cols] - a_std[cols]
            dist = np.sqrt(np.mean(diff**2, axis=1))  # RMS standardized distance
            sim = np.exp(-(dist**2) / (2.0 * self.sigma**2))
            block_sims[b.name] = sim
            block_dists[b.name] = dist
            composite += w[b.name] * sim
        composite /= total_w

        # library-calibrated percentile of each composite
        order_vals = np.sort(composite)
        percentiles = np.searchsorted(order_vals, composite, side="right") / n

        profiles = [
            PhyschemProfile(
                sequence=self.sequences[i],
                composite=float(composite[i]),
                percentile=float(percentiles[i]),
                blocks=tuple(
                    BlockScore(b.name, float(block_sims[b.name][i]), float(block_dists[b.name][i]))
                    for b in self.blocks
                ),
            )
            for i in range(n)
        ]
        profiles.sort(key=lambda p: p.composite, reverse=True)
        return profiles[:top_k] if top_k is not None else profiles
