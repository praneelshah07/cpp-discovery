"""Descriptor framework: registry, the ``DescriptorSet`` value object, and the
top-level ``compute_descriptors`` entry point.

Design
------
Every descriptor is a **registered pure function** ``str -> dict[str, float]``.
Returning a mapping (rather than a bare float) lets scalar descriptors
(``{"net_charge": 3.0}``) and multi-dimensional descriptor *blocks*
(45 hydrophobicity scales, a 10-D Kidera vector, ...) share one uniform,
swappable interface. Blocks are selected by name from configuration.

Biological guard
----------------
Descriptors are only computed for sequences over the 20 canonical amino acids.
Non-canonical residues (B/J/O/U/X/Z) make properties like isoelectric point or
GRAVY ill-defined — biopython raises on them and ``peptides`` silently returns
misleading values — so we refuse them explicitly rather than emit numbers a
researcher might trust. Such peptides are flagged at import; filter to canonical
before computing descriptors.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from ..core.exceptions import ValidationError
from ..core.registry import Registry
from ..core.schema import Peptide
from ..core.types import is_canonical_sequence

logger = logging.getLogger(__name__)

#: A descriptor block: sequence -> {feature_name: value}. Registered by name.
DescriptorFn = Callable[[str], Mapping[str, float]]

DESCRIPTOR_REGISTRY: Registry[DescriptorFn] = Registry("descriptor block")


def register_descriptor(name: str) -> Callable[[DescriptorFn], DescriptorFn]:
    """Decorator registering a descriptor block under ``name``."""
    result = DESCRIPTOR_REGISTRY.register(name)
    assert callable(result)  # decorator form always returns a callable
    return result  # type: ignore[return-value]


def _finite_float(value: Any, *, feature: str) -> float:
    """Coerce a library return value to a finite Python float.

    Guards against NaN/inf leaking into the store or ML matrices, which would
    silently corrupt downstream distance and model computations.
    """
    out = float(value)  # also unwraps numpy scalars
    if not math.isfinite(out):
        raise ValidationError(f"Descriptor {feature!r} produced a non-finite value: {out!r}")
    return out


class DescriptorSet(BaseModel):
    """Immutable bundle of computed descriptors for one peptide."""

    model_config = ConfigDict(frozen=True)

    sequence: str
    peptide_id: str | None = None
    values: Mapping[str, float]
    blocks: tuple[str, ...]

    def __getitem__(self, feature: str) -> float:
        return self.values[feature]

    def __contains__(self, feature: object) -> bool:
        return feature in self.values

    @property
    def feature_names(self) -> tuple[str, ...]:
        """All feature names, sorted for deterministic ordering."""
        return tuple(sorted(self.values))

    def vector(self, feature_names: Sequence[str] | None = None) -> npt.NDArray[np.float64]:
        """Return descriptor values as a float vector.

        ``feature_names`` fixes the column order (essential for ML matrices);
        if omitted, the sorted feature order is used. Missing features raise,
        because a silently zero-filled feature would bias a model.
        """
        names = tuple(feature_names) if feature_names is not None else self.feature_names
        missing = [n for n in names if n not in self.values]
        if missing:
            raise KeyError(f"DescriptorSet is missing requested features: {missing}")
        return np.array([self.values[n] for n in names], dtype=float)


def compute_descriptors(
    sequence: str,
    *,
    blocks: Iterable[str] | None = None,
    peptide_id: str | None = None,
    require_canonical: bool = True,
) -> DescriptorSet:
    """Compute descriptor blocks for ``sequence``.

    Parameters
    ----------
    blocks:
        Names of registered blocks to run. ``None`` runs every registered block
        (the full battery — ideal for finding which properties discriminate
        similar peptides).
    require_canonical:
        If ``True`` (default), reject sequences with non-canonical residues.
    """
    seq = sequence.strip().upper()
    if not seq:
        raise ValidationError("Cannot compute descriptors for an empty sequence.")
    if require_canonical and not is_canonical_sequence(seq):
        raise ValidationError(
            f"Descriptors require canonical amino acids; sequence {sequence!r} "
            f"contains non-canonical residues. Filter to canonical peptides first."
        )

    selected = tuple(blocks) if blocks is not None else DESCRIPTOR_REGISTRY.names()
    values = dict(_compute_descriptor_values(seq, selected))

    return DescriptorSet(
        sequence=seq, peptide_id=peptide_id, values=values, blocks=selected
    )


@lru_cache(maxsize=256)
def blocks_for(features: tuple[str, ...]) -> tuple[str, ...]:
    """Minimal set of registered blocks that together emit all of ``features``.

    The full battery has ~210 features across many blocks, but most callers use
    only a handful. Because each block is a pure function of the sequence and
    feature names are globally unique, computing only the blocks that carry the
    requested features yields values **byte-identical** to running the whole
    battery — the other blocks are simply skipped. The block→feature mapping is
    fixed, so we probe it once with a canonical reference sequence and cache.
    """
    wanted = set(features)
    probe = "LLIILRRRIRKQAHAHSK"  # canonical reference; block feature keys are sequence-independent
    return tuple(
        name
        for name in DESCRIPTOR_REGISTRY.names()
        if wanted.intersection(compute_descriptors(probe, blocks=(name,)).values)
    )


@lru_cache(maxsize=131072)
def _compute_one_block(seq: str, block_name: str) -> tuple[tuple[str, float], ...]:
    """Memoized single descriptor block for one sequence.

    Caching at the **block** level (not the block-*tuple* level) is what removes
    the last of the redundant work: a block such as ``hydrophobic_moment`` is
    requested inside many different block-sets (the algae SAR, the mechanistic
    insertion term, the hemolysis model, ...), and this ensures it is computed
    exactly once per sequence and reused across all of them. Values are
    byte-for-byte identical to computing the block inline.
    """
    fn = DESCRIPTOR_REGISTRY.get(block_name)
    return tuple((feature, _finite_float(raw, feature=feature)) for feature, raw in fn(seq).items())


@lru_cache(maxsize=16384)
def _compute_descriptor_values(
    seq: str, selected: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    """Memoized core of :func:`compute_descriptors`, composed from per-block caches.

    Descriptor blocks are pure functions of the (canonical, upper-cased) sequence,
    yet the pipeline requests overlapping block-sets for the same peptides many
    times per run (library standardization, the algae-fit scorer, the mechanistic
    insertion term, hemolysis featurization, ...). This composes the requested
    ``selected`` blocks from :func:`_compute_one_block`, so each block is computed
    once per sequence and shared across every block-set. The outer cache also
    short-circuits repeated identical ``(sequence, blocks)`` requests. Returned as
    a tuple of items so the cached object is immutable; callers rebuild a ``dict``.
    """
    values: dict[str, float] = {}
    for block_name in selected:
        for feature, val in _compute_one_block(seq, block_name):
            if feature in values:
                raise ValidationError(
                    f"Duplicate feature name {feature!r} emitted by block "
                    f"{block_name!r}; feature names must be globally unique."
                )
            values[feature] = val
    return tuple(values.items())


def compute_for_peptide(
    peptide: Peptide,
    *,
    blocks: Iterable[str] | None = None,
    require_canonical: bool = True,
) -> DescriptorSet:
    """Convenience wrapper carrying the peptide's ``peptide_id`` through."""
    return compute_descriptors(
        peptide.sequence,
        blocks=blocks,
        peptide_id=peptide.peptide_id,
        require_canonical=require_canonical,
    )
