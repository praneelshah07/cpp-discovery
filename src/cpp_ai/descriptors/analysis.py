"""Utilities for turning descriptor sets into matrices and ranking features by
discriminative power.

This directly serves the question "which properties best distinguish these
seemingly similar peptides?" Given a group of peptides, it ranks descriptors by
how much they spread across the group — the descriptors that vary most (relative
to their scale) are the ones separating otherwise-similar candidates. This is an
*exploratory* aid; principled feature selection with labels arrives in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from .base import DescriptorSet


@dataclass(frozen=True)
class DescriptorMatrix:
    """A dense descriptor matrix with aligned row/column labels."""

    peptide_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    matrix: npt.NDArray[np.float64]  # shape (n_peptides, n_features)


def descriptor_matrix(descriptor_sets: Sequence[DescriptorSet]) -> DescriptorMatrix:
    """Stack descriptor sets into one matrix over their shared features.

    Only features present in *every* set are included, so the matrix is never
    silently zero-filled. Row order follows the input order.
    """
    if not descriptor_sets:
        raise ValidationError("Need at least one DescriptorSet to build a matrix.")

    shared = set(descriptor_sets[0].feature_names)
    for ds in descriptor_sets[1:]:
        shared &= set(ds.feature_names)
    if not shared:
        raise ValidationError("Descriptor sets share no common features.")

    feature_names = tuple(sorted(shared))
    rows = [ds.vector(feature_names) for ds in descriptor_sets]
    ids = tuple(ds.peptide_id or f"row{i}" for i, ds in enumerate(descriptor_sets))
    return DescriptorMatrix(ids, feature_names, np.vstack(rows))


@dataclass(frozen=True)
class FeatureSpread:
    """Discriminative statistics for a single descriptor across a peptide group."""

    feature: str
    mean: float
    std: float
    coefficient_of_variation: float  # std / |mean|, NaN-safe
    value_range: float               # max - min


def discriminative_ranking(
    descriptor_sets: Sequence[DescriptorSet],
    *,
    by: str = "coefficient_of_variation",
) -> list[FeatureSpread]:
    """Rank descriptors by how strongly they vary across the given peptides.

    Parameters
    ----------
    by:
        Sort key: ``"coefficient_of_variation"`` (scale-relative spread, good for
        comparing features on different units), ``"std"``, or ``"value_range"``.

    Returns features most-discriminative first. Requires >= 2 peptides.
    """
    if len(descriptor_sets) < 2:
        raise ValidationError("Discriminative ranking needs at least 2 peptides.")
    valid_keys = {"coefficient_of_variation", "std", "value_range"}
    if by not in valid_keys:
        raise ValidationError(f"`by` must be one of {sorted(valid_keys)}, got {by!r}.")

    dm = descriptor_matrix(descriptor_sets)
    means = dm.matrix.mean(axis=0)
    stds = dm.matrix.std(axis=0, ddof=1)
    ranges = dm.matrix.max(axis=0) - dm.matrix.min(axis=0)

    spreads: list[FeatureSpread] = []
    for j, feature in enumerate(dm.feature_names):
        mean = float(means[j])
        std = float(stds[j])
        cv = std / abs(mean) if mean != 0 else (0.0 if std == 0 else float("inf"))
        spreads.append(
            FeatureSpread(
                feature=feature,
                mean=mean,
                std=std,
                coefficient_of_variation=cv,
                value_range=float(ranges[j]),
            )
        )

    spreads.sort(key=lambda s: getattr(s, by), reverse=True)
    return spreads
