"""Phase 2: physicochemical & QSAR descriptors.

Computes a broad, multi-scale battery of descriptors for each peptide so that
downstream phases can discover which properties best distinguish candidates:

* geometry, net charge (multi pK-scale), isoelectric point (multi pK-scale)
* hydrophobicity across ~45 published scales + hydrophobic moment (helix/sheet)
* molecular weight, aliphatic index, Boman index, instability, aromaticity
* predicted secondary-structure fractions
* amino-acid composition + interpretable residue-group fractions
* an aggregation-propensity proxy (clearly labelled heuristic)
* QSAR embeddings: Z-scales, Kidera, VHSE, FASGAI, T/ST-scales, BLOSUM,
  Cruciani, Atchley, MS-WHIM, ProtFP, and more

Every descriptor is a registered pure function (see ``DESCRIPTOR_REGISTRY``),
individually testable and swappable. Descriptors require canonical sequences.
"""

from __future__ import annotations

# Import submodules for their registration side effects.
from . import aggregation, composition, hydrophobicity, motifs, physicochemical, qsar  # noqa: F401
from .analysis import (
    DescriptorMatrix,
    FeatureSpread,
    descriptor_matrix,
    discriminative_ranking,
)
from .base import (
    DESCRIPTOR_REGISTRY,
    DescriptorSet,
    compute_descriptors,
    compute_for_peptide,
    register_descriptor,
)

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "DescriptorSet",
    "compute_descriptors",
    "compute_for_peptide",
    "register_descriptor",
    "DescriptorMatrix",
    "FeatureSpread",
    "descriptor_matrix",
    "discriminative_ranking",
]
