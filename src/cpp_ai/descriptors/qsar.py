"""QSAR descriptor sets: multidimensional physicochemical embeddings.

Rationale
---------
QSAR descriptor sets (Z-scales, Kidera factors, VHSE, FASGAI, ...) are the
result of applying dimensionality reduction (PCA / factor analysis) to large
tables of measured amino-acid properties. Each set summarizes a residue as a
handful of orthogonal, interpretable axes (e.g. Z1 ≈ hydrophobicity,
Z2 ≈ bulk/steric, Z3 ≈ polarity), averaged over the peptide.

These are the platform's most powerful tools for **differentiating seemingly
similar peptides**: a conservative substitution that barely moves gross charge
or hydrophobicity can shift several QSAR axes measurably. Computing the full
battery gives feature-selection (Phase 5) and similarity (Phase 4) a rich space
to find the properties that best separate candidates.

Each set is registered as its own block so it can be selected or swapped
independently.
"""

from __future__ import annotations

from typing import Mapping

from peptides import Peptide as PeptidesPeptide

from ..core.registry import Registry
from .base import DESCRIPTOR_REGISTRY, DescriptorFn

# (block name, peptides.Peptide method, feature prefix). Every method returns a
# namedtuple whose ._asdict() gives {field: value}.
_QSAR_SPECS: tuple[tuple[str, str, str], ...] = (
    ("zscales", "z_scales", "Z"),
    ("kidera", "kidera_factors", "KF"),
    ("vhse", "vhse_scales", "VHSE"),
    ("fasgai", "fasgai_vectors", "FASGAI"),
    ("tscales", "t_scales", "T"),
    ("stscales", "st_scales", "ST"),
    ("blosum", "blosum_indices", "BLOSUM"),
    ("cruciani", "cruciani_properties", "CRUCIANI"),
    ("atchley", "atchley_factors", "ATCHLEY"),
    ("ms_whim", "ms_whim_scores", "MSWHIM"),
    ("protfp", "protfp_descriptors", "PROTFP"),
    ("sneath", "sneath_vectors", "SNEATH"),
    ("physical", "physical_descriptors", "PHYS"),
    ("pcp", "pcp_descriptors", "PCP"),
    ("svger", "svger_descriptors", "SVGER"),
    ("vstpv", "vstpv_descriptors", "VSTPV"),
)


def _make_qsar_block(method_name: str, prefix: str) -> DescriptorFn:
    """Build a descriptor block that averages one QSAR set over the sequence."""

    def block(sequence: str) -> Mapping[str, float]:
        pep = PeptidesPeptide(sequence)
        result = getattr(pep, method_name)()
        return {f"{prefix}_{field}": value for field, value in result._asdict().items()}

    block.__name__ = f"qsar_{method_name}"
    block.__doc__ = f"Average {prefix} QSAR descriptor set over the sequence."
    return block


def _register_qsar_blocks(registry: Registry[DescriptorFn]) -> None:
    """Register every QSAR block, skipping any absent from this library version."""
    for block_name, method_name, prefix in _QSAR_SPECS:
        if not hasattr(PeptidesPeptide, method_name):
            continue
        registry.register(block_name, _make_qsar_block(method_name, prefix))


_register_qsar_blocks(DESCRIPTOR_REGISTRY)
