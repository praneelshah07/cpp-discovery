"""Biological descriptor blocks for interpretable, non-redundant similarity.

The screening index's ~53-feature cosine collapsed several concepts into one
number and over-represented some (charge appeared ~10x across pK scales and
cationic fractions; amphipathicity only twice). This module instead defines a
small set of **biological blocks**, each represented by a *curated,
non-redundant* feature set, so every concept contributes as exactly one
similarity — and the per-block scores are shown separately.

Each block carries a default weight; amphipathicity is up-weighted because it is
biologically central to membrane entry yet was previously under-represented.
Weights are fully user-adjustable at scoring time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiologicalBlock:
    """One biological concept and the (non-redundant) features representing it."""

    name: str
    features: tuple[str, ...]
    default_weight: float
    description: str


# Curated representative features per concept. Deliberately minimal: one strong
# proxy per idea rather than many correlated ones (e.g. a single net-charge
# estimate instead of nine pK-scale variants).
BIOLOGICAL_BLOCKS: tuple[BiologicalBlock, ...] = (
    BiologicalBlock(
        "charge", ("charge_pH7.4_Lehninger",), 1.0,
        "Net charge at pH 7.4 (single representative pK scale).",
    ),
    BiologicalBlock(
        "hydrophobicity", ("gravy_kyte_doolittle",), 1.0,
        "Mean hydrophobicity (GRAVY).",
    ),
    BiologicalBlock(
        "amphipathicity", ("hydrophobic_moment_alpha",), 1.5,
        "Helical hydrophobic moment — up-weighted (key to membrane entry).",
    ),
    BiologicalBlock(
        "structure", ("helix_fraction", "instability_index_biopython"), 1.0,
        "Predicted helicity and stability.",
    ),
    BiologicalBlock(
        "composition", ("frac_group_aromatic", "frac_group_polar_uncharged", "frac_group_special"), 1.0,
        "Residue-group composition beyond charge/hydrophobicity (aromatic, polar, special).",
    ),
    BiologicalBlock(
        "aggregation", ("aggregation_peak_window_hydrophobicity",), 0.75,
        "Aggregation-prone-region proxy (heuristic).",
    ),
    BiologicalBlock(
        "arrangement",
        ("longest_basic_run", "longest_hydrophobic_run", "cationic_aromatic_contacts", "charge_segregation"),
        1.0,
        "Sequence arrangement: residue clustering, cationic-aromatic contacts, charge segregation.",
    ),
)

# Descriptor blocks that must be computed to obtain every feature above.
REQUIRED_DESCRIPTOR_BLOCKS: tuple[str, ...] = (
    "charge", "biopython_props", "hydrophobic_moment", "composition", "aggregation", "arrangement",
)


def all_block_features() -> tuple[str, ...]:
    """Every curated feature name across all biological blocks (deduplicated)."""
    seen: dict[str, None] = {}
    for block in BIOLOGICAL_BLOCKS:
        for f in block.features:
            seen.setdefault(f, None)
    return tuple(seen)
