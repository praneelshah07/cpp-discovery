"""Amino-acid composition descriptors.

Rationale
---------
The fractional composition over the 20 canonical residues is the most basic
sequence fingerprint. For CPPs it directly exposes the features that matter
most: arginine and lysine content (cationic uptake), tryptophan content
(membrane interaction), and proline content (structural flexibility). Grouped
physicochemical fractions summarize these into interpretable bins.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from .base import register_descriptor
from ..core.types import CANONICAL_AMINO_ACIDS_ORDERED

# Interpretable residue groups for CPP analysis.
_GROUPS: dict[str, frozenset[str]] = {
    "cationic": frozenset("KR"),          # arginine + lysine: primary uptake driver
    "anionic": frozenset("DE"),
    "aromatic": frozenset("FWY"),         # tryptophan-rich CPPs interact with membranes
    "hydrophobic": frozenset("AILMFWV"),
    "polar_uncharged": frozenset("STNQ"),
    "special": frozenset("CGP"),          # proline/glycine flexibility, cysteine
}


@register_descriptor("composition")
def composition(sequence: str) -> Mapping[str, float]:
    """Per-residue fractions plus interpretable group fractions and histidine."""
    n = len(sequence)
    counts = Counter(sequence)
    out: dict[str, float] = {
        f"frac_{aa}": counts.get(aa, 0) / n for aa in CANONICAL_AMINO_ACIDS_ORDERED
    }
    for group_name, residues in _GROUPS.items():
        out[f"frac_group_{group_name}"] = sum(counts.get(a, 0) for a in residues) / n
    # Histidine is conditionally protonatable near physiological pH; surface it.
    out["frac_histidine"] = counts.get("H", 0) / n
    return out
