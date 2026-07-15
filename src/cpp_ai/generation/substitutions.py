"""Substitution strategies: which residues may replace a given residue.

Each strategy is a registered pure function ``residue -> tuple[replacements]``
(never including the original), encoding a different design intent:

* ``all_canonical`` — any of the other 19 canonical residues (exhaustive scan).
* ``conservative`` — biochemically similar residues (positive BLOSUM62 score),
  the safest edits: most likely to preserve fold and function.
* ``charge_preserving`` — residues in the same charge class, so net charge (the
  dominant CPP-uptake driver) is held roughly constant while other properties
  vary.
* ``hydrophobic`` — replace with hydrophobic residues, to probe the effect of
  increasing amphipathicity/membrane affinity.

All strategies emit only canonical residues, matching the platform default that
designs must be genetically encodable for the recombinant CPP-mCherry fusion.
"""

from __future__ import annotations

from typing import Callable

from Bio.Align import substitution_matrices

from ..core.registry import Registry
from ..core.types import CANONICAL_AMINO_ACIDS_ORDERED

#: A strategy maps an original residue to its allowed replacements (not itself).
SubstitutionStrategy = Callable[[str], tuple[str, ...]]

SUBSTITUTION_REGISTRY: Registry[SubstitutionStrategy] = Registry("substitution strategy")

# Charge classes must match the net_charge metric (constraints.net_charge) so
# that charge_preserving genuinely preserves net charge: only K/R are +1 and
# D/E are -1; everything else (including His) is treated as neutral.
_POSITIVE = frozenset("KR")
_NEGATIVE = frozenset("DE")
_HYDROPHOBIC = frozenset("AVLIMFW")

_BLOSUM62 = substitution_matrices.load("BLOSUM62")


def _others(residues: frozenset[str], exclude: str) -> tuple[str, ...]:
    return tuple(sorted(r for r in residues if r != exclude))


def all_canonical(residue: str) -> tuple[str, ...]:
    return tuple(a for a in CANONICAL_AMINO_ACIDS_ORDERED if a != residue)


def conservative(residue: str) -> tuple[str, ...]:
    """Residues with a positive BLOSUM62 substitution score (excluding self)."""
    if residue not in CANONICAL_AMINO_ACIDS_ORDERED:
        return ()
    out = []
    for other in CANONICAL_AMINO_ACIDS_ORDERED:
        if other == residue:
            continue
        try:
            if _BLOSUM62[residue, other] > 0:
                out.append(other)
        except (KeyError, IndexError):
            continue
    return tuple(out)


def charge_preserving(residue: str) -> tuple[str, ...]:
    """Other residues sharing the original's charge class."""
    if residue in _POSITIVE:
        return _others(_POSITIVE, residue)
    if residue in _NEGATIVE:
        return _others(_NEGATIVE, residue)
    # neutral: everything not positive or negative
    neutral = frozenset(CANONICAL_AMINO_ACIDS_ORDERED) - _POSITIVE - _NEGATIVE
    return _others(neutral, residue)


def hydrophobic(residue: str) -> tuple[str, ...]:
    """Replace with a hydrophobic residue (excluding self)."""
    return _others(_HYDROPHOBIC, residue)


SUBSTITUTION_REGISTRY.register("all_canonical", all_canonical)
SUBSTITUTION_REGISTRY.register("conservative", conservative)
SUBSTITUTION_REGISTRY.register("charge_preserving", charge_preserving)
SUBSTITUTION_REGISTRY.register("hydrophobic", hydrophobic)
