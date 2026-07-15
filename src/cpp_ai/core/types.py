"""Fundamental biological constants and helpers shared across the platform.

This module is the single source of truth for what counts as a "canonical"
amino acid. Downstream phases (generation, optimization) must default to these
20 residues because the intended downstream experiment is a recombinant
CPP-mCherry fusion protein, which can only encode genetically-standard amino
acids. Non-canonical residues are supported elsewhere behind an explicit
opt-in, but they are *never* the default.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

# The 20 canonical (proteinogenic, genetically-encodable) amino acids,
# one-letter codes. Sorted for deterministic iteration.
CANONICAL_AMINO_ACIDS: Final[frozenset[str]] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Ordered tuple for stable indexing (e.g. composition vectors, one-hot).
CANONICAL_AMINO_ACIDS_ORDERED: Final[tuple[str, ...]] = tuple(
    sorted(CANONICAL_AMINO_ACIDS)
)

# Ambiguity / placeholder codes that legitimately appear in sequence databases
# but are NOT usable for design. Kept distinct so importers can flag them
# rather than silently dropping records.
AMBIGUOUS_CODES: Final[frozenset[str]] = frozenset("BJOUXZ")

# One-letter -> three-letter mapping, useful for reports and figures.
ONE_TO_THREE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
        "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
        "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
        "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    }
)


def is_canonical_sequence(sequence: str) -> bool:
    """Return ``True`` iff every character is a canonical amino acid.

    The check is case-sensitive on purpose: callers are expected to
    canonicalize (upper-case, strip) sequences during preprocessing before
    they reach this function, so a lower-case letter here signals a bug
    upstream rather than something to silently accept.
    """
    if not sequence:
        return False
    return all(residue in CANONICAL_AMINO_ACIDS for residue in sequence)


def non_canonical_residues(sequence: str) -> tuple[str, ...]:
    """Return the distinct characters in ``sequence`` that are not canonical.

    Order of first appearance is preserved to make error messages readable.
    """
    seen: dict[str, None] = {}
    for residue in sequence:
        if residue not in CANONICAL_AMINO_ACIDS and residue not in seen:
            seen[residue] = None
    return tuple(seen)
