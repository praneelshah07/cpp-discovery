"""Phase 6: constrained candidate generation.

Generate mutational variants of known CPPs (single/double/triple substitutions)
under swappable substitution strategies and biologically-motivated constraints:

* strategies (``SUBSTITUTION_REGISTRY``): ``all_canonical``, ``conservative``,
  ``charge_preserving``, ``hydrophobic``
* constraints (``MutationConstraints``): locked positions/motifs, canonical-only,
  charge bounds, length bounds, custom predicates

Every variant is a :class:`~cpp_ai.core.schema.Peptide` whose metadata records
its parent and exact mutations. Generation defaults to canonical residues so
candidates remain genetically encodable for the recombinant CPP-mCherry fusion.
"""

from __future__ import annotations

from .constraints import Mutation, MutationConstraints, net_charge
from .generator import VariantGenerator
from .substitutions import (
    SUBSTITUTION_REGISTRY,
    SubstitutionStrategy,
    all_canonical,
    charge_preserving,
    conservative,
    hydrophobic,
)

__all__ = [
    "SUBSTITUTION_REGISTRY",
    "SubstitutionStrategy",
    "all_canonical",
    "conservative",
    "charge_preserving",
    "hydrophobic",
    "Mutation",
    "MutationConstraints",
    "net_charge",
    "VariantGenerator",
]
