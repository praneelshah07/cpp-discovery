"""Core primitives: schema, registry, exceptions, and biological constants.

This subpackage has no heavy dependencies and is safe to import anywhere.
"""

from __future__ import annotations

from .exceptions import (
    CppAiError,
    DuplicateComponentError,
    ProvenanceError,
    RegistryError,
    UnknownComponentError,
    ValidationError,
)
from .registry import Registry
from .schema import Peptide, ProvenanceRecord, compute_peptide_id
from .types import (
    AMBIGUOUS_CODES,
    CANONICAL_AMINO_ACIDS,
    CANONICAL_AMINO_ACIDS_ORDERED,
    ONE_TO_THREE,
    is_canonical_sequence,
    non_canonical_residues,
)

__all__ = [
    # exceptions
    "CppAiError",
    "ValidationError",
    "ProvenanceError",
    "RegistryError",
    "DuplicateComponentError",
    "UnknownComponentError",
    # registry
    "Registry",
    # schema
    "Peptide",
    "ProvenanceRecord",
    "compute_peptide_id",
    # types
    "CANONICAL_AMINO_ACIDS",
    "CANONICAL_AMINO_ACIDS_ORDERED",
    "AMBIGUOUS_CODES",
    "ONE_TO_THREE",
    "is_canonical_sequence",
    "non_canonical_residues",
]
