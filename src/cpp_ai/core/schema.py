"""Standardized, immutable data schema shared by every module.

Design principles encoded here:

* **Raw data is immutable.** Every model is ``frozen`` so an imported record
  cannot be mutated in place. Derived data is produced as *new* objects.
* **Provenance is mandatory.** A :class:`Peptide` cannot exist without a
  :class:`ProvenanceRecord` describing where it came from.
* **Identity is content-addressed.** ``peptide_id`` is derived from the
  sequence, so the same peptide from two databases resolves to the same ID
  (with two provenance records), which is exactly the dedup semantics we want.

These types are deliberately dependency-light (only ``pydantic``/``numpy``) so
the whole downstream stack can share them without importing heavy libraries.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .exceptions import ValidationError
from .types import is_canonical_sequence, non_canonical_residues

_PEPTIDE_ID_PREFIX = "pep_"
_PEPTIDE_ID_HASH_LEN = 16  # hex chars of the sha256 digest kept for the ID


def compute_peptide_id(sequence: str) -> str:
    """Return a stable, content-addressed identifier for a sequence.

    The sequence is upper-cased before hashing so that ``"kla"`` and ``"KLA"``
    map to the same peptide. Collisions are astronomically unlikely at 16 hex
    chars (64 bits) for the peptide-space sizes this platform handles.
    """
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValidationError("Cannot compute a peptide_id for an empty sequence.")
    digest = hashlib.sha256(normalized.encode("ascii")).hexdigest()
    return f"{_PEPTIDE_ID_PREFIX}{digest[:_PEPTIDE_ID_HASH_LEN]}"


class ProvenanceRecord(BaseModel):
    """Immutable record of where a piece of data originated.

    ``imported_at`` is timezone-aware UTC. ``extra`` preserves any
    source-specific bookkeeping (row index, accession, license note) verbatim.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str = Field(..., min_length=1, description="Logical source name, e.g. 'CPPsite3'.")
    original_id: Optional[str] = Field(
        default=None, description="Identifier of the record in its source database, if any."
    )
    source_file: Optional[str] = Field(
        default=None, description="Path or URL the record was imported from."
    )
    file_sha256: Optional[str] = Field(
        default=None, description="SHA-256 of the source file, for exact reproducibility."
    )
    imported_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of import.",
    )
    extra: Mapping[str, Any] = Field(default_factory=dict)


class Peptide(BaseModel):
    """A single peptide with its sequence, content-addressed ID, and provenance.

    ``sequence`` is stored in canonical form (stripped, upper-cased). It is
    validated to contain only A-Z letters so obviously-malformed input is
    rejected at construction, but ambiguity codes (B, J, O, U, X, Z) are
    *allowed* here because they genuinely occur in source databases and must
    not be silently dropped. Use :pyattr:`is_canonical` to gate design-time
    use on the 20 proteinogenic residues.
    """

    model_config = ConfigDict(frozen=True)

    sequence: str = Field(..., min_length=1)
    peptide_id: str = Field(default="")  # auto-filled from sequence if omitted
    provenance: ProvenanceRecord
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("sequence")
    @classmethod
    def _canonicalize_and_check(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Peptide sequence must not be empty.")
        if not normalized.isalpha():
            raise ValueError(
                f"Peptide sequence contains non-letter characters: {value!r}"
            )
        return normalized

    @model_validator(mode="after")
    def _fill_and_verify_id(self) -> "Peptide":
        expected = compute_peptide_id(self.sequence)
        if not self.peptide_id:
            # frozen model: set via object.__setattr__ during validation only.
            object.__setattr__(self, "peptide_id", expected)
        elif self.peptide_id != expected:
            raise ValueError(
                f"peptide_id {self.peptide_id!r} does not match the content hash "
                f"of the sequence (expected {expected!r})."
            )
        return self

    @property
    def length(self) -> int:
        """Number of residues."""
        return len(self.sequence)

    @property
    def is_canonical(self) -> bool:
        """``True`` iff the sequence uses only the 20 canonical amino acids."""
        return is_canonical_sequence(self.sequence)

    @property
    def non_canonical_residues(self) -> tuple[str, ...]:
        """Distinct non-canonical residue codes present, in order of appearance."""
        return non_canonical_residues(self.sequence)

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        *,
        dataset: str,
        original_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **provenance_fields: Any,
    ) -> "Peptide":
        """Convenience constructor that builds provenance inline.

        Keeps call sites terse while still forcing a ``dataset`` to be named,
        so a peptide can never be created without provenance.
        """
        provenance = ProvenanceRecord(
            dataset=dataset, original_id=original_id, **provenance_fields
        )
        return cls(
            sequence=sequence,
            provenance=provenance,
            metadata=dict(metadata or {}),
        )
