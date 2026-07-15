"""Tests for cpp_ai.core.schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from cpp_ai.core.schema import Peptide, ProvenanceRecord, compute_peptide_id


def test_peptide_id_is_deterministic_and_case_insensitive() -> None:
    assert compute_peptide_id("KLA") == compute_peptide_id("kla")
    assert compute_peptide_id("KLA") == compute_peptide_id(" kla ")


def test_peptide_id_has_prefix() -> None:
    assert compute_peptide_id("KLA").startswith("pep_")


def test_compute_peptide_id_rejects_empty() -> None:
    from cpp_ai.core.exceptions import ValidationError

    with pytest.raises(ValidationError):
        compute_peptide_id("   ")


def _prov() -> ProvenanceRecord:
    return ProvenanceRecord(dataset="unit-test")


def test_peptide_autofills_id() -> None:
    pep = Peptide(sequence="LLIILRRK", provenance=_prov())
    assert pep.peptide_id == compute_peptide_id("LLIILRRK")


def test_peptide_canonicalizes_sequence() -> None:
    pep = Peptide(sequence="  lliilrrk  ", provenance=_prov())
    assert pep.sequence == "LLIILRRK"


def test_peptide_same_sequence_same_id_across_sources() -> None:
    a = Peptide.from_sequence("LLIILRRK", dataset="CPPsite3", original_id="1")
    b = Peptide.from_sequence("lliilrrk", dataset="POSEIDON", original_id="99")
    assert a.peptide_id == b.peptide_id
    assert a.provenance != b.provenance


def test_peptide_rejects_mismatched_explicit_id() -> None:
    with pytest.raises(PydanticValidationError):
        Peptide(sequence="LLIILRRK", peptide_id="pep_deadbeefdeadbeef", provenance=_prov())


def test_peptide_accepts_matching_explicit_id() -> None:
    expected = compute_peptide_id("LLIILRRK")
    pep = Peptide(sequence="LLIILRRK", peptide_id=expected, provenance=_prov())
    assert pep.peptide_id == expected


@pytest.mark.parametrize("bad", ["", "   ", "LLII-RRK", "LL9K", "LL RRK"])
def test_peptide_rejects_non_letter_sequences(bad: str) -> None:
    with pytest.raises(PydanticValidationError):
        Peptide(sequence=bad, provenance=_prov())


def test_peptide_allows_ambiguous_codes_but_flags_them() -> None:
    # Ambiguity codes occur in real databases and must not be dropped,
    # but must be flagged as non-canonical for design-time use.
    pep = Peptide(sequence="LLXIRRK", provenance=_prov())
    assert not pep.is_canonical
    assert pep.non_canonical_residues == ("X",)


def test_peptide_is_frozen() -> None:
    pep = Peptide(sequence="LLIILRRK", provenance=_prov())
    with pytest.raises(PydanticValidationError):
        pep.sequence = "AAAA"  # type: ignore[misc]


def test_peptide_length_property() -> None:
    assert Peptide(sequence="LLIILRRK", provenance=_prov()).length == 8


def test_provenance_requires_dataset() -> None:
    with pytest.raises(PydanticValidationError):
        ProvenanceRecord(dataset="")


def test_provenance_timestamp_is_utc_aware() -> None:
    rec = ProvenanceRecord(dataset="x")
    assert rec.imported_at.tzinfo is not None


def test_from_sequence_forces_provenance() -> None:
    pep = Peptide.from_sequence("KWKLFKKI", dataset="CPPsite3", original_id="42")
    assert pep.provenance.dataset == "CPPsite3"
    assert pep.provenance.original_id == "42"
