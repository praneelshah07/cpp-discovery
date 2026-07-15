"""Tests for the evidence-ledger record type."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cpp_ai.core.schema import compute_peptide_id
from cpp_ai.evidence import Citation, EvidenceEntry


def _entry(**over: object) -> EvidenceEntry:
    base: dict[str, object] = dict(
        peptide_name="R9",
        sequence="RRRRRRRRR",
        organism="mammalian",
        outcome="success",
        citation=Citation(title="Some paper", doi="10.1000/xyz"),
    )
    base.update(over)
    return EvidenceEntry(**base)  # type: ignore[arg-type]


def test_peptide_id_is_content_addressed() -> None:
    e = _entry()
    assert e.peptide_id == compute_peptide_id("RRRRRRRRR")


def test_same_sequence_two_papers_same_id() -> None:
    a = _entry(citation=Citation(title="A", doi="10.1/a"))
    b = _entry(citation=Citation(title="B", doi="10.2/b"))
    assert a.peptide_id == b.peptide_id  # aggregation semantics the SAR layer needs


def test_sequence_is_canonicalized() -> None:
    assert _entry(sequence=" rrrrrrrrr ").sequence == "RRRRRRRRR"


def test_non_letter_sequence_rejected() -> None:
    with pytest.raises(ValidationError):
        _entry(sequence="RRR-9")


def test_wrong_explicit_peptide_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _entry(peptide_id="pep_deadbeefdeadbeef")


def test_citation_requires_something_resolvable() -> None:
    doi_only = Citation(title="t", doi="10.1/x")
    url_only = Citation(title="t", url="https://example.org/paper")
    assert doi_only.resolvable() == "https://doi.org/10.1/x"
    assert url_only.resolvable() == "https://example.org/paper"


def test_doi_prefix_is_stripped() -> None:
    assert Citation(title="t", doi="https://doi.org/10.1/x").doi == "10.1/x"


def test_is_positive_logic() -> None:
    assert _entry(outcome="success", toxicity="none").is_positive
    assert _entry(outcome="partial").is_positive
    assert not _entry(outcome="fail").is_positive
    # enters cells but toxic -> not a usable win
    assert not _entry(outcome="success", toxicity="toxic").is_positive
