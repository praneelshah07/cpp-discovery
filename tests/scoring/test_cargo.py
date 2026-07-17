"""Tests for the advisory cargo-delivery evidence axis (not in the core ranking)."""

from __future__ import annotations

from cpp_ai.scoring.cargo import cargo_annotation, cargo_class, cargo_what_if_factor


def test_curated_lookup() -> None:
    assert cargo_class("LLIILRRRIRKQAHAHSK") == "noncovalent_protein"       # pVEC
    assert cargo_class("RRRRRRRRR") == "nucleic_acid_only"                   # R9 (algae protein NEG)
    assert cargo_class("YGRKKRRQRRR") == "intact_covalent_protein"          # TAT


def test_default_is_no_evidence() -> None:
    a = cargo_annotation("MKKLLKKLLKKLL")  # not curated
    assert a.cargo_class == "no_evidence"
    assert a.evidence_source == ""


def test_what_if_factor_ordering() -> None:
    # a demonstrated protein deliverer outweighs a nucleic-acid-only one
    assert cargo_what_if_factor("LLIILRRRIRKQAHAHSK") > cargo_what_if_factor("RRRRRRRRR")


def test_what_if_bounded() -> None:
    for seq in ("LLIILRRRIRKQAHAHSK", "RRRRRRRRR", "MKKLLKKLL"):
        assert 0.0 < cargo_what_if_factor(seq) <= 1.0
