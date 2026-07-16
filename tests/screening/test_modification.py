"""Tests for tested-form modification classification."""

from __future__ import annotations

from cpp_ai.screening.candidate import ScreenCandidate
from cpp_ai.screening.modification import classify_modifications


def test_free_and_na_are_unmodified() -> None:
    m = classify_modifications("Free", "Free", "NA")
    assert m.modification_class == "none"
    assert m.is_modified is False
    assert m.genetically_encodable is True
    assert m.summary == "none"


def test_amidation_is_terminal_and_not_encodable() -> None:
    m = classify_modifications("Free", "Amidation", "NA")
    assert m.modification_class == "terminal"
    assert m.is_modified is True
    assert m.genetically_encodable is False
    assert "Amidation" in m.summary


def test_stearylation_is_conjugate() -> None:
    m = classify_modifications("Stearylation", "Amidation", "NA")
    assert m.modification_class == "conjugate"  # conjugate outranks terminal
    assert not m.genetically_encodable


def test_fluorophore_is_conjugate() -> None:
    for tag in ("Conjugated with FITC", "Conjugation with fluorescein", "Conjugated with TAMRA"):
        assert classify_modifications(tag, "Free", "NA").modification_class == "conjugate"


def test_noncanonical_chem_mod() -> None:
    m = classify_modifications("Free", "Free", "O = Ornithine")
    assert m.modification_class == "noncanonical"
    assert not m.genetically_encodable


def test_nanoparticle_chem_is_conjugate_not_noncanonical() -> None:
    assert classify_modifications("Free", "Free", "Nanoparticles").modification_class == "conjugate"


def test_class_priority_noncanonical_wins() -> None:
    m = classify_modifications("Acetylation", "Amidation", "O = Ornithine")
    assert m.modification_class == "noncanonical"  # most significant wins


def test_fusion_confidence_grading() -> None:
    # cloneable = full confidence; functional conjugate low; noncanonical lowest
    assert classify_modifications("Free", "Free", "NA").fusion_confidence == 1.0
    assert classify_modifications("Free", "Amidation", "NA").fusion_confidence == 0.9
    assert classify_modifications("Conjugated with FITC", "Free", "NA").fusion_confidence == 0.8
    assert classify_modifications("Stearylation", "Free", "NA").fusion_confidence == 0.4
    assert classify_modifications("Free", "Free", "O = Ornithine").fusion_confidence == 0.15


def test_fusion_confidence_takes_the_worst() -> None:
    # fluorescein (0.8) + amidation (0.9) -> min 0.8
    assert classify_modifications(
        "Conjugation with fluorescein", "Amidation", "NA"
    ).fusion_confidence == 0.8


def test_candidate_exposes_modification() -> None:
    c = ScreenCandidate(
        sequence="LLIILRRRIRKQAHAHSK", name="pVEC-like",
        n_term_mod="Conjugation with fluorescein", c_term_mod="Amidation", chem_mod="NA",
    )
    assert c.modification.modification_class == "conjugate"
    assert c.modification.genetically_encodable is False
    # a plain candidate defaults to encodable
    plain = ScreenCandidate(sequence="LLIILRRRIRKQAHAHSK", name="bare")
    assert plain.modification.genetically_encodable is True
