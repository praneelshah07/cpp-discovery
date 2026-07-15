"""Tests for substitution strategies."""

from __future__ import annotations

from cpp_ai.generation import (
    SUBSTITUTION_REGISTRY,
    all_canonical,
    charge_preserving,
    conservative,
    hydrophobic,
)


def test_strategies_registered() -> None:
    for name in ("all_canonical", "conservative", "charge_preserving", "hydrophobic"):
        assert name in SUBSTITUTION_REGISTRY


def test_all_canonical_returns_nineteen_excluding_self() -> None:
    result = all_canonical("K")
    assert len(result) == 19
    assert "K" not in result


def test_conservative_excludes_self_and_includes_similar() -> None:
    result = conservative("L")
    assert "L" not in result
    assert "I" in result  # Leu/Ile are conservative (positive BLOSUM62)
    assert "D" not in result  # Leu->Asp is non-conservative


def test_conservative_of_charged_residue() -> None:
    # K<->R is the canonical conservative charged swap.
    assert "R" in conservative("K")


def test_charge_preserving_positive_stays_positive() -> None:
    # K is positive; only R shares that class.
    assert charge_preserving("K") == ("R",)


def test_charge_preserving_negative_stays_negative() -> None:
    assert charge_preserving("D") == ("E",)


def test_charge_preserving_neutral_excludes_charged() -> None:
    result = charge_preserving("A")
    assert "A" not in result
    for charged in "KRDE":
        assert charged not in result
    assert "H" in result  # His is treated as neutral here


def test_hydrophobic_returns_hydrophobic_only() -> None:
    result = hydrophobic("K")
    assert set(result) <= set("AVLIMFW")
    assert "D" not in result and "K" not in result
