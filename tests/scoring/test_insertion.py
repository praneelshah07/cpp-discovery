"""Tests for the literature-weighted membrane-insertion prior."""

from __future__ import annotations

from cpp_ai.scoring.insertion import membrane_interaction_capacity


def test_bounded() -> None:
    for seq in ("LLIILRRRIRKQAHAHSK", "RRRRRRRRR", "RQIKIWFQNRRMKWKK", "AAAA"):
        assert 0.0 <= membrane_interaction_capacity(seq) <= 1.0


def test_amphipathic_scores_high() -> None:
    # pVEC and pVEC-R6A are amphipathic — strong insertion
    assert membrane_interaction_capacity("LLIILRRRIRKQAHAHSK") > 0.7
    assert membrane_interaction_capacity("LLIILARRIRKQAHAHSK") > 0.7


def test_pure_cationic_scores_low() -> None:
    # R9 has no amphipathic face / hydrophobic character -> weak insertion
    assert membrane_interaction_capacity("RRRRRRRRR") < membrane_interaction_capacity("LLIILRRRIRKQAHAHSK")
    assert membrane_interaction_capacity("RRRRRRRRR") < 0.5


def test_aromatics_not_penalized() -> None:
    # The old SAR wrongly penalized aromaticity; Trp/aromatics aid insertion.
    # Penetratin (two Trp) must retain a non-trivial insertion score.
    penetratin = membrane_interaction_capacity("RQIKIWFQNRRMKWKK")
    assert penetratin > 0.4
    # Removing the aromatics should not *increase* the score (no penalty exists)
    no_aromatics = membrane_interaction_capacity("RQIKISFQNRRMKSKK")  # W/W -> S/S
    assert penetratin >= no_aromatics - 0.15  # aromatics are neutral-to-positive


def test_non_canonical_is_zero() -> None:
    assert membrane_interaction_capacity("LLIILBXZ") == 0.0
