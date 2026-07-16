"""Tests for the membrane-lysis (AMP-like) risk heuristic."""

from __future__ import annotations

from cpp_ai.scoring import membrane_lysis_risk

# Known-gentle translocating CPPs vs known membrane-active/lytic amphipaths.
_GENTLE = {
    "pVEC": "LLIILRRRIRKQAHAHSK",
    "pVEC-R6A": "LLIILARRIRKQAHAHSK",
}
_LYTIC = {
    "TP10": "AGYLLGKINLKALAALAKKIL",
    "MAP": "KLALKLALKALKAALKLA",
}


def test_risk_bounded() -> None:
    for seq in list(_GENTLE.values()) + list(_LYTIC.values()):
        assert 0.0 <= membrane_lysis_risk(seq) <= 1.0


def test_gentle_cpps_score_low() -> None:
    for seq in _GENTLE.values():
        assert membrane_lysis_risk(seq) < 0.4


def test_lytic_amphipaths_score_high() -> None:
    for seq in _LYTIC.values():
        assert membrane_lysis_risk(seq) > 0.6


def test_lytic_ranked_above_gentle() -> None:
    # the discriminator amphipathicity alone cannot make: net hydrophobicity does
    assert min(membrane_lysis_risk(s) for s in _LYTIC.values()) > max(
        membrane_lysis_risk(s) for s in _GENTLE.values()
    )


def test_non_canonical_is_safe_default() -> None:
    assert membrane_lysis_risk("LLIILBXZ") == 0.0
