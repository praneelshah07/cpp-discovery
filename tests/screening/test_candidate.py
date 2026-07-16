"""Tests for ScreenCandidate and helpers."""

from __future__ import annotations

from cpp_ai.screening import ScreenCandidate, charge_toxicity_flag


def test_toxicity_flag_thresholds() -> None:
    assert charge_toxicity_flag(9) == "HIGH-RISK"
    assert charge_toxicity_flag(8) == "HIGH-RISK"
    assert charge_toxicity_flag(6) == "moderate"
    assert charge_toxicity_flag(5) == "lower"
    assert charge_toxicity_flag(-1) == "lower"


def test_candidate_properties() -> None:
    c = ScreenCandidate("RRRRRRRRR", name="R9")
    assert c.net_charge == 9
    assert c.length == 9
    assert c.toxicity_flag == "HIGH-RISK"


def test_endocytic_and_lytic_flags() -> None:
    endo = ScreenCandidate("KWKLFKKI", mechanism="Endocytosis pathway")
    lytic = ScreenCandidate("KWKLFKKI", mechanism="Direct translocation")
    amp = ScreenCandidate("KWKLFKKI", category="Antimicrobial")
    assert endo.is_endocytic and not endo.lytic_risk
    assert lytic.lytic_risk and not lytic.is_endocytic
    assert amp.lytic_risk


def test_with_similarity_and_record() -> None:
    c = ScreenCandidate("KWKLFKKI", name="x").with_similarity(0.876)
    rec = c.to_record()
    assert rec["similarity"] == 0.876
    assert rec["net_charge"] == c.net_charge
    assert rec["toxicity_flag"] == c.toxicity_flag
