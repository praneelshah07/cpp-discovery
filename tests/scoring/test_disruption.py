"""Tests for the trained membrane-disruption prior (Box 3)."""

from __future__ import annotations

from cpp_ai.scoring.disruption import (
    is_trained_model_available,
    hemolysis_prior,
)


def test_bounded() -> None:
    for seq in ("LLIILRRRIRKQAHAHSK", "GIGAVLKVLTTGLPALISWIKRKRQQ", "YGRKKRRQRRR"):
        assert 0.0 <= hemolysis_prior(seq) <= 1.0


def test_non_canonical_is_zero() -> None:
    assert hemolysis_prior("LLIILBXZ") == 0.0


def test_lytic_ranked_above_gentle() -> None:
    # The whole point: a real hemolytic peptide must out-score a gentle CPP.
    # (Skips gracefully if the trained model isn't present — heuristic fallback
    # is validated separately in test_lysis.)
    if not is_trained_model_available():
        return
    melittin = hemolysis_prior("GIGAVLKVLTTGLPALISWIKRKRQQ")
    pvec = hemolysis_prior("LLIILRRRIRKQAHAHSK")
    assert melittin > 0.7  # trained model catches melittin (heuristic did not)
    assert pvec < 0.3
    assert melittin > pvec


def test_fallback_when_no_model(monkeypatch) -> None:
    # With the model forced unavailable, it must fall back to the heuristic (>=0).
    import cpp_ai.scoring.disruption as disruption

    monkeypatch.setattr(disruption, "_cache", {"model": None})
    val = disruption.hemolysis_prior("KLALKLALKALKAALKLA")
    assert 0.0 <= val <= 1.0
