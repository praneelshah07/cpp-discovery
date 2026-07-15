"""Tests for the explanation rules."""

from __future__ import annotations

from cpp_ai.ranking import NearestPeptide
from cpp_ai.ranking.explain import (
    VALIDATION_CAVEAT,
    Evidence,
    RankingThresholds,
    build_reasons,
    build_strengths,
    build_weaknesses,
)

_TH = RankingThresholds()


def _ev(**overrides) -> Evidence:
    base = dict(
        sequence="LLIILRRRIRKQAHAHSK",
        net_charge=6,
        length=18,
        is_canonical=True,
        cysteine_count=0,
        mutation_summary="1 substitution(s) from pep_x: K11R",
    )
    base.update(overrides)
    return Evidence(**base)


def test_reasons_never_empty() -> None:
    assert build_reasons(_ev(), _TH)  # even with no strong signals


def test_reason_high_cpp() -> None:
    reasons = build_reasons(_ev(cpp_probability=0.9), _TH)
    assert any("CPP likelihood" in r for r in reasons)


def test_reason_high_similarity() -> None:
    reasons = build_reasons(_ev(similarity_score=0.95, reference_id="pVEC"), _TH)
    assert any("similarity to pVEC" in r for r in reasons)


def test_reason_novelty() -> None:
    nearest = (NearestPeptide("AAAA", 0.2, peptide_id="famous"),)
    reasons = build_reasons(_ev(nearest=nearest), _TH)
    assert any("Novel" in r for r in reasons)


def test_strength_favorable_charge() -> None:
    strengths = build_strengths(_ev(net_charge=6), _TH)
    assert any("cationic net charge" in s for s in strengths)


def test_strength_expression_compatible() -> None:
    strengths = build_strengths(_ev(is_canonical=True, length=18), _TH)
    assert any("expression-compatible" in s for s in strengths)


def test_weaknesses_always_include_caveat() -> None:
    weaknesses = build_weaknesses(_ev(), _TH)
    assert weaknesses[-1] == VALIDATION_CAVEAT


def test_weakness_non_canonical() -> None:
    weaknesses = build_weaknesses(_ev(is_canonical=False), _TH)
    assert any("non-canonical" in w for w in weaknesses)


def test_weakness_excessive_charge() -> None:
    weaknesses = build_weaknesses(_ev(net_charge=12), _TH)
    assert any("exceeds the recommended" in w for w in weaknesses)


def test_weakness_high_uncertainty() -> None:
    weaknesses = build_weaknesses(_ev(uncertainty=0.8), _TH)
    assert any("uncertainty" in w for w in weaknesses)


def test_weakness_cysteine_load() -> None:
    weaknesses = build_weaknesses(_ev(cysteine_count=4), _TH)
    assert any("cysteine" in w.lower() for w in weaknesses)


def test_weakness_aggregation() -> None:
    weaknesses = build_weaknesses(_ev(aggregation_peak=3.0), _TH)
    assert any("aggregation" in w for w in weaknesses)
