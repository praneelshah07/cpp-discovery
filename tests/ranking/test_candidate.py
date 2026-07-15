"""Tests for the RankedCandidate structural guarantee."""

from __future__ import annotations

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.ranking import NearestPeptide, RankedCandidate


def _kwargs(**overrides):
    base = dict(
        sequence="KWKLFKKI",
        overall_score=0.8,
        similarity_score=0.9,
        confidence=0.85,
        mutation_summary="1 substitution from pep_x: K1A",
        reasons=("High CPP likelihood.",),
        strengths=("Cationic.",),
        weaknesses=("Requires validation.",),
        nearest_literature=(NearestPeptide("KWKLFKKI", 1.0),),
        components={"cpp_likelihood": 0.8},
    )
    base.update(overrides)
    return base


def test_valid_construction() -> None:
    c = RankedCandidate(**_kwargs())
    assert c.overall_score == 0.8
    assert c.reasons


def test_empty_reasons_rejected() -> None:
    with pytest.raises(ValidationError):
        RankedCandidate(**_kwargs(reasons=()))


def test_empty_mutation_summary_rejected() -> None:
    with pytest.raises(ValidationError):
        RankedCandidate(**_kwargs(mutation_summary=""))


def test_empty_weaknesses_rejected() -> None:
    with pytest.raises(ValidationError):
        RankedCandidate(**_kwargs(weaknesses=()))
